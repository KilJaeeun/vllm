#pragma once
/*
 * Fused PACA GEMM Kernel v3 — warp-cooperative GEMV + PACA
 *
 *   Y[m,n] = Σ_k X[m,k] * W_T[n,k]  +  Σ_r X[m,K-R+r] * P[aid[m],r,n] * scale
 *
 * Key insight: W is pre-transposed to [N, K] so that the K-reduction reads
 * contiguous memory (W_T[n, :] is one row).  Each warp of 32 threads cooperates
 * on the K-reduction for ONE output column via vectorised coalesced loads +
 * warp shuffle.  This saturates HBM bandwidth even for M=1.
 *
 *   Grid :  (ceil(N / WARPS_PER_BLOCK), M)
 *   Block:  (32 * WARPS_PER_BLOCK,)          — 32 lanes per warp
 *
 * Each warp:
 *   1. reads X[m, k] and W_T[n, k] in 128-bit coalesced chunks (VEC=8)
 *   2. FMA accumulates, then warp shuffle to sum across 32 lanes
 *   3. lane 0 adds PACA adapter contribution (R is small, serial is fine)
 *   4. lane 0 stores final result to Y[m, n]
 *
 * Weight layouts:
 *   W_T: [N, K]  row-major  (transposed base weight, contiguous along K)
 *   P  : [num_adapters, R, N]  row-major  (native layout)
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float h2f(__half v) { return __half2float(v); }
__device__ __forceinline__ __half f2h(float v) { return __float2half(v); }

template <int N>
struct alignas(sizeof(__half) * N) hvec { __half data[N]; };

template <int WARPS_PER_BLOCK>
__global__ void fused_paca_warp_kernel(
    __half*       __restrict__ Y,      // [M, N]
    const __half* __restrict__ X,      // [M, K]
    const __half* __restrict__ W_T,    // [N, K]  ← transposed!
    const __half* __restrict__ P,      // [A, R, N]
    const int64_t* __restrict__ aids,  // [M]
    int64_t K, int64_t N, int64_t R,
    float scale)
{
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;

    const int64_t m = blockIdx.y;
    const int64_t n = (int64_t)blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (n >= N) return;

    const __half* x_row = X + m * K;
    const __half* w_row = W_T + n * K;            // W_T[n, :] — contiguous!

    float acc = 0.f;

    /* ---- Base GEMM: coalesced K-reduction across 32 lanes ---- */
    /*  lane_id*VEC, lane_id*VEC + 32*VEC, ...                     */
    /*  32 lanes × VEC=8 = 256 elements per loop step              */
    constexpr int VEC = 8;
    constexpr int STRIDE = 32 * VEC;              // 256

    for (int64_t k = lane_id * VEC; k < K; k += STRIDE) {
        hvec<VEC> xv = *reinterpret_cast<const hvec<VEC>*>(x_row + k);
        hvec<VEC> wv = *reinterpret_cast<const hvec<VEC>*>(w_row + k);
        #pragma unroll
        for (int v = 0; v < VEC; ++v)
            acc = __fmaf_rn(h2f(xv.data[v]), h2f(wv.data[v]), acc);
    }

    /* ---- warp shuffle reduction ---- */
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, offset);

    /* ---- PACA adapter: parallelize R across all 32 lanes ---- */
    {
        int64_t aid = aids[m];
        float pa = 0.f;
        if (aid >= 0) {
            const __half* xp = x_row + K - R;
            const __half* pp = P + aid * R * N + n;   // P[aid, 0, n], stride N
            for (int64_t r = lane_id; r < R; r += 32)
                pa = __fmaf_rn(h2f(xp[r]), h2f(pp[r * N]), pa);
            /* warp reduce the adapter partial sums */
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                pa += __shfl_down_sync(0xffffffff, pa, off);
        }
        if (lane_id == 0) {
            acc += pa * scale;
            Y[m * N + n] = f2h(acc);
        }
    }
}

/* ---- launcher ---- */
void launch_fused_paca_gemm(
    __half* Y, const __half* X, const __half* W_T, const __half* P,
    const int64_t* aids,
    int64_t M, int64_t K, int64_t N, int64_t R,
    float scale, cudaStream_t stream)
{
    constexpr int WARPS = 8;                       // 8 warps = 256 threads
    dim3 grid((N + WARPS - 1) / WARPS, M);
    dim3 block(32 * WARPS);

    fused_paca_warp_kernel<WARPS>
        <<<grid, block, 0, stream>>>(Y, X, W_T, P, aids, K, N, R, scale);
}
