/**
 * Fused PACA v2 — Tensor-core GEMM + PACA adapter epilogue
 *
 * Y[m,n] = Σ_k X[m,k] * W[k,n]  +  Σ_r X[m,K-R+r] * P[aid[m],r,n] * scale
 *
 * Uses nvcuda::wmma (m16n16k16 FP16 tensor cores) for the base GEMM.
 * Adapter is fused into the epilogue — zero extra kernel launches.
 *
 * Tile: BM=128, BN=128, BK=32.  8 warps (256 threads) per block.
 * Each warp computes a 32×64 output sub-tile (2×4 wmma 16×16 tiles).
 */

#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

/* ---------- Tile configuration ---------- */
static constexpr int TC_BM = 128;
static constexpr int TC_BN = 128;
static constexpr int TC_BK = 32;

static constexpr int TC_WM = 16;   // wmma tile M
static constexpr int TC_WN = 16;   // wmma tile N
static constexpr int TC_WK = 16;   // wmma tile K

/* Warp layout inside block: 4 rows × 2 cols = 8 warps = 256 threads */
static constexpr int TC_WARPS_M = 4;
static constexpr int TC_WARPS_N = 2;
static constexpr int TC_NWARPS  = TC_WARPS_M * TC_WARPS_N;

/* Each warp owns a 32×64 output sub-tile */
static constexpr int TC_WTM = TC_BM / TC_WARPS_M;   // 32
static constexpr int TC_WTN = TC_BN / TC_WARPS_N;   // 64

/* wmma tiles per warp */
static constexpr int TC_NTM = TC_WTM / TC_WM;        // 2
static constexpr int TC_NTN = TC_WTN / TC_WN;        // 4

/* h2f / f2h defined in fused_paca_kernel.cuh */

/* ================================================================== */
/*  Kernel                                                             */
/* ================================================================== */
__global__ void __launch_bounds__(TC_NWARPS * 32)
fused_paca_tc_kernel(
    const half*  __restrict__ X,       // [M, K]
    const half*  __restrict__ W,       // [K, N]
    const half*  __restrict__ P,       // [A, R, N]
    const int64_t* __restrict__ aids,  // [M]
    half*        __restrict__ Y,       // [M, N]
    int64_t M, int64_t K, int64_t N, int64_t R,
    float scale)
{
    const int bm = blockIdx.y * TC_BM;
    const int bn = blockIdx.x * TC_BN;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int wm_id   = warp_id / TC_WARPS_N;   // 0..3
    const int wn_id   = warp_id % TC_WARPS_N;   // 0..1

    /* ---------- shared memory for A and B tiles ---------- */
    __shared__ half smem_X[TC_BM * TC_BK];     // 8 KB
    __shared__ half smem_W[TC_BK * TC_BN];     // 8 KB

    /* ---------- accumulator fragments (per warp, in registers) ---------- */
    wmma::fragment<wmma::accumulator, TC_WM, TC_WN, TC_WK, float>
        acc[TC_NTM][TC_NTN];

    #pragma unroll
    for (int i = 0; i < TC_NTM; ++i)
        #pragma unroll
        for (int j = 0; j < TC_NTN; ++j)
            wmma::fill_fragment(acc[i][j], 0.0f);

    /* ================ Base GEMM mainloop ================ */
    const int nthreads = TC_NWARPS * 32;  // 256
    const int elems_X  = TC_BM * TC_BK;   // 4096
    const int elems_W  = TC_BK * TC_BN;   // 4096

    for (int64_t k_off = 0; k_off < K; k_off += TC_BK) {

        /* --- cooperative global → shared memory load --- */
        for (int i = threadIdx.x; i < elems_X; i += nthreads) {
            int r = i / TC_BK, c = i % TC_BK;
            int gm = bm + r;
            int64_t gk = k_off + c;
            smem_X[i] = (gm < M && gk < K) ? X[gm * K + gk] : f2h(0.f);
        }
        for (int i = threadIdx.x; i < elems_W; i += nthreads) {
            int r = i / TC_BN, c = i % TC_BN;
            int64_t gk = k_off + r;
            int gn = bn + c;
            smem_W[i] = (gk < K && gn < N) ? W[gk * N + gn] : f2h(0.f);
        }
        __syncthreads();

        /* --- wmma matmul on shared memory tiles --- */
        #pragma unroll
        for (int kk = 0; kk < TC_BK; kk += TC_WK) {
            #pragma unroll
            for (int tm = 0; tm < TC_NTM; ++tm) {
                wmma::fragment<wmma::matrix_a, TC_WM, TC_WN, TC_WK,
                               half, wmma::row_major> a_frag;
                int a_row = wm_id * TC_WTM + tm * TC_WM;
                wmma::load_matrix_sync(a_frag,
                                       smem_X + a_row * TC_BK + kk, TC_BK);
                #pragma unroll
                for (int tn = 0; tn < TC_NTN; ++tn) {
                    wmma::fragment<wmma::matrix_b, TC_WM, TC_WN, TC_WK,
                                   half, wmma::row_major> b_frag;
                    int b_col = wn_id * TC_WTN + tn * TC_WN;
                    wmma::load_matrix_sync(b_frag,
                                           smem_W + kk * TC_BN + b_col, TC_BN);
                    wmma::mma_sync(acc[tm][tn], a_frag, b_frag, acc[tm][tn]);
                }
            }
        }
        __syncthreads();
    }

    /* ================ Epilogue: store + PACA adapter ================ */
    /*
     * Strategy: for each of the 8 wmma fragments this warp owns,
     *   1) store_matrix_sync → per-warp smem buffer (256 floats = 1 KB)
     *   2) each of 32 threads processes 8 elements:
     *      read smem, add adapter contribution, write to Y in global memory
     *
     * Reuse smem_X (8 KB) as float[2048], giving each warp 256 floats.
     */
    float* smem_epi = reinterpret_cast<float*>(smem_X);   // 8 KB → 2048 floats

    #pragma unroll
    for (int tm = 0; tm < TC_NTM; ++tm) {
        #pragma unroll
        for (int tn = 0; tn < TC_NTN; ++tn) {

            const int frag_m = bm + wm_id * TC_WTM + tm * TC_WM;
            const int frag_n = bn + wn_id * TC_WTN + tn * TC_WN;

            /* 1) fragment → smem */
            wmma::store_matrix_sync(
                smem_epi + warp_id * (TC_WM * TC_WN),
                acc[tm][tn],
                TC_WN,
                wmma::mem_row_major);
            __syncwarp();

            /* 2) smem → add adapter → global */
            #pragma unroll
            for (int idx = lane_id; idx < TC_WM * TC_WN; idx += 32) {
                const int lm = idx / TC_WN;
                const int ln = idx % TC_WN;
                const int m = frag_m + lm;
                const int n = frag_n + ln;

                if (m < M && n < N) {
                    float val = smem_epi[warp_id * (TC_WM * TC_WN) + idx];

                    /* ---- PACA adapter ---- */
                    int64_t aid = aids[m];
                    if (aid >= 0) {
                        const half* xp = X + (int64_t)m * K + K - R;
                        const half* pp = P + aid * R * N + n;   // stride N in R
                        float pa = 0.f;
                        for (int64_t r = 0; r < R; ++r)
                            pa += h2f(xp[r]) * h2f(pp[r * N]);
                        val += pa * scale;
                    }

                    Y[(int64_t)m * N + n] = f2h(val);
                }
            }
            __syncwarp();
        }
    }
}

/* ================================================================== */
/*  Host-side launcher                                                 */
/* ================================================================== */
void launch_fused_paca_tc(
    half* Y, const half* X, const half* W, const half* P,
    const int64_t* aids,
    int64_t M, int64_t K, int64_t N, int64_t R,
    float scale, cudaStream_t stream)
{
    dim3 grid((N + TC_BN - 1) / TC_BN,
              (M + TC_BM - 1) / TC_BM);
    dim3 block(TC_NWARPS * 32);   // 256

    fused_paca_tc_kernel<<<grid, block, 0, stream>>>(
        X, W, P, aids, Y, M, K, N, R, scale);
}
