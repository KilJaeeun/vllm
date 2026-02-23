/**
 * build_x_repeated_v2_kernel — 128-bit vectorized version (uint4 = 8 × fp16).
 *
 * Same semantics as v1 but uses 128-bit coalesced vector loads/stores
 * (LDG.128 / STG.128) to maximize HBM bandwidth utilization.
 *
 * Requires K, R to be multiples of 8 (always true in practice:
 *   K ∈ {1024, 2048, 4096, 8192, ...}, R ∈ {8, 16, 32, 64}).
 *
 * Grid:  (ceil(K_ext/8 / 256), M)
 * Block: 256 threads
 * Each thread processes one uint4 (8 halves = 16 bytes) per iteration.
 */

#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

static constexpr int BXR_V2_THREADS = 256;
static constexpr int BXR_V2_VEC     = 8;     /* halves per vector (128 bits) */

__global__ void build_x_repeated_v2_kernel(
    __half*        __restrict__ X_repeated,  // [M, K_ext]
    const __half*  __restrict__ X,           // [M, K]
    const int64_t* __restrict__ adapter_ids, // [M]
    int64_t M, int64_t K, int64_t K_ext,
    int64_t R, int64_t A)
{
    const int64_t m = blockIdx.y;
    if (m >= M) return;

    const int64_t aid = adapter_ids[m];
    const int64_t adp_start = K + aid * R;          /* first col of adapter slot */

    /* Row pointers reinterpreted as uint4 (128 bits = 8 halves) */
    const uint4* x_vec   = reinterpret_cast<const uint4*>(X + m * K);
    uint4*       out_vec = reinterpret_cast<uint4*>(X_repeated + m * K_ext);

    /* Vector counts (all exact divisions since K, R, K_ext are multiples of 8) */
    const int64_t K_vecs     = K >> 3;               /* K / 8            */
    const int64_t ext_vecs   = K_ext >> 3;           /* K_ext / 8        */
    const int64_t adp_vec_s  = adp_start >> 3;       /* adp_start / 8    */
    const int64_t R_vecs     = R >> 3;               /* R / 8            */

    const uint4 zero4 = make_uint4(0u, 0u, 0u, 0u);

    /* Adapter source: X[m, K-R : K]  (last R elements of input row) */
    const uint4* adp_src = reinterpret_cast<const uint4*>(X + m * K + K - R);

    for (int64_t vi = blockIdx.x * BXR_V2_THREADS + threadIdx.x;
         vi < ext_vecs;
         vi += (int64_t)gridDim.x * BXR_V2_THREADS)
    {
        uint4 val;
        if (vi < K_vecs) {
            /* Base region: copy from input */
            val = x_vec[vi];
        } else if (vi >= adp_vec_s && vi < adp_vec_s + R_vecs) {
            /* Adapter region: copy adapter slice from input tail */
            val = adp_src[vi - adp_vec_s];
        } else {
            /* Zero padding (other adapter slots) */
            val = zero4;
        }
        out_vec[vi] = val;
    }
}

/* Host launcher */
void launch_build_x_repeated_v2(
    __half* X_repeated, const __half* X, const int64_t* adapter_ids,
    int64_t M, int64_t K, int64_t K_ext, int64_t R, int64_t A,
    cudaStream_t stream)
{
    const int64_t ext_vecs = K_ext >> 3;
    int grid_x = (int)((ext_vecs + BXR_V2_THREADS - 1) / BXR_V2_THREADS);
    dim3 grid(grid_x, (int)M);
    dim3 block(BXR_V2_THREADS);

    build_x_repeated_v2_kernel<<<grid, block, 0, stream>>>(
        X_repeated, X, adapter_ids, M, K, K_ext, R, A);
}
