/**
 * build_x_repeated_kernel — single CUDA kernel to construct X_repeated
 * for concat-GEMM.
 *
 * X_repeated[m, k] =
 *   if k < K:                                X[m, k]              (base input)
 *   elif k in [K + aid*R, K + (aid+1)*R):    X[m, K-R+(k-K-aid*R)]  (repeated adapter slice)
 *   else:                                    0                    (zero padding)
 *
 * Where aid = adapter_ids[m].
 *
 * One kernel launch replaces: zeros() + scatter_() + cat() = 3 launches.
 *
 * Grid:  (ceil(K_ext / 256), M)    — each row handled by ceil(K_ext/256) blocks
 * Block: 256 threads
 */

#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

static constexpr int BXR_THREADS = 256;

__global__ void build_x_repeated_kernel(
    __half*        __restrict__ X_repeated,  // [M, K_ext] output
    const __half*  __restrict__ X,           // [M, K] input
    const int64_t* __restrict__ adapter_ids, // [M]
    int64_t M, int64_t K, int64_t K_ext,
    int64_t R, int64_t A)
{
    const int64_t m = blockIdx.y;
    if (m >= M) return;

    const int64_t aid = adapter_ids[m];
    const bool has_adapter = (aid >= 0 && aid < A);
    const int64_t adapter_col_start = has_adapter ? (K + aid * R) : -1;

    const __half* x_row = X + m * K;
    __half* out_row = X_repeated + m * K_ext;

    for (int64_t col = blockIdx.x * BXR_THREADS + threadIdx.x;
         col < K_ext;
         col += (int64_t)gridDim.x * BXR_THREADS)
    {
        __half val;
        if (col < K) {
            val = x_row[col];
        } else if (has_adapter && col >= adapter_col_start && col < adapter_col_start + R) {
            val = x_row[K - R + (col - adapter_col_start)];
        } else {
            val = __float2half(0.0f);
        }
        out_row[col] = val;
    }
}

/* Host launcher */
void launch_build_x_repeated(
    __half* X_repeated, const __half* X, const int64_t* adapter_ids,
    int64_t M, int64_t K, int64_t K_ext, int64_t R, int64_t A,
    cudaStream_t stream)
{
    int grid_x = (int)((K_ext + BXR_THREADS - 1) / BXR_THREADS);
    dim3 grid(grid_x, (int)M);
    dim3 block(BXR_THREADS);

    build_x_repeated_kernel<<<grid, block, 0, stream>>>(
        X_repeated, X, adapter_ids, M, K, K_ext, R, A);
}
