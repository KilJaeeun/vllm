/* v1 — warp-cooperative GEMV (best for M ≤ 4, used by multi_paca_custom) */
#include "fused_paca_kernel.cuh"

/* v2 — wmma tensor-core GEMM + PACA epilogue (used by multi_paca_custom M>4) */
#include "fused_paca_tc_kernel.cuh"

/* X_repeated construction kernel (used by multi_paca concat-GEMM) */
#include "build_x_repeated_kernel.cuh"

/* X_repeated v2: 128-bit vectorized version */
#include "build_x_repeated_v2_kernel.cuh"
