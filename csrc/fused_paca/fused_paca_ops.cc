#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

/* ---------- v1: warp-cooperative GEMV (multi_paca_custom M<=4) ---------- */
void launch_fused_paca_gemm(
    __half* Y, const __half* X, const __half* W_T, const __half* P,
    const int64_t* aids,
    int64_t M, int64_t K, int64_t N, int64_t R,
    float scale, cudaStream_t stream);

/* ---------- v2: wmma tensor-core GEMM (multi_paca_custom M>4) ---------- */
void launch_fused_paca_tc(
    __half* Y, const __half* X, const __half* W, const __half* P,
    const int64_t* aids,
    int64_t M, int64_t K, int64_t N, int64_t R,
    float scale, cudaStream_t stream);

/* ---------- build_x_repeated: construct X_repeated for concat-GEMM ---------- */
void launch_build_x_repeated(
    __half* X_repeated, const __half* X, const int64_t* adapter_ids,
    int64_t M, int64_t K, int64_t K_ext, int64_t R, int64_t A,
    cudaStream_t stream);

/* ---------- build_x_repeated v2: 128-bit vectorized ---------- */
void launch_build_x_repeated_v2(
    __half* X_repeated, const __half* X, const int64_t* adapter_ids,
    int64_t M, int64_t K, int64_t K_ext, int64_t R, int64_t A,
    cudaStream_t stream);

/* ============================================================ */
/*  v1: dispatch_fused_paca  (W_T = transposed weight [N, K])   */
/* ============================================================ */
void dispatch_fused_paca(
    torch::Tensor Y, torch::Tensor X, torch::Tensor W_T,
    torch::Tensor P, torch::Tensor adapter_ids,
    int64_t R, double scale)
{
    int64_t M = X.size(0), K = X.size(1), N = W_T.size(0);
    cudaStream_t s = c10::cuda::getCurrentCUDAStream().stream();
    launch_fused_paca_gemm(
        reinterpret_cast<__half*>(Y.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(X.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(W_T.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(P.data_ptr<at::Half>()),
        adapter_ids.data_ptr<int64_t>(),
        M, K, N, R, (float)scale, s);
}

/* ============================================================ */
/*  v2: dispatch_fused_paca_tc  (W = original weight [K, N])    */
/* ============================================================ */
void dispatch_fused_paca_tc(
    torch::Tensor Y, torch::Tensor X, torch::Tensor W,
    torch::Tensor P, torch::Tensor adapter_ids,
    int64_t R, double scale)
{
    int64_t M = X.size(0), K = X.size(1), N = W.size(1);
    TORCH_CHECK(W.size(0) == K, "W must be [K, N]");
    cudaStream_t s = c10::cuda::getCurrentCUDAStream().stream();
    launch_fused_paca_tc(
        reinterpret_cast<__half*>(Y.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(X.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(W.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(P.data_ptr<at::Half>()),
        adapter_ids.data_ptr<int64_t>(),
        M, K, N, R, (float)scale, s);
}

/* ============================================================ */
/*  build_x_repeated: construct X_repeated for concat-GEMM      */
/* ============================================================ */
void dispatch_build_x_repeated(
    torch::Tensor X_repeated, torch::Tensor X,
    torch::Tensor adapter_ids,
    int64_t R, int64_t A)
{
    int64_t M = X.size(0), K = X.size(1);
    int64_t K_ext = K + A * R;
    TORCH_CHECK(X_repeated.size(0) == M && X_repeated.size(1) == K_ext,
                "X_repeated must be [M, K+A*R]");
    cudaStream_t s = c10::cuda::getCurrentCUDAStream().stream();
    launch_build_x_repeated(
        reinterpret_cast<__half*>(X_repeated.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(X.data_ptr<at::Half>()),
        adapter_ids.data_ptr<int64_t>(),
        M, K, K_ext, R, A, s);
}

/* ============================================================ */
/*  build_x_repeated v2: 128-bit vectorized                     */
/* ============================================================ */
void dispatch_build_x_repeated_v2(
    torch::Tensor X_repeated, torch::Tensor X,
    torch::Tensor adapter_ids,
    int64_t R, int64_t A)
{
    int64_t M = X.size(0), K = X.size(1);
    int64_t K_ext = K + A * R;
    TORCH_CHECK(X_repeated.size(0) == M && X_repeated.size(1) == K_ext,
                "X_repeated must be [M, K+A*R]");
    TORCH_CHECK(K % 8 == 0 && R % 8 == 0,
                "Vectorized v2 requires K and R to be multiples of 8");
    cudaStream_t s = c10::cuda::getCurrentCUDAStream().stream();
    launch_build_x_repeated_v2(
        reinterpret_cast<__half*>(X_repeated.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(X.data_ptr<at::Half>()),
        adapter_ids.data_ptr<int64_t>(),
        M, K, K_ext, R, A, s);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dispatch_fused_paca", &dispatch_fused_paca,
          "v1: warp-cooperative GEMV + PACA (W_T transposed)");
    m.def("dispatch_fused_paca_tc", &dispatch_fused_paca_tc,
          "v2: wmma tensor-core GEMM + PACA epilogue (W original)");
    m.def("dispatch_build_x_repeated", &dispatch_build_x_repeated,
          "Build X_repeated for multi_paca concat-GEMM");
    m.def("dispatch_build_x_repeated_v2", &dispatch_build_x_repeated_v2,
          "Build X_repeated v2: 128-bit vectorized (requires K,R % 8 == 0)");
}
