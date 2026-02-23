"""JIT build for fused_paca CUDA extension (v1 + v2 tensor-core)."""
import os, torch
from torch.utils.cpp_extension import load

_DIR = os.path.dirname(os.path.abspath(__file__))

fused_paca = load(
    name="fused_paca",
    sources=[
        os.path.join(_DIR, "fused_paca_ops.cc"),
        os.path.join(_DIR, "fused_paca_all.cu"),
    ],
    extra_include_paths=[_DIR],
    extra_cuda_cflags=[
        "-O3",
        "-std=c++17",
        "--use_fast_math",
        "-lineinfo",
        # Force SM80+ for wmma tensor cores
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_90,code=sm_90",
    ],
    extra_cflags=["-O3", "-std=c++17"],
    verbose=True,
)

if __name__ == "__main__":
    print("fused_paca built OK:", fused_paca)
