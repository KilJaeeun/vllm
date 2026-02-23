"""
PaCA linear layer for vLLM — replaces base+LoRA forward with concat-GEMM.

Instead of:  Y = base(X) + shrink_expand(X)   [3 kernel launches]
Does:        Y = X_repeated @ W_merged          [2 kernel launches]

W_merged = [base_weight^T ; P[0]*scale ; ... ; P[A-1]*scale ; zeros(null)]
                 [K, N]       [R, N]                            [R, N]
Shape: [K + (max_loras+1)*R, N]  — last slot is null (for no-adapter tokens)
"""

import os
import importlib.util

import torch
import torch.nn as nn
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import LoRAConfig

# ---------------------------------------------------------------------------
#  Lazy JIT loader for fused_paca CUDA extension
# ---------------------------------------------------------------------------
_fused_paca_mod = None


def _get_fused_paca():
    global _fused_paca_mod
    if _fused_paca_mod is not None:
        return _fused_paca_mod

    build_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "csrc", "fused_paca", "build.py"
    )
    if not os.path.exists(build_path):
        raise FileNotFoundError(f"fused_paca build.py not found at {build_path}")
    spec = importlib.util.spec_from_file_location("fused_paca_build", build_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _fused_paca_mod = mod.fused_paca
    return _fused_paca_mod


# ---------------------------------------------------------------------------
#  Register CUDA kernel as torch custom op for torch.compile compatibility
# ---------------------------------------------------------------------------
_fused_op_registered = False


def _ensure_fused_op():
    """Register build_x_repeated_v2 as a torch custom op (once)."""
    global _fused_op_registered
    if _fused_op_registered:
        return
    _fused_op_registered = True

    @torch.library.custom_op(
        "fused_paca::build_x_repeated", mutates_args=("x_repeated",),
    )
    def build_x_repeated(
        x_repeated: torch.Tensor, x: torch.Tensor,
        adapter_ids: torch.Tensor, R: int, A: int,
    ) -> None:
        _get_fused_paca().dispatch_build_x_repeated_v2(
            x_repeated, x, adapter_ids, R, A,
        )

    @build_x_repeated.register_fake
    def _(x_repeated, x, adapter_ids, R, A):
        pass  # x_repeated is mutated in-place


class PaCALinearLayer(nn.Module):
    """
    Drop-in replacement for vLLM's BaseLinearLayerWithLoRA.

    Implements PaCA concat-GEMM: builds X_repeated (CUDA kernel), then
    single cuBLAS GEMM with W_merged.

    Adapter slot management:
      - Slots 0..max_loras-1: active PaCA adapters
      - Slot max_loras: null slot (zero-filled, for tokens without adapters)
      - token_lora_indices from vLLM maps tokens to slots (-1 → null slot)
    """

    def __init__(self, base_layer: nn.Module):
        super().__init__()
        self.base_layer = base_layer
        self.input_size = base_layer.input_size
        self.tp_size = getattr(base_layer, "tp_size", 1)
        self.tp_rank = getattr(base_layer, "tp_rank", 0)

        # Will be set by create_lora_weights()
        self.R: int = 0
        self.max_loras: int = 0
        self.n_slices: int = 1
        self.output_size: int = 0
        self.output_slices: tuple = ()
        self.w_merged: torch.Tensor | None = None
        self.paca_p_stacked: torch.Tensor | None = None
        self.punica_wrapper = None

        # Dummy stacked tensors for LoRA manager compatibility
        self.lora_a_stacked: tuple = ()
        self.lora_b_stacked: tuple = ()

    def _get_base_weight(self) -> torch.Tensor:
        """Extract base weight as [K, N] from the underlying layer."""
        layer = self.base_layer
        if hasattr(layer, "weight"):
            w = layer.weight.data  # [N, K] in PyTorch convention
        elif hasattr(layer, "weight_packed"):
            w = layer.weight_packed.data
        elif hasattr(layer, "qweight"):
            raise NotImplementedError("PaCA does not support quantized weights yet")
        else:
            raise ValueError(f"Cannot extract weight from {type(layer)}")

        # Transpose to [K, N] for concat-GEMM
        return w.T.contiguous()

    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: "LoRAConfig",
        model_config=None,
        n_slices: int = 1,
        output_slices: tuple | None = None,
    ) -> None:
        """Initialize W_merged and PaCA weight slots."""
        self.R = lora_config.max_lora_rank
        self.max_loras = max_loras
        self.lora_config = lora_config
        self.output_size = getattr(
            self.base_layer, "output_size_per_partition",
            getattr(self.base_layer, "output_size", 0),
        )
        self.n_slices = n_slices
        if output_slices is not None:
            self.output_slices = output_slices
        else:
            self.output_slices = (self.output_size,)

        R = self.R
        K = self.input_size
        N = self.output_size
        A = max_loras + 1  # +1 for null slot

        device = next(self.base_layer.parameters()).device
        dtype = lora_config.lora_dtype

        # PaCA P weights per slot: [A, R, N]
        self.paca_p_stacked = torch.zeros(A, R, N, dtype=dtype, device=device)

        # W_merged = [base_weight ; paca_slots ; null_slot]
        base_w = self._get_base_weight()  # [K, N]
        self.w_merged = torch.zeros(K + A * R, N, dtype=dtype, device=device)
        self.w_merged[:K, :].copy_(base_w.to(dtype))

        # Dummy stacked tensors shaped for LoRA manager compatibility.
        # Manager warmup accesses lora_a_stacked[i].shape[-1] (=K)
        # and lora_b_stacked[i].shape[-2] (=N_slice) for each slice.
        self.lora_a_stacked = tuple(
            torch.zeros(1, 1, R, K, dtype=dtype, device=device)
            for _ in range(self.n_slices)
        )
        self.lora_b_stacked = tuple(
            torch.zeros(1, 1, N_slice, R, dtype=dtype, device=device)
            for N_slice in self.output_slices
        )

        # Try to enable fused CUDA kernel
        self._use_fused_kernel = False
        if dtype == torch.float16 and K % 8 == 0 and R % 8 == 0:
            try:
                _get_fused_paca()
                _ensure_fused_op()
                self._use_fused_kernel = True
            except Exception:
                pass

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor | list | None,
        lora_b: torch.Tensor | list,
    ):
        """
        Load PaCA weight P into a slot and update W_merged.

        For simple layers: lora_b is [N, R] (vLLM convention).
        For packed layers: lora_b is a list of [N_i, R] tensors per slice.
        lora_a is ignored (PaCA has no shrink matrix).
        """
        self.reset_lora(index)

        # Packed layers: concatenate per-slice P tensors along N dimension
        if isinstance(lora_b, (list, tuple)):
            parts = [lb for lb in lora_b if lb is not None]
            lora_b = torch.cat(parts, dim=0)  # [N_total, R]

        # lora_b arrives as [N, R] from vLLM; transpose to PaCA P [R, N]
        P = lora_b.T.contiguous() if lora_b.ndim == 2 else lora_b
        if P.ndim != 2:
            raise ValueError(f"Expected P to be 2D [R, N], got shape {P.shape}")

        R_actual, N_actual = P.shape
        self.paca_p_stacked[index, :R_actual, :N_actual].copy_(
            P.to(self.paca_p_stacked.dtype), non_blocking=True
        )
        K = self.input_size
        R = self.R
        self.w_merged[K + index * R : K + index * R + R_actual, :N_actual].copy_(
            P.to(self.w_merged.dtype), non_blocking=True
        )

    def reset_lora(self, index: int):
        """Zero out a PaCA slot in both paca_p_stacked and W_merged."""
        if self.paca_p_stacked is not None:
            self.paca_p_stacked[index].zero_()
        K = self.input_size
        R = self.R
        if self.w_merged is not None:
            self.w_merged[K + index * R : K + (index + 1) * R, :].zero_()

    def set_mapping(self, punica_wrapper):
        """Bind the punica wrapper (used for token_lora_indices)."""
        self.punica_wrapper = punica_wrapper

    def slice_lora_a(self, lora_a):
        return lora_a

    def slice_lora_b(self, lora_b):
        """For TP, slice the output dimension. Handles list (packed) or tensor."""
        if isinstance(lora_b, (list, tuple)):
            return lora_b  # packed layers: per-slice slicing handled by manager
        if self.tp_size > 1:
            shard_size = self.output_size
            start = self.tp_rank * shard_size
            end = start + shard_size
            return lora_b[start:end, :]  # slice N dim
        return lora_b

    def apply(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        """
        PaCA forward pass.

        Fused path: CUDA kernel builds X_repeated + cuBLAS GEMM.
        Fallback: PyTorch scatter_ builds X_repeated + cuBLAS GEMM.
        """
        original_shape = x.shape
        x_flat = x.view(-1, x.shape[-1])
        M, K = x_flat.shape

        token_indices = self.punica_wrapper.token_lora_indices
        adapter_ids = token_indices[:M].clone()
        adapter_ids[adapter_ids < 0] = self.max_loras  # null slot
        A = self.max_loras + 1
        R = self.R
        K_ext = K + A * R

        if self._use_fused_kernel and x.dtype == torch.float16:
            # CUDA kernel: vectorized X_repeated construction
            X_repeated = torch.empty(M, K_ext, dtype=x.dtype, device=x.device)
            torch.ops.fused_paca.build_x_repeated(
                X_repeated, x_flat, adapter_ids.long(), R, A,
            )
        else:
            # PyTorch fallback: scatter_ X_repeated construction
            X_repeated = torch.zeros(M, K_ext, dtype=x.dtype, device=x.device)
            X_repeated[:, :K] = x_flat
            adapter_starts = K + adapter_ids * R
            col_offsets = torch.arange(R, device=x.device)
            col_indices = adapter_starts.unsqueeze(1) + col_offsets.unsqueeze(0)
            X_repeated.scatter_(1, col_indices, x_flat[:, K - R :])

        output = torch.mm(X_repeated, self.w_merged)

        if bias is not None:
            output = output + bias

        output = output.view(*original_shape[:-1], output.shape[-1])
        return output

    def forward(
        self, input_: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Forward compatible with vLLM's ColumnParallelLinear interface."""
        bias = (
            self.base_layer.bias
            if hasattr(self.base_layer, "bias")
            and self.base_layer.bias is not None
            and not getattr(self.base_layer, "skip_bias_add", False)
            else None
        )
        output = self.apply(input_, bias)

        if not getattr(self.base_layer, "return_bias", True):
            return output

        output_bias = (
            self.base_layer.bias
            if hasattr(self.base_layer, "bias")
            and getattr(self.base_layer, "skip_bias_add", False)
            else None
        )
        return output, output_bias

    @classmethod
    def can_replace_layer(cls, source_layer, lora_config, packed_modules_list,
                          model_config=None) -> bool:
        """PaCA can replace any linear layer (TP=1 for now)."""
        from vllm.model_executor.layers.linear import (
            ColumnParallelLinear, MergedColumnParallelLinear,
            RowParallelLinear, ReplicatedLinear, QKVParallelLinear,
        )
        return isinstance(
            source_layer,
            (ColumnParallelLinear, MergedColumnParallelLinear,
             RowParallelLinear, ReplicatedLinear, QKVParallelLinear),
        )

    @property
    def weight(self) -> torch.Tensor:
        if hasattr(self.base_layer, "weight"):
            return self.base_layer.weight
        elif hasattr(self.base_layer, "weight_packed"):
            return self.base_layer.weight_packed
        else:
            raise ValueError(f"Unsupported base layer: {self.base_layer}")

    @property
    def bias(self) -> torch.Tensor | None:
        return getattr(self.base_layer, "bias", None)
