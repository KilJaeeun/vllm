"""
Patch vLLM model to use PaCA concat-GEMM layers instead of LoRA layers.

Called at the end of LoRAModelManager._create_lora_modules() when
VLLM_PACA_ENABLED=1 environment variable is set.
"""

import logging

import torch.nn as nn

from vllm.lora.paca_layer import PaCALinearLayer

logger = logging.getLogger(__name__)


def patch_model_for_paca(
    model: nn.Module,
    lora_config,
    max_loras: int,
    target_modules: set[str] | None = None,
) -> nn.Module:
    """
    Replace LoRA-wrapped linear layers in a vLLM model with PaCA layers.

    Iterates over named modules and replaces BaseLinearLayerWithLoRA
    instances with PaCALinearLayer.

    Args:
        model: vLLM model (already initialized with LoRA support)
        lora_config: vLLM LoRAConfig
        max_loras: Maximum number of concurrent adapters
        target_modules: Set of module suffixes to replace (e.g. {"q_proj", "k_proj"}).
                       If None, replaces all LoRA-wrapped linear layers.

    Returns:
        The modified model (in-place).
    """
    from vllm.lora.layers import BaseLayerWithLoRA

    replaced = 0
    modules_to_replace = {}

    for name, module in model.named_modules():
        if not isinstance(module, BaseLayerWithLoRA):
            continue
        if not hasattr(module, "base_layer"):
            continue

        # Check target_modules filter
        if target_modules is not None:
            module_suffix = name.split(".")[-1]
            if module_suffix not in target_modules:
                continue

        modules_to_replace[name] = module

    for name, old_module in modules_to_replace.items():
        n_slices = getattr(old_module, "n_slices", 1)
        output_slices = getattr(old_module, "output_slices", None)

        paca_layer = PaCALinearLayer(old_module.base_layer)
        paca_layer.create_lora_weights(
            max_loras, lora_config,
            n_slices=n_slices,
            output_slices=output_slices,
        )

        # Preserve the punica wrapper reference (for token_lora_indices)
        if hasattr(old_module, "punica_wrapper"):
            paca_layer.set_mapping(old_module.punica_wrapper)

        # Replace in the model
        _replace_submodule(model, name, paca_layer)
        replaced += 1
        logger.info("Replaced %s with PaCALinearLayer", name)

    logger.info("Patched %d layers for PaCA concat-GEMM", replaced)
    return model


def _replace_submodule(model: nn.Module, target: str, new_module: nn.Module):
    """Replace a named submodule in the model."""
    parts = target.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def patch_manager_for_paca(manager):
    """
    Patch a LoRAModelManager so its modules dict points to PaCA layers.

    Call this after patch_model_for_paca() to update the manager's
    module references.
    """
    updated = 0
    for module_name in list(manager.modules.keys()):
        # Re-fetch the module from the model (which may have been replaced)
        try:
            module = manager.model.get_submodule(module_name)
        except AttributeError:
            continue

        if isinstance(module, PaCALinearLayer):
            manager.modules[module_name] = module
            updated += 1

    logger.info("Updated %d manager module references to PaCA layers", updated)
