"""
Utility functions for PyTorch hooks.
"""

import torch
import torch.nn as nn
from typing import Optional, Callable, Any


def get_module_by_name(model: nn.Module, layer_name: str) -> Optional[nn.Module]:
    """
    Get a module from a model by its name.

    Args:
        model: The PyTorch model.
        layer_name: The name of the layer (e.g., 'transformer.layer.0').

    Returns:
        The module if found, None otherwise.
    """
    try:
        # Split the name by dots and traverse
        parts = layer_name.split('.')
        module = model

        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                # Try numeric indexing for Sequential modules
                try:
                    idx = int(part)
                    module = module[idx]
                except (ValueError, TypeError, IndexError):
                    return None

        return module
    except Exception:
        return None


def register_forward_hook(
    module: nn.Module,
    hook_fn: Callable,
) -> Any:
    """
    Register a forward hook on a module.

    Args:
        module: The module to hook.
        hook_fn: The hook function.

    Returns:
        The hook handle.
    """
    return module.register_forward_hook(hook_fn)


def register_backward_hook(
    module: nn.Module,
    hook_fn: Callable,
) -> Any:
    """
    Register a backward hook on a module.

    Args:
        module: The module to hook.
        hook_fn: The hook function.

    Returns:
        The hook handle.
    """
    return module.register_full_backward_hook(hook_fn)


def get_layer_names(model: nn.Module, leaf_only: bool = True) -> list[str]:
    """
    Get all layer names from a model.

    Args:
        model: The PyTorch model.
        leaf_only: If True, only return leaf modules (no children).

    Returns:
        List of layer names.
    """
    names = []

    for name, module in model.named_modules():
        if not leaf_only or len(list(module.children())) == 0:
            names.append(name)

    return names


def count_parameters(module: nn.Module) -> int:
    """
    Count the number of parameters in a module.

    Args:
        module: The PyTorch module.

    Returns:
        Number of parameters.
    """
    return sum(p.numel() for p in module.parameters())


def get_module_info(module: nn.Module) -> dict:
    """
    Get information about a module.

    Args:
        module: The PyTorch module.

    Returns:
        Dictionary with module information.
    """
    return {
        'type': module.__class__.__name__,
        'num_parameters': count_parameters(module),
        'num_children': len(list(module.children())),
        'trainable': any(p.requires_grad for p in module.parameters()),
    }


def tensor_to_dict(tensor: torch.Tensor, compute_stats: bool = True) -> dict:
    """
    Convert a tensor to a dictionary with metadata and optional statistics.

    Args:
        tensor: The PyTorch tensor.
        compute_stats: Whether to compute statistics.

    Returns:
        Dictionary with tensor information.
    """
    data = {
        'shape': list(tensor.shape),
        'dtype': str(tensor.dtype),
        'device': str(tensor.device),
        'requires_grad': tensor.requires_grad,
        'numel': tensor.numel(),
    }

    if compute_stats and tensor.numel() > 0:
        t = tensor.detach().cpu().float()
        data['stats'] = {
            'mean': float(t.mean()),
            'std': float(t.std()),
            'min': float(t.min()),
            'max': float(t.max()),
            'abs_mean': float(t.abs().mean()),
        }

    return data
