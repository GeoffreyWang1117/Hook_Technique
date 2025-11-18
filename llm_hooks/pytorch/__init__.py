"""
PyTorch-specific hooks for forward and backward passes.
"""

from llm_hooks.pytorch.forward_hook import ForwardHook
from llm_hooks.pytorch.backward_hook import BackwardHook
from llm_hooks.pytorch.layer_hook import LayerHook
from llm_hooks.pytorch.utils import (
    get_module_by_name,
    register_forward_hook,
    register_backward_hook,
)

__all__ = [
    "ForwardHook",
    "BackwardHook",
    "LayerHook",
    "get_module_by_name",
    "register_forward_hook",
    "register_backward_hook",
]
