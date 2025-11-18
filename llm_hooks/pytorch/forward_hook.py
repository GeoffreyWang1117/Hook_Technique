"""
Forward hook for capturing forward pass activations.
"""

import torch
import torch.nn as nn
from typing import Any, Optional, Callable, List, Dict
from llm_hooks.core.base_hook import BaseHook
from llm_hooks.core.hook_registry import register_hook
from llm_hooks.pytorch.utils import get_module_by_name
import numpy as np


@register_hook("forward")
class ForwardHook(BaseHook):
    """
    Hook for capturing forward pass activations.

    This hook captures input and output tensors during the forward pass,
    along with statistics like mean, std, min, max, and sparsity.
    """

    def __init__(
        self,
        layer_name: Optional[str] = None,
        layer_pattern: Optional[str] = None,
        capture_input: bool = True,
        capture_output: bool = True,
        compute_stats: bool = True,
        name: Optional[str] = None,
    ):
        """
        Initialize the forward hook.

        Args:
            layer_name: Specific layer name to hook (e.g., 'transformer.layer.0').
            layer_pattern: Pattern to match layer names (e.g., 'transformer.layer.*').
            capture_input: Whether to capture input tensors.
            capture_output: Whether to capture output tensors.
            compute_stats: Whether to compute statistics on tensors.
            name: Optional name for the hook.
        """
        super().__init__(name=name or "ForwardHook")
        self.layer_name = layer_name
        self.layer_pattern = layer_pattern
        self.capture_input = capture_input
        self.capture_output = capture_output
        self.compute_stats = compute_stats

    def setup(self, model: nn.Module) -> None:
        """
        Setup forward hooks on the model.

        Args:
            model: The PyTorch model to hook.
        """
        if self.layer_name:
            # Hook a specific layer
            module = get_module_by_name(model, self.layer_name)
            if module is not None:
                handle = module.register_forward_hook(self._create_hook_fn(self.layer_name))
                self._hook_handles.append(handle)
        elif self.layer_pattern:
            # Hook layers matching pattern
            for name, module in model.named_modules():
                if self._matches_pattern(name, self.layer_pattern):
                    handle = module.register_forward_hook(self._create_hook_fn(name))
                    self._hook_handles.append(handle)
        else:
            # Hook all modules
            for name, module in model.named_modules():
                if len(list(module.children())) == 0:  # Only leaf modules
                    handle = module.register_forward_hook(self._create_hook_fn(name))
                    self._hook_handles.append(handle)

    def teardown(self) -> None:
        """Remove all forward hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def _create_hook_fn(self, layer_name: str) -> Callable:
        """Create a hook function for a specific layer."""
        def hook_fn(module: nn.Module, input: tuple, output: Any):
            if not self.enabled:
                return

            data = {}

            # Capture input
            if self.capture_input and input:
                input_data = self._process_tensor(input[0] if isinstance(input, tuple) else input)
                data['input'] = input_data

            # Capture output
            if self.capture_output:
                output_data = self._process_tensor(output)
                data['output'] = output_data

            # Create and store result
            result = self.create_result(
                hook_type='forward',
                data=data,
                layer_name=layer_name,
                metadata={
                    'module_type': module.__class__.__name__,
                }
            )
            self.add_result(result)

        return hook_fn

    def _process_tensor(self, tensor: Any) -> Dict[str, Any]:
        """Process a tensor and extract statistics."""
        if not isinstance(tensor, torch.Tensor):
            return {'type': 'non_tensor', 'value': str(tensor)}

        data = {
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'device': str(tensor.device),
        }

        if self.compute_stats and tensor.numel() > 0:
            # Detach and move to CPU for statistics
            t = tensor.detach().cpu()

            # Handle different dtypes
            if t.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                t_float = t.float()
                data['stats'] = {
                    'mean': float(t_float.mean()),
                    'std': float(t_float.std()),
                    'min': float(t_float.min()),
                    'max': float(t_float.max()),
                    'abs_mean': float(t_float.abs().mean()),
                    'sparsity': float((t_float == 0).float().mean()),
                }

                # Compute percentiles
                flat = t_float.flatten()
                if len(flat) > 0:
                    data['stats']['percentiles'] = {
                        'p25': float(torch.quantile(flat, 0.25)),
                        'p50': float(torch.quantile(flat, 0.50)),
                        'p75': float(torch.quantile(flat, 0.75)),
                        'p95': float(torch.quantile(flat, 0.95)),
                    }

        return data

    def _matches_pattern(self, name: str, pattern: str) -> bool:
        """Check if a name matches a pattern (simple wildcard support)."""
        import re
        pattern_re = pattern.replace('*', '.*').replace('?', '.')
        return bool(re.match(f'^{pattern_re}$', name))
