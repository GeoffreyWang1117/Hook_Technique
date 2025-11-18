"""
Backward hook for capturing gradient information.
"""

import torch
import torch.nn as nn
from typing import Any, Optional, Callable, Dict
from llm_hooks.core.base_hook import BaseHook
from llm_hooks.core.hook_registry import register_hook
from llm_hooks.pytorch.utils import get_module_by_name


@register_hook("backward")
class BackwardHook(BaseHook):
    """
    Hook for capturing backward pass gradients.

    This hook captures gradient tensors during the backward pass,
    along with statistics for analyzing gradient flow.
    """

    def __init__(
        self,
        layer_name: Optional[str] = None,
        layer_pattern: Optional[str] = None,
        capture_input_grad: bool = True,
        capture_output_grad: bool = True,
        compute_stats: bool = True,
        detect_vanishing: bool = True,
        detect_exploding: bool = True,
        name: Optional[str] = None,
    ):
        """
        Initialize the backward hook.

        Args:
            layer_name: Specific layer name to hook.
            layer_pattern: Pattern to match layer names.
            capture_input_grad: Whether to capture input gradients.
            capture_output_grad: Whether to capture output gradients.
            compute_stats: Whether to compute gradient statistics.
            detect_vanishing: Whether to detect vanishing gradients.
            detect_exploding: Whether to detect exploding gradients.
            name: Optional name for the hook.
        """
        super().__init__(name=name or "BackwardHook")
        self.layer_name = layer_name
        self.layer_pattern = layer_pattern
        self.capture_input_grad = capture_input_grad
        self.capture_output_grad = capture_output_grad
        self.compute_stats = compute_stats
        self.detect_vanishing = detect_vanishing
        self.detect_exploding = detect_exploding

        # Thresholds for gradient issues
        self.vanishing_threshold = 1e-6
        self.exploding_threshold = 100.0

    def setup(self, model: nn.Module) -> None:
        """
        Setup backward hooks on the model.

        Args:
            model: The PyTorch model to hook.
        """
        if self.layer_name:
            module = get_module_by_name(model, self.layer_name)
            if module is not None:
                handle = module.register_full_backward_hook(
                    self._create_hook_fn(self.layer_name)
                )
                self._hook_handles.append(handle)
        elif self.layer_pattern:
            for name, module in model.named_modules():
                if self._matches_pattern(name, self.layer_pattern):
                    handle = module.register_full_backward_hook(
                        self._create_hook_fn(name)
                    )
                    self._hook_handles.append(handle)
        else:
            for name, module in model.named_modules():
                if len(list(module.children())) == 0:
                    handle = module.register_full_backward_hook(
                        self._create_hook_fn(name)
                    )
                    self._hook_handles.append(handle)

    def teardown(self) -> None:
        """Remove all backward hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def _create_hook_fn(self, layer_name: str) -> Callable:
        """Create a backward hook function for a specific layer."""
        def hook_fn(module: nn.Module, grad_input: tuple, grad_output: tuple):
            if not self.enabled:
                return

            data = {}

            # Capture input gradients
            if self.capture_input_grad and grad_input:
                input_grads = []
                for i, grad in enumerate(grad_input):
                    if grad is not None:
                        grad_data = self._process_gradient(grad, f"input_{i}")
                        input_grads.append(grad_data)
                data['input_grads'] = input_grads

            # Capture output gradients
            if self.capture_output_grad and grad_output:
                output_grads = []
                for i, grad in enumerate(grad_output):
                    if grad is not None:
                        grad_data = self._process_gradient(grad, f"output_{i}")
                        output_grads.append(grad_data)
                data['output_grads'] = output_grads

            # Create and store result
            result = self.create_result(
                hook_type='backward',
                data=data,
                layer_name=layer_name,
                metadata={
                    'module_type': module.__class__.__name__,
                }
            )
            self.add_result(result)

        return hook_fn

    def _process_gradient(self, grad: torch.Tensor, grad_name: str) -> Dict[str, Any]:
        """Process a gradient tensor and extract statistics."""
        data = {
            'name': grad_name,
            'shape': list(grad.shape),
            'dtype': str(grad.dtype),
            'device': str(grad.device),
        }

        if self.compute_stats and grad.numel() > 0:
            g = grad.detach().cpu().float()

            stats = {
                'mean': float(g.mean()),
                'std': float(g.std()),
                'min': float(g.min()),
                'max': float(g.max()),
                'abs_mean': float(g.abs().mean()),
                'abs_max': float(g.abs().max()),
                'norm': float(torch.norm(g)),
                'num_zeros': int((g == 0).sum()),
                'num_nans': int(torch.isnan(g).sum()),
                'num_infs': int(torch.isinf(g).sum()),
            }

            # Check for gradient issues
            issues = []
            if self.detect_vanishing and stats['abs_mean'] < self.vanishing_threshold:
                issues.append('vanishing_gradient')

            if self.detect_exploding and stats['abs_max'] > self.exploding_threshold:
                issues.append('exploding_gradient')

            if stats['num_nans'] > 0:
                issues.append('nan_gradient')

            if stats['num_infs'] > 0:
                issues.append('inf_gradient')

            stats['issues'] = issues
            data['stats'] = stats

        return data

    def _matches_pattern(self, name: str, pattern: str) -> bool:
        """Check if a name matches a pattern."""
        import re
        pattern_re = pattern.replace('*', '.*').replace('?', '.')
        return bool(re.match(f'^{pattern_re}$', name))
