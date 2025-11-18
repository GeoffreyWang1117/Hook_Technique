"""
Gradient flow tracking and visualization.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
from llm_hooks.core.base_hook import BaseHook
from llm_hooks.core.hook_registry import register_hook
import numpy as np


@register_hook("gradient")
class GradientTracker(BaseHook):
    """
    Track gradient flow through the network.

    Monitors gradient magnitudes, detects vanishing/exploding gradients,
    and analyzes backpropagation paths.
    """

    def __init__(
        self,
        layer_pattern: str = "*",
        detect_issues: bool = True,
        vanishing_threshold: float = 1e-6,
        exploding_threshold: float = 100.0,
        name: Optional[str] = None,
    ):
        """
        Initialize the gradient tracker.

        Args:
            layer_pattern: Pattern to match layers.
            detect_issues: Whether to detect gradient issues.
            vanishing_threshold: Threshold for vanishing gradient detection.
            exploding_threshold: Threshold for exploding gradient detection.
            name: Optional name for the hook.
        """
        super().__init__(name=name or "GradientTracker")
        self.layer_pattern = layer_pattern
        self.detect_issues = detect_issues
        self.vanishing_threshold = vanishing_threshold
        self.exploding_threshold = exploding_threshold

        # Gradient flow data
        self.gradient_flow: List[Dict[str, Any]] = []

    def setup(self, model: nn.Module) -> None:
        """Setup gradient tracking hooks."""
        # Register hooks on parameters
        for name, param in model.named_parameters():
            if param.requires_grad and self._matches_pattern(name, self.layer_pattern):
                handle = param.register_hook(
                    self._create_gradient_hook(name)
                )
                self._hook_handles.append(handle)

    def teardown(self) -> None:
        """Remove all hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self.gradient_flow.clear()

    def _create_gradient_hook(self, param_name: str):
        """Create a gradient tracking hook."""
        def hook_fn(grad: torch.Tensor):
            if not self.enabled:
                return grad

            if grad is not None:
                data = self._analyze_gradient(param_name, grad)
                self.gradient_flow.append(data)

                # Create result
                result = self.create_result(
                    hook_type='gradient',
                    data=data,
                    layer_name=param_name,
                )
                self.add_result(result)

            return grad

        return hook_fn

    def _analyze_gradient(self, param_name: str, grad: torch.Tensor) -> Dict[str, Any]:
        """Analyze a gradient tensor."""
        g = grad.detach().cpu().float()

        data = {
            'param_name': param_name,
            'shape': list(grad.shape),
            'dtype': str(grad.dtype),
        }

        # Basic statistics
        data['stats'] = {
            'mean': float(g.mean()),
            'std': float(g.std()),
            'min': float(g.min()),
            'max': float(g.max()),
            'abs_mean': float(g.abs().mean()),
            'abs_max': float(g.abs().max()),
            'norm_l1': float(torch.norm(g, p=1)),
            'norm_l2': float(torch.norm(g, p=2)),
            'num_zeros': int((g == 0).sum()),
            'num_nans': int(torch.isnan(g).sum()),
            'num_infs': int(torch.isinf(g).sum()),
        }

        # Issue detection
        if self.detect_issues:
            issues = []

            if data['stats']['abs_mean'] < self.vanishing_threshold:
                issues.append({
                    'type': 'vanishing_gradient',
                    'severity': 'high',
                    'value': data['stats']['abs_mean'],
                })

            if data['stats']['abs_max'] > self.exploding_threshold:
                issues.append({
                    'type': 'exploding_gradient',
                    'severity': 'high',
                    'value': data['stats']['abs_max'],
                })

            if data['stats']['num_nans'] > 0:
                issues.append({
                    'type': 'nan_gradient',
                    'severity': 'critical',
                    'count': data['stats']['num_nans'],
                })

            if data['stats']['num_infs'] > 0:
                issues.append({
                    'type': 'inf_gradient',
                    'severity': 'critical',
                    'count': data['stats']['num_infs'],
                })

            # Check for dead gradients
            zero_ratio = data['stats']['num_zeros'] / g.numel()
            if zero_ratio > 0.9:
                issues.append({
                    'type': 'mostly_zero_gradient',
                    'severity': 'medium',
                    'zero_ratio': float(zero_ratio),
                })

            data['issues'] = issues

        return data

    def get_gradient_flow_summary(self) -> Dict[str, Any]:
        """Get a summary of gradient flow through the network."""
        if not self.gradient_flow:
            return {}

        # Organize by parameter
        param_gradients: Dict[str, List[Dict[str, Any]]] = {}
        for grad_data in self.gradient_flow:
            param_name = grad_data['param_name']
            if param_name not in param_gradients:
                param_gradients[param_name] = []
            param_gradients[param_name].append(grad_data)

        # Aggregate statistics
        summary = {}
        all_issues = []

        for param_name, grads in param_gradients.items():
            # Average statistics across timesteps
            avg_stats = {
                'mean': np.mean([g['stats']['mean'] for g in grads]),
                'abs_mean': np.mean([g['stats']['abs_mean'] for g in grads]),
                'abs_max': np.max([g['stats']['abs_max'] for g in grads]),
                'norm_l2': np.mean([g['stats']['norm_l2'] for g in grads]),
            }

            # Collect issues
            param_issues = []
            for g in grads:
                if 'issues' in g and g['issues']:
                    param_issues.extend(g['issues'])
                    all_issues.extend(g['issues'])

            summary[param_name] = {
                'stats': avg_stats,
                'num_updates': len(grads),
                'issues': param_issues,
                'has_issues': len(param_issues) > 0,
            }

        # Global summary
        summary['global'] = {
            'total_parameters': len(param_gradients),
            'total_issues': len(all_issues),
            'issue_types': list(set(issue['type'] for issue in all_issues)),
        }

        return summary

    def get_problematic_layers(self) -> List[str]:
        """Get a list of layers with gradient issues."""
        summary = self.get_gradient_flow_summary()

        problematic = []
        for param_name, data in summary.items():
            if param_name == 'global':
                continue

            if data.get('has_issues', False):
                problematic.append(param_name)

        return problematic

    def _matches_pattern(self, name: str, pattern: str) -> bool:
        """Check if a name matches a pattern."""
        import re
        pattern_re = pattern.replace('*', '.*').replace('?', '.')
        return bool(re.match(f'^{pattern_re}$', name))
