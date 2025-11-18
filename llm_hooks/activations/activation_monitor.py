"""
Activation monitoring for dead neuron detection and distribution analysis.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
from llm_hooks.core.base_hook import BaseHook
from llm_hooks.core.hook_registry import register_hook
from llm_hooks.pytorch.utils import get_module_by_name
import numpy as np


@register_hook("activation")
class ActivationMonitor(BaseHook):
    """
    Monitor activation statistics and detect dead neurons.

    Tracks activation distributions, sparsity, and identifies neurons
    that consistently produce zero or near-zero outputs.
    """

    def __init__(
        self,
        layer_pattern: str = "*",
        dead_neuron_threshold: float = 0.01,
        track_distribution: bool = True,
        track_sparsity: bool = True,
        name: Optional[str] = None,
    ):
        """
        Initialize the activation monitor.

        Args:
            layer_pattern: Pattern to match layers.
            dead_neuron_threshold: Threshold for dead neuron detection.
            track_distribution: Whether to track activation distributions.
            track_sparsity: Whether to track activation sparsity.
            name: Optional name for the hook.
        """
        super().__init__(name=name or "ActivationMonitor")
        self.layer_pattern = layer_pattern
        self.dead_neuron_threshold = dead_neuron_threshold
        self.track_distribution = track_distribution
        self.track_sparsity = track_sparsity

        # Accumulated statistics
        self.neuron_activations: Dict[str, List[torch.Tensor]] = {}

    def setup(self, model: nn.Module) -> None:
        """Setup activation monitoring hooks."""
        for name, module in model.named_modules():
            if self._should_hook(name, module):
                handle = module.register_forward_hook(
                    self._create_activation_hook(name)
                )
                self._hook_handles.append(handle)

    def teardown(self) -> None:
        """Remove all hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self.neuron_activations.clear()

    def _should_hook(self, name: str, module: nn.Module) -> bool:
        """Check if a module should be hooked."""
        # Only hook leaf modules (no children)
        if len(list(module.children())) > 0:
            return False

        # Check pattern
        if not self._matches_pattern(name, self.layer_pattern):
            return False

        # Only hook layers that produce activations
        return isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.ReLU, nn.GELU))

    def _create_activation_hook(self, layer_name: str):
        """Create an activation monitoring hook."""
        def hook_fn(module: nn.Module, input: tuple, output: torch.Tensor):
            if not self.enabled:
                return

            if isinstance(output, torch.Tensor):
                data = self._analyze_activations(layer_name, output)

                result = self.create_result(
                    hook_type='activation',
                    data=data,
                    layer_name=layer_name,
                    metadata={
                        'module_type': module.__class__.__name__,
                    }
                )
                self.add_result(result)

        return hook_fn

    def _analyze_activations(self, layer_name: str, activations: torch.Tensor) -> Dict[str, Any]:
        """Analyze activation tensor."""
        data = {
            'shape': list(activations.shape),
            'dtype': str(activations.dtype),
        }

        # Move to CPU for analysis
        act = activations.detach().cpu().float()

        # Basic statistics
        data['stats'] = {
            'mean': float(act.mean()),
            'std': float(act.std()),
            'min': float(act.min()),
            'max': float(act.max()),
            'abs_mean': float(act.abs().mean()),
        }

        # Sparsity analysis
        if self.track_sparsity:
            zero_ratio = (act == 0).float().mean()
            near_zero_ratio = (act.abs() < 0.01).float().mean()

            data['sparsity'] = {
                'zero_ratio': float(zero_ratio),
                'near_zero_ratio': float(near_zero_ratio),
                'active_ratio': float(1 - zero_ratio),
            }

        # Distribution analysis
        if self.track_distribution:
            flat = act.flatten()
            if len(flat) > 0:
                data['distribution'] = {
                    'percentile_25': float(torch.quantile(flat, 0.25)),
                    'percentile_50': float(torch.quantile(flat, 0.50)),
                    'percentile_75': float(torch.quantile(flat, 0.75)),
                    'percentile_95': float(torch.quantile(flat, 0.95)),
                    'percentile_99': float(torch.quantile(flat, 0.99)),
                }

                # Histogram
                hist, bin_edges = torch.histogram(flat, bins=20)
                data['histogram'] = {
                    'counts': hist.tolist(),
                    'bin_edges': bin_edges.tolist(),
                }

        # Dead neuron detection (for linear/conv layers)
        if len(act.shape) >= 2:
            # Aggregate over batch and spatial dimensions, keep channel dimension
            if len(act.shape) == 2:  # [batch, features]
                neuron_means = act.abs().mean(dim=0)
            elif len(act.shape) == 3:  # [batch, seq, features]
                neuron_means = act.abs().mean(dim=[0, 1])
            elif len(act.shape) == 4:  # [batch, channels, h, w]
                neuron_means = act.abs().mean(dim=[0, 2, 3])
            else:
                neuron_means = None

            if neuron_means is not None:
                # Store for accumulated analysis
                if layer_name not in self.neuron_activations:
                    self.neuron_activations[layer_name] = []
                self.neuron_activations[layer_name].append(neuron_means)

                # Detect dead neurons
                dead_mask = neuron_means < self.dead_neuron_threshold
                num_dead = dead_mask.sum().item()
                total_neurons = len(neuron_means)

                data['dead_neurons'] = {
                    'count': int(num_dead),
                    'total': int(total_neurons),
                    'ratio': float(num_dead / total_neurons) if total_neurons > 0 else 0,
                    'dead_indices': dead_mask.nonzero().flatten().tolist()[:100],  # Limit output
                }

        return data

    def _matches_pattern(self, name: str, pattern: str) -> bool:
        """Check if a name matches a pattern."""
        import re
        pattern_re = pattern.replace('*', '.*').replace('?', '.')
        return bool(re.match(f'^{pattern_re}$', name))

    def get_dead_neuron_report(self) -> Dict[str, Any]:
        """Generate a report of dead neurons across all layers."""
        report = {}

        for layer_name, activations_list in self.neuron_activations.items():
            if not activations_list:
                continue

            # Average across all samples
            avg_activations = torch.stack(activations_list).mean(dim=0)

            # Find consistently dead neurons
            dead_mask = avg_activations < self.dead_neuron_threshold
            num_dead = dead_mask.sum().item()
            total = len(avg_activations)

            report[layer_name] = {
                'total_neurons': int(total),
                'dead_neurons': int(num_dead),
                'dead_ratio': float(num_dead / total) if total > 0 else 0,
                'mean_activation': float(avg_activations.mean()),
                'std_activation': float(avg_activations.std()),
            }

        return report
