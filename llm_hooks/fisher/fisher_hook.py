"""
Fisher information estimation for parameter importance analysis.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from llm_hooks.core.base_hook import BaseHook
from llm_hooks.core.hook_registry import register_hook


@register_hook("fisher")
class FisherHook(BaseHook):
    """
    Hook for estimating Fisher information to identify important parameters.

    Useful for structured pruning and understanding parameter sensitivity.
    """

    def __init__(
        self,
        layer_pattern: str = "*",
        accumulate_samples: int = 100,
        name: Optional[str] = None,
    ):
        """
        Initialize the Fisher hook.

        Args:
            layer_pattern: Pattern to match layers.
            accumulate_samples: Number of samples to accumulate gradients over.
            name: Optional name for the hook.
        """
        super().__init__(name=name or "FisherHook")
        self.layer_pattern = layer_pattern
        self.accumulate_samples = accumulate_samples

        # Fisher information matrices (stored as squared gradients)
        self.fisher_info: Dict[str, torch.Tensor] = {}
        self.sample_count = 0

    def setup(self, model: nn.Module) -> None:
        """Setup Fisher information tracking."""
        # Initialize Fisher information storage
        for name, param in model.named_parameters():
            if param.requires_grad and self._matches_pattern(name, self.layer_pattern):
                self.fisher_info[name] = torch.zeros_like(param.data)

        # Register backward hooks on parameters
        for name, param in model.named_parameters():
            if name in self.fisher_info:
                # Use grad hook on parameter
                handle = param.register_hook(
                    self._create_fisher_hook(name)
                )
                self._hook_handles.append(handle)

    def teardown(self) -> None:
        """Remove all hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self.fisher_info.clear()
        self.sample_count = 0

    def _create_fisher_hook(self, param_name: str):
        """Create a Fisher information accumulation hook."""
        def hook_fn(grad: torch.Tensor):
            if not self.enabled or self.sample_count >= self.accumulate_samples:
                return grad

            # Accumulate squared gradients (Fisher information approximation)
            if param_name in self.fisher_info:
                self.fisher_info[param_name] += grad.detach() ** 2

            return grad

        return hook_fn

    def on_batch_end(self):
        """Call this after each batch to increment sample count."""
        self.sample_count += 1

    def get_fisher_importance(self, normalize: bool = True) -> Dict[str, torch.Tensor]:
        """
        Get Fisher information for all parameters.

        Args:
            normalize: Whether to normalize by sample count.

        Returns:
            Dictionary mapping parameter names to Fisher information.
        """
        if normalize and self.sample_count > 0:
            return {
                name: fisher / self.sample_count
                for name, fisher in self.fisher_info.items()
            }
        return self.fisher_info.copy()

    def get_pruning_mask(
        self,
        pruning_ratio: float,
        granularity: str = "weight"
    ) -> Dict[str, torch.Tensor]:
        """
        Generate pruning masks based on Fisher importance.

        Args:
            pruning_ratio: Fraction of parameters to prune (0-1).
            granularity: Pruning granularity - 'weight', 'neuron', or 'channel'.

        Returns:
            Dictionary of pruning masks (1 = keep, 0 = prune).
        """
        masks = {}
        fisher_scores = self.get_fisher_importance(normalize=True)

        for name, fisher in fisher_scores.items():
            if granularity == "weight":
                # Weight-level pruning
                threshold = torch.quantile(fisher.flatten(), pruning_ratio)
                mask = (fisher > threshold).float()

            elif granularity == "neuron":
                # Neuron-level pruning (for linear layers)
                if len(fisher.shape) == 2:  # [out_features, in_features]
                    neuron_importance = fisher.sum(dim=1)
                    threshold = torch.quantile(neuron_importance, pruning_ratio)
                    neuron_mask = (neuron_importance > threshold).float()
                    mask = neuron_mask.unsqueeze(1).expand_as(fisher)
                else:
                    # Fallback to weight-level
                    threshold = torch.quantile(fisher.flatten(), pruning_ratio)
                    mask = (fisher > threshold).float()

            elif granularity == "channel":
                # Channel-level pruning (for conv layers)
                if len(fisher.shape) == 4:  # [out_ch, in_ch, h, w]
                    channel_importance = fisher.sum(dim=[1, 2, 3])
                    threshold = torch.quantile(channel_importance, pruning_ratio)
                    channel_mask = (channel_importance > threshold).float()
                    mask = channel_mask.view(-1, 1, 1, 1).expand_as(fisher)
                else:
                    # Fallback to weight-level
                    threshold = torch.quantile(fisher.flatten(), pruning_ratio)
                    mask = (fisher > threshold).float()

            else:
                raise ValueError(f"Unknown granularity: {granularity}")

            masks[name] = mask

        return masks

    def get_sensitivity_report(self) -> Dict[str, Any]:
        """Generate a sensitivity report for all parameters."""
        fisher_scores = self.get_fisher_importance(normalize=True)

        report = {}
        for name, fisher in fisher_scores.items():
            fisher_cpu = fisher.cpu()

            report[name] = {
                'shape': list(fisher.shape),
                'mean_importance': float(fisher_cpu.mean()),
                'std_importance': float(fisher_cpu.std()),
                'max_importance': float(fisher_cpu.max()),
                'min_importance': float(fisher_cpu.min()),
                'total_importance': float(fisher_cpu.sum()),
            }

        # Global statistics
        all_importances = torch.cat([f.flatten() for f in fisher_scores.values()])
        report['global'] = {
            'total_parameters': int(all_importances.numel()),
            'mean_importance': float(all_importances.mean()),
            'std_importance': float(all_importances.std()),
        }

        return report

    def _matches_pattern(self, name: str, pattern: str) -> bool:
        """Check if a name matches a pattern."""
        import re
        pattern_re = pattern.replace('*', '.*').replace('?', '.')
        return bool(re.match(f'^{pattern_re}$', name))
