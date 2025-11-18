"""
Attention pattern monitoring hook.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List
from llm_hooks.core.base_hook import BaseHook
from llm_hooks.core.hook_registry import register_hook
from llm_hooks.pytorch.utils import get_module_by_name
import numpy as np


@register_hook("attention")
class AttentionMonitor(BaseHook):
    """
    Hook for monitoring attention patterns in transformer models.

    This hook captures attention scores, patterns, and statistics to analyze
    how the model attends to different parts of the input.
    """

    def __init__(
        self,
        layer_pattern: str = "*attention*",
        record_scores: bool = True,
        record_patterns: bool = True,
        compute_entropy: bool = True,
        compute_sparsity: bool = True,
        max_samples: int = 100,
        name: Optional[str] = None,
    ):
        """
        Initialize the attention monitor.

        Args:
            layer_pattern: Pattern to match attention layer names.
            record_scores: Whether to record raw attention scores.
            record_patterns: Whether to record attention patterns.
            compute_entropy: Whether to compute attention entropy.
            compute_sparsity: Whether to compute attention sparsity.
            max_samples: Maximum number of samples to record.
            name: Optional name for the hook.
        """
        super().__init__(name=name or "AttentionMonitor")
        self.layer_pattern = layer_pattern
        self.record_scores = record_scores
        self.record_patterns = record_patterns
        self.compute_entropy = compute_entropy
        self.compute_sparsity = compute_sparsity
        self.max_samples = max_samples
        self.sample_count = 0

    def setup(self, model: nn.Module) -> None:
        """Setup attention monitoring hooks."""
        for name, module in model.named_modules():
            if self._matches_pattern(name, self.layer_pattern):
                # Hook forward pass to capture attention
                handle = module.register_forward_hook(
                    self._create_attention_hook(name)
                )
                self._hook_handles.append(handle)

    def teardown(self) -> None:
        """Remove all hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self.sample_count = 0

    def _create_attention_hook(self, layer_name: str):
        """Create an attention monitoring hook."""
        def hook_fn(module: nn.Module, input: tuple, output: Any):
            if not self.enabled or self.sample_count >= self.max_samples:
                return

            # Try to extract attention scores from output
            attention_scores = self._extract_attention_scores(output)

            if attention_scores is not None:
                data = self._analyze_attention(attention_scores)
                data['layer_name'] = layer_name

                result = self.create_result(
                    hook_type='attention',
                    data=data,
                    layer_name=layer_name,
                    metadata={
                        'module_type': module.__class__.__name__,
                        'sample_id': self.sample_count,
                    }
                )
                self.add_result(result)
                self.sample_count += 1

        return hook_fn

    def _extract_attention_scores(self, output: Any) -> Optional[torch.Tensor]:
        """
        Extract attention scores from module output.

        Handles common output formats:
        - Direct tensor
        - Tuple with (output, attention_weights)
        - Dict with 'attentions' key
        """
        if isinstance(output, torch.Tensor):
            # Assume output is attention scores if 4D (batch, heads, seq, seq)
            if len(output.shape) == 4:
                return output
            return None

        elif isinstance(output, tuple):
            # Check each element for attention scores
            for item in output:
                if isinstance(item, torch.Tensor) and len(item.shape) == 4:
                    return item
            return None

        elif isinstance(output, dict):
            if 'attentions' in output:
                return output['attentions']
            elif 'attention_scores' in output:
                return output['attention_scores']

        return None

    def _analyze_attention(self, attention_scores: torch.Tensor) -> Dict[str, Any]:
        """Analyze attention scores and extract statistics."""
        data = {
            'shape': list(attention_scores.shape),  # [batch, heads, seq, seq]
        }

        # Move to CPU for analysis
        scores = attention_scores.detach().cpu().float()
        batch_size, num_heads, seq_len, _ = scores.shape

        data['num_heads'] = num_heads
        data['seq_length'] = seq_len

        # Record raw scores if requested (sampled to save memory)
        if self.record_scores:
            # Only record first sample in batch, first few heads
            max_heads = min(4, num_heads)
            data['scores_sample'] = scores[0, :max_heads].numpy().tolist()

        # Compute statistics per head
        head_stats = []
        for head_idx in range(num_heads):
            head_scores = scores[:, head_idx, :, :]  # [batch, seq, seq]

            stats = {
                'head_id': head_idx,
                'mean': float(head_scores.mean()),
                'std': float(head_scores.std()),
                'min': float(head_scores.min()),
                'max': float(head_scores.max()),
            }

            # Compute entropy (measure of attention distribution)
            if self.compute_entropy:
                # Entropy per position
                entropy = -(head_scores * torch.log(head_scores + 1e-10)).sum(dim=-1)
                stats['entropy_mean'] = float(entropy.mean())
                stats['entropy_std'] = float(entropy.std())

            # Compute sparsity
            if self.compute_sparsity:
                # Sparsity: percentage of weights below threshold
                threshold = 0.01
                sparsity = (head_scores < threshold).float().mean()
                stats['sparsity'] = float(sparsity)

                # Effective sequence length (number of tokens with >threshold attention)
                effective_len = (head_scores > threshold).sum(dim=-1).float().mean()
                stats['effective_length'] = float(effective_len)

            # Attention pattern analysis
            if self.record_patterns:
                # Diagonal dominance (self-attention)
                diagonal = torch.diagonal(head_scores[0], dim1=-2, dim2=-1)
                stats['diagonal_mean'] = float(diagonal.mean())

                # Positional bias
                first_token_attn = head_scores[0, :, 0].mean()
                last_token_attn = head_scores[0, :, -1].mean()
                stats['first_token_attn'] = float(first_token_attn)
                stats['last_token_attn'] = float(last_token_attn)

            head_stats.append(stats)

        data['head_stats'] = head_stats

        # Aggregate statistics across heads
        data['global_stats'] = {
            'mean_entropy': np.mean([s.get('entropy_mean', 0) for s in head_stats]),
            'mean_sparsity': np.mean([s.get('sparsity', 0) for s in head_stats]),
            'head_diversity': float(scores.std(dim=1).mean()),  # How different are heads
        }

        return data

    def _matches_pattern(self, name: str, pattern: str) -> bool:
        """Check if a name matches a pattern."""
        import re
        pattern_re = pattern.replace('*', '.*').replace('?', '.')
        return bool(re.match(f'^{pattern_re}$', name))

    def get_attention_summary(self) -> Dict[str, Any]:
        """Get a summary of all captured attention patterns."""
        results = self.get_results()

        if not results:
            return {}

        # Aggregate statistics
        all_entropies = []
        all_sparsities = []
        layer_stats = {}

        for result in results:
            data = result.data
            layer = data.get('layer_name', 'unknown')

            if layer not in layer_stats:
                layer_stats[layer] = []

            for head_stat in data.get('head_stats', []):
                if 'entropy_mean' in head_stat:
                    all_entropies.append(head_stat['entropy_mean'])
                if 'sparsity' in head_stat:
                    all_sparsities.append(head_stat['sparsity'])

                layer_stats[layer].append(head_stat)

        summary = {
            'total_samples': len(results),
            'layers_monitored': list(layer_stats.keys()),
        }

        if all_entropies:
            summary['entropy'] = {
                'mean': float(np.mean(all_entropies)),
                'std': float(np.std(all_entropies)),
                'min': float(np.min(all_entropies)),
                'max': float(np.max(all_entropies)),
            }

        if all_sparsities:
            summary['sparsity'] = {
                'mean': float(np.mean(all_sparsities)),
                'std': float(np.std(all_sparsities)),
            }

        return summary
