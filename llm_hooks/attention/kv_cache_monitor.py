"""
KV Cache monitoring and optimization hook.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
from llm_hooks.core.base_hook import BaseHook
from llm_hooks.core.hook_registry import register_hook
import numpy as np


@register_hook("kv_cache")
class KVCacheMonitor(BaseHook):
    """
    Hook for monitoring KV cache usage and efficiency.

    Tracks key-value cache sizes, memory usage, and access patterns
    to optimize inference performance.
    """

    def __init__(
        self,
        layer_pattern: str = "*attention*",
        track_size: bool = True,
        track_memory: bool = True,
        track_reuse: bool = True,
        name: Optional[str] = None,
    ):
        """
        Initialize the KV cache monitor.

        Args:
            layer_pattern: Pattern to match attention layers with KV cache.
            track_size: Whether to track cache size growth.
            track_memory: Whether to track memory usage.
            track_reuse: Whether to track cache reuse patterns.
            name: Optional name for the hook.
        """
        super().__init__(name=name or "KVCacheMonitor")
        self.layer_pattern = layer_pattern
        self.track_size = track_size
        self.track_memory = track_memory
        self.track_reuse = track_reuse

        # Internal state
        self.cache_sizes: Dict[str, List[int]] = {}
        self.cache_memory: Dict[str, List[float]] = {}

    def setup(self, model: nn.Module) -> None:
        """Setup KV cache monitoring hooks."""
        for name, module in model.named_modules():
            if self._matches_pattern(name, self.layer_pattern):
                handle = module.register_forward_hook(
                    self._create_kv_hook(name)
                )
                self._hook_handles.append(handle)

    def teardown(self) -> None:
        """Remove all hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self.cache_sizes.clear()
        self.cache_memory.clear()

    def _create_kv_hook(self, layer_name: str):
        """Create a KV cache monitoring hook."""
        def hook_fn(module: nn.Module, input: tuple, output: Any):
            if not self.enabled:
                return

            # Try to extract KV cache info
            kv_info = self._extract_kv_cache_info(module, input, output)

            if kv_info:
                data = self._analyze_kv_cache(layer_name, kv_info)

                result = self.create_result(
                    hook_type='kv_cache',
                    data=data,
                    layer_name=layer_name,
                    metadata={
                        'module_type': module.__class__.__name__,
                    }
                )
                self.add_result(result)

        return hook_fn

    def _extract_kv_cache_info(
        self,
        module: nn.Module,
        input: tuple,
        output: Any
    ) -> Optional[Dict[str, Any]]:
        """
        Extract KV cache information from module.

        This is highly model-specific. Common patterns:
        - Cache stored in module attributes (e.g., module.past_key_value)
        - Cache passed in inputs/outputs as tuples
        - Cache in dict outputs
        """
        info = {}

        # Check module attributes
        if hasattr(module, 'past_key_value') and module.past_key_value is not None:
            key_cache, value_cache = module.past_key_value
            info['key_cache'] = key_cache
            info['value_cache'] = value_cache

        # Check output for cache
        elif isinstance(output, tuple) and len(output) > 1:
            # Common pattern: (output, (key, value))
            if isinstance(output[1], tuple) and len(output[1]) == 2:
                key_cache, value_cache = output[1]
                if isinstance(key_cache, torch.Tensor) and isinstance(value_cache, torch.Tensor):
                    info['key_cache'] = key_cache
                    info['value_cache'] = value_cache

        # Check dict output
        elif isinstance(output, dict):
            if 'past_key_value' in output:
                past_kv = output['past_key_value']
                if isinstance(past_kv, tuple) and len(past_kv) == 2:
                    info['key_cache'] = past_kv[0]
                    info['value_cache'] = past_kv[1]

        return info if info else None

    def _analyze_kv_cache(self, layer_name: str, kv_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze KV cache information."""
        data = {}

        key_cache = kv_info.get('key_cache')
        value_cache = kv_info.get('value_cache')

        if key_cache is not None:
            # Track size
            if self.track_size:
                cache_seq_len = key_cache.shape[-2]  # Typically [batch, heads, seq, dim]
                data['cache_sequence_length'] = cache_seq_len

                if layer_name not in self.cache_sizes:
                    self.cache_sizes[layer_name] = []
                self.cache_sizes[layer_name].append(cache_seq_len)

                data['cache_growth'] = {
                    'current': cache_seq_len,
                    'max': max(self.cache_sizes[layer_name]),
                    'avg': float(np.mean(self.cache_sizes[layer_name])),
                }

            # Track memory
            if self.track_memory:
                key_memory = key_cache.element_size() * key_cache.numel()
                value_memory = value_cache.element_size() * value_cache.numel()
                total_memory = key_memory + value_memory

                data['memory_bytes'] = {
                    'key': key_memory,
                    'value': value_memory,
                    'total': total_memory,
                }

                data['memory_mb'] = {
                    'total': total_memory / (1024 ** 2),
                }

                if layer_name not in self.cache_memory:
                    self.cache_memory[layer_name] = []
                self.cache_memory[layer_name].append(total_memory)

            # Cache statistics
            data['cache_stats'] = {
                'key_shape': list(key_cache.shape),
                'value_shape': list(value_cache.shape),
                'key_dtype': str(key_cache.dtype),
                'value_dtype': str(value_cache.dtype),
            }

            # Analyze cache content
            key_cpu = key_cache.detach().cpu().float()
            value_cpu = value_cache.detach().cpu().float()

            data['content_stats'] = {
                'key_norm': float(torch.norm(key_cpu)),
                'value_norm': float(torch.norm(value_cpu)),
                'key_mean': float(key_cpu.mean()),
                'value_mean': float(value_cpu.mean()),
                'key_std': float(key_cpu.std()),
                'value_std': float(value_cpu.std()),
            }

        return data

    def _matches_pattern(self, name: str, pattern: str) -> bool:
        """Check if a name matches a pattern."""
        import re
        pattern_re = pattern.replace('*', '.*').replace('?', '.')
        return bool(re.match(f'^{pattern_re}$', name))

    def get_cache_summary(self) -> Dict[str, Any]:
        """Get a summary of KV cache usage."""
        summary = {
            'layers_monitored': list(self.cache_sizes.keys()),
        }

        # Size statistics
        if self.cache_sizes:
            all_sizes = []
            for layer, sizes in self.cache_sizes.items():
                all_sizes.extend(sizes)

            summary['size_stats'] = {
                'mean': float(np.mean(all_sizes)),
                'max': int(np.max(all_sizes)),
                'min': int(np.min(all_sizes)),
                'std': float(np.std(all_sizes)),
            }

        # Memory statistics
        if self.cache_memory:
            all_memory = []
            for layer, memory in self.cache_memory.items():
                all_memory.extend(memory)

            total_mb = sum(all_memory) / (1024 ** 2)
            summary['memory_stats'] = {
                'total_mb': float(total_mb),
                'mean_mb': float(np.mean(all_memory) / (1024 ** 2)),
                'max_mb': float(np.max(all_memory) / (1024 ** 2)),
            }

        # Per-layer breakdown
        layer_breakdown = {}
        for layer in self.cache_sizes.keys():
            layer_breakdown[layer] = {
                'max_size': max(self.cache_sizes[layer]) if self.cache_sizes[layer] else 0,
                'avg_size': float(np.mean(self.cache_sizes[layer])) if self.cache_sizes[layer] else 0,
            }

            if layer in self.cache_memory:
                layer_breakdown[layer]['total_memory_mb'] = sum(self.cache_memory[layer]) / (1024 ** 2)

        summary['layer_breakdown'] = layer_breakdown

        return summary
