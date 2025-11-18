"""
Bottleneck detection and performance analysis.
"""

from typing import Dict, Any, List
from llm_hooks.core.hook_result import HookResultCollection
import numpy as np


class BottleneckAnalyzer:
    """
    Analyze hook results to identify performance bottlenecks.
    """

    def __init__(self, results: HookResultCollection):
        """
        Initialize the analyzer.

        Args:
            results: Collection of hook results to analyze.
        """
        self.results = results

    def identify_slow_layers(self, threshold_percentile: float = 90) -> List[str]:
        """
        Identify layers that are slower than the threshold.

        Args:
            threshold_percentile: Percentile threshold for slow layer detection.

        Returns:
            List of slow layer names.
        """
        # Extract timing information from CUDA profiler results
        cuda_results = self.results.filter_by_type('cuda_profile')

        if not cuda_results:
            return []

        layer_times = {}
        for result in cuda_results:
            for event in result.data.get('events', []):
                key = event['key']
                cuda_time = event.get('cuda_time', 0)

                if key not in layer_times:
                    layer_times[key] = []
                layer_times[key].append(cuda_time)

        # Calculate average times
        avg_times = {k: np.mean(v) for k, v in layer_times.items()}

        # Find threshold
        all_times = list(avg_times.values())
        threshold = np.percentile(all_times, threshold_percentile)

        # Identify slow layers
        slow_layers = [
            layer for layer, time in avg_times.items()
            if time > threshold
        ]

        return slow_layers

    def detect_gradient_issues(self) -> Dict[str, List[str]]:
        """
        Detect gradient-related issues.

        Returns:
            Dictionary mapping issue types to affected layers.
        """
        gradient_results = self.results.filter_by_type('gradient')

        issues = {
            'vanishing': [],
            'exploding': [],
            'nan': [],
            'inf': [],
        }

        for result in gradient_results:
            layer_name = result.layer_name
            result_issues = result.data.get('issues', [])

            for issue in result_issues:
                issue_type = issue['type']

                if 'vanishing' in issue_type:
                    issues['vanishing'].append(layer_name)
                elif 'exploding' in issue_type:
                    issues['exploding'].append(layer_name)
                elif 'nan' in issue_type:
                    issues['nan'].append(layer_name)
                elif 'inf' in issue_type:
                    issues['inf'].append(layer_name)

        return issues

    def find_dead_neurons(self) -> Dict[str, int]:
        """
        Find layers with dead neurons.

        Returns:
            Dictionary mapping layer names to dead neuron counts.
        """
        activation_results = self.results.filter_by_type('activation')

        dead_neurons = {}
        for result in activation_results:
            layer_name = result.layer_name
            dead_info = result.data.get('dead_neurons', {})

            if dead_info.get('count', 0) > 0:
                dead_neurons[layer_name] = dead_info['count']

        return dead_neurons

    def analyze_memory_usage(self) -> Dict[str, Any]:
        """
        Analyze memory usage patterns.

        Returns:
            Dictionary with memory analysis.
        """
        kv_cache_results = self.results.filter_by_type('kv_cache')

        if not kv_cache_results:
            return {}

        total_memory = []
        cache_sizes = []

        for result in kv_cache_results:
            memory_mb = result.data.get('memory_mb', {}).get('total', 0)
            cache_len = result.data.get('cache_sequence_length', 0)

            total_memory.append(memory_mb)
            cache_sizes.append(cache_len)

        return {
            'total_memory_mb': sum(total_memory),
            'avg_memory_mb': np.mean(total_memory) if total_memory else 0,
            'max_memory_mb': max(total_memory) if total_memory else 0,
            'avg_cache_size': np.mean(cache_sizes) if cache_sizes else 0,
            'max_cache_size': max(cache_sizes) if cache_sizes else 0,
        }

    def generate_summary(self) -> Dict[str, Any]:
        """
        Generate a comprehensive bottleneck analysis summary.

        Returns:
            Dictionary with analysis summary.
        """
        return {
            'slow_layers': self.identify_slow_layers(),
            'gradient_issues': self.detect_gradient_issues(),
            'dead_neurons': self.find_dead_neurons(),
            'memory_analysis': self.analyze_memory_usage(),
            'total_results': len(self.results),
        }
