"""
CUDA kernel profiling using PyTorch profiler.
"""

import torch
from typing import Optional, Dict, Any, List
from llm_hooks.core.base_hook import BaseHook
from llm_hooks.core.hook_registry import register_hook
import time


@register_hook("cuda")
class CUDAProfiler(BaseHook):
    """
    CUDA kernel profiling hook.

    Uses PyTorch's profiler to track CUDA kernel execution,
    memory usage, and performance bottlenecks.
    """

    def __init__(
        self,
        with_stack: bool = True,
        with_flops: bool = True,
        profile_memory: bool = True,
        record_shapes: bool = True,
        name: Optional[str] = None,
    ):
        """
        Initialize the CUDA profiler.

        Args:
            with_stack: Whether to record call stacks.
            with_flops: Whether to estimate FLOPs.
            profile_memory: Whether to profile memory usage.
            record_shapes: Whether to record tensor shapes.
            name: Optional name for the hook.
        """
        super().__init__(name=name or "CUDAProfiler")
        self.with_stack = with_stack
        self.with_flops = with_flops
        self.profile_memory = profile_memory
        self.record_shapes = record_shapes

        self.profiler: Optional[torch.profiler.profile] = None
        self.profiling_results: List[Any] = []

    def setup(self, model: torch.nn.Module) -> None:
        """Setup CUDA profiling."""
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, profiler will run in CPU mode")

        # Create profiler
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ] if torch.cuda.is_available() else [torch.profiler.ProfilerActivity.CPU],
            with_stack=self.with_stack,
            with_flops=self.with_flops,
            profile_memory=self.profile_memory,
            record_shapes=self.record_shapes,
        )

    def teardown(self) -> None:
        """Stop profiling and cleanup."""
        if self.profiler is not None:
            self.profiler.__exit__(None, None, None)
            self.profiler = None

    def start_profiling(self):
        """Start profiling context."""
        if self.profiler is not None:
            self.profiler.__enter__()

    def stop_profiling(self):
        """Stop profiling and collect results."""
        if self.profiler is not None:
            self.profiler.__exit__(None, None, None)

            # Get profiling results
            events = self.profiler.key_averages()
            self.profiling_results.append(events)

            # Create result
            data = self._analyze_profile(events)
            result = self.create_result(
                hook_type='cuda_profile',
                data=data,
            )
            self.add_result(result)

    def _analyze_profile(self, events) -> Dict[str, Any]:
        """Analyze profiling events."""
        data = {
            'events': [],
            'total_cuda_time': 0.0,
            'total_cpu_time': 0.0,
            'total_memory': 0,
        }

        for evt in events:
            event_data = {
                'key': evt.key,
                'cpu_time': evt.cpu_time_total,
                'cuda_time': evt.cuda_time_total if torch.cuda.is_available() else 0,
                'count': evt.count,
                'cpu_time_avg': evt.cpu_time_total / evt.count if evt.count > 0 else 0,
            }

            if self.with_flops and hasattr(evt, 'flops'):
                event_data['flops'] = evt.flops

            if self.profile_memory and hasattr(evt, 'cpu_memory_usage'):
                event_data['cpu_memory'] = evt.cpu_memory_usage
                event_data['cuda_memory'] = evt.cuda_memory_usage if torch.cuda.is_available() else 0

            data['events'].append(event_data)
            data['total_cpu_time'] += evt.cpu_time_total
            if torch.cuda.is_available():
                data['total_cuda_time'] += evt.cuda_time_total

        # Sort by CUDA time
        data['events'].sort(key=lambda x: x.get('cuda_time', 0), reverse=True)

        # Keep top 50 events
        data['top_events'] = data['events'][:50]
        data['num_events'] = len(data['events'])

        return data

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of performance metrics."""
        results = self.get_results()

        if not results:
            return {}

        total_cuda_time = sum(r.data['total_cuda_time'] for r in results)
        total_cpu_time = sum(r.data['total_cpu_time'] for r in results)

        summary = {
            'total_cuda_time_ms': total_cuda_time / 1000.0,  # Convert to ms
            'total_cpu_time_ms': total_cpu_time / 1000.0,
            'num_profiles': len(results),
        }

        # Find most expensive kernels
        all_events = []
        for r in results:
            all_events.extend(r.data.get('events', []))

        # Aggregate by key
        kernel_times = {}
        for evt in all_events:
            key = evt['key']
            if key not in kernel_times:
                kernel_times[key] = {'cuda_time': 0, 'cpu_time': 0, 'count': 0}

            kernel_times[key]['cuda_time'] += evt['cuda_time']
            kernel_times[key]['cpu_time'] += evt['cpu_time']
            kernel_times[key]['count'] += evt['count']

        # Top kernels
        top_kernels = sorted(
            kernel_times.items(),
            key=lambda x: x[1]['cuda_time'],
            reverse=True
        )[:20]

        summary['top_kernels'] = [
            {
                'name': name,
                'cuda_time_ms': stats['cuda_time'] / 1000.0,
                'cpu_time_ms': stats['cpu_time'] / 1000.0,
                'count': stats['count'],
            }
            for name, stats in top_kernels
        ]

        return summary

    def export_trace(self, filename: str):
        """Export profiling trace to file."""
        if self.profiler is not None:
            self.profiler.export_chrome_trace(filename)
            print(f"Exported trace to {filename}")
