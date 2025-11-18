"""
Benchmark script for measuring hook overhead.
"""

import torch
import torch.nn as nn
import time
import argparse
from pathlib import Path
import json
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_hooks import HookManager
from llm_hooks.pytorch import ForwardHook, BackwardHook
from llm_hooks.attention import AttentionMonitor


def create_test_model(model_type='small'):
    """Create a test model."""
    if model_type == 'small':
        return nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
    elif model_type == 'medium':
        layers = []
        for _ in range(10):
            layers.extend([
                nn.Linear(256, 256),
                nn.ReLU(),
            ])
        layers.append(nn.Linear(256, 10))
        return nn.Sequential(*layers)
    elif model_type == 'transformer':
        return nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=2048,
            batch_first=True,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def benchmark_inference(model, input_data, num_iterations=100):
    """Benchmark inference time."""
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_data)

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.time()
            _ = model(input_data)
            times.append(time.time() - start)

    return times


def run_benchmark(model_type='small', batch_size=32, num_iterations=100):
    """Run comprehensive benchmark."""
    print("=" * 80)
    print(f"Benchmarking: {model_type} model, batch_size={batch_size}")
    print("=" * 80)

    # Create model and data
    model = create_test_model(model_type)
    input_shape = {
        'small': (batch_size, 128),
        'medium': (batch_size, 256),
        'transformer': (batch_size, 16, 512),
    }
    input_data = torch.randn(*input_shape[model_type])

    results = {}

    # 1. Baseline (no hooks)
    print("\n1. Baseline (no hooks)...")
    baseline_times = benchmark_inference(model, input_data, num_iterations)
    baseline_mean = sum(baseline_times) / len(baseline_times)
    results['baseline'] = {
        'mean_ms': baseline_mean * 1000,
        'std_ms': torch.tensor(baseline_times).std().item() * 1000,
        'min_ms': min(baseline_times) * 1000,
        'max_ms': max(baseline_times) * 1000,
    }

    # 2. Forward hook only
    print("2. With Forward Hook...")
    manager = HookManager()
    manager.register(ForwardHook(layer_pattern="*", compute_stats=True, capture_output=False))
    manager.apply_to_model(model)

    forward_times = benchmark_inference(model, input_data, num_iterations)
    forward_mean = sum(forward_times) / len(forward_times)
    results['forward_hook'] = {
        'mean_ms': forward_mean * 1000,
        'overhead_pct': ((forward_mean - baseline_mean) / baseline_mean) * 100,
    }

    manager.teardown()

    # 3. Forward + Backward hook
    print("3. With Forward + Backward Hooks...")
    manager = HookManager()
    manager.register(ForwardHook(layer_pattern="*", compute_stats=True, capture_output=False))
    manager.register(BackwardHook(layer_pattern="*", compute_stats=True))
    manager.apply_to_model(model)

    both_times = benchmark_inference(model, input_data, num_iterations)
    both_mean = sum(both_times) / len(both_times)
    results['both_hooks'] = {
        'mean_ms': both_mean * 1000,
        'overhead_pct': ((both_mean - baseline_mean) / baseline_mean) * 100,
    }

    manager.teardown()

    # 4. Multiple hooks
    print("4. With Multiple Hooks...")
    manager = HookManager()
    manager.register(ForwardHook(layer_pattern="*", compute_stats=True, capture_output=False))
    manager.register(BackwardHook(layer_pattern="*", compute_stats=True))
    if model_type == 'transformer':
        manager.register(AttentionMonitor(max_samples=50))
    manager.apply_to_model(model)

    multi_times = benchmark_inference(model, input_data, num_iterations)
    multi_mean = sum(multi_times) / len(multi_times)
    results['multi_hooks'] = {
        'mean_ms': multi_mean * 1000,
        'overhead_pct': ((multi_mean - baseline_mean) / baseline_mean) * 100,
    }

    manager.teardown()

    # Print results
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(f"\nBaseline: {results['baseline']['mean_ms']:.2f} ms (±{results['baseline']['std_ms']:.2f})")
    print(f"Forward Hook: {results['forward_hook']['mean_ms']:.2f} ms ({results['forward_hook']['overhead_pct']:+.1f}% overhead)")
    print(f"Forward + Backward: {results['both_hooks']['mean_ms']:.2f} ms ({results['both_hooks']['overhead_pct']:+.1f}% overhead)")
    print(f"Multiple Hooks: {results['multi_hooks']['mean_ms']:.2f} ms ({results['multi_hooks']['overhead_pct']:+.1f}% overhead)")
    print("=" * 80)

    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark hook overhead')
    parser.add_argument('--model', choices=['small', 'medium', 'transformer'], default='small',
                        help='Model size to benchmark')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--iterations', type=int, default=100, help='Number of iterations')
    parser.add_argument('--output', type=str, default='./benchmark_results.json',
                        help='Output file for results')

    args = parser.parse_args()

    # Run benchmark
    results = run_benchmark(
        model_type=args.model,
        batch_size=args.batch_size,
        num_iterations=args.iterations
    )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
