"""
Advanced Example: Multi-Model Comparison Analysis

This example demonstrates how to compare multiple models side-by-side using
the LLM Hook Analysis Framework. Useful for:
- A/B testing different architectures
- Comparing model variants (pruned vs original)
- Benchmarking different configurations
- Analyzing trade-offs (accuracy vs speed vs memory)

Author: LLM Hook Framework Team
"""

import torch
import torch.nn as nn
from llm_hooks.core import HookManager
from llm_hooks.pytorch import ForwardHook
from llm_hooks.activations import ActivationMonitor
from llm_hooks.utils import count_parameters, get_model_size, compare_models
from llm_hooks.analysis import ReportGenerator
import matplotlib.pyplot as plt
import seaborn as sns
import time
from typing import Dict, List, Any
import numpy as np

sns.set_style("whitegrid")


# Define multiple model variants for comparison
class BaselineModel(nn.Module):
    """Baseline model - standard architecture."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.fc4(x)
        return x


class CompactModel(nn.Module):
    """Compact model - fewer parameters."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.fc4(x)
        return x


class DeepModel(nn.Module):
    """Deep model - more layers."""

    def __init__(self):
        super().__init__()
        layers = []
        sizes = [128, 256, 256, 256, 256, 128, 10]
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:  # No activation on last layer
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.3))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ModelComparator:
    """Compare multiple models using hooks."""

    def __init__(self, models: Dict[str, nn.Module]):
        """
        Initialize comparator with multiple models.

        Args:
            models: Dictionary mapping model names to model instances
        """
        self.models = models
        self.managers = {}
        self.results = {}
        self.metrics = {}

        # Initialize manager for each model
        for name, model in models.items():
            self.managers[name] = HookManager(name=f"manager_{name}")

    def setup_hooks(self, hook_types: List[str] = None):
        """
        Setup hooks for all models.

        Args:
            hook_types: List of hook types to use ['forward', 'activation']
        """
        if hook_types is None:
            hook_types = ["forward", "activation"]

        for name, manager in self.managers.items():
            print(f"Setting up hooks for {name}...")

            if "forward" in hook_types:
                forward_hook = ForwardHook(compute_stats=True, capture_output=False)
                manager.register(forward_hook)

            if "activation" in hook_types:
                activation_hook = ActivationMonitor(
                    track_dead_neurons=True, dead_neuron_threshold=1e-3
                )
                manager.register(activation_hook)

            manager.apply_to_model(self.models[name])

        print("✓ Hooks configured for all models")

    def run_inference(
        self, input_data: torch.Tensor, num_runs: int = 10, warmup_runs: int = 3
    ):
        """
        Run inference on all models and collect metrics.

        Args:
            input_data: Input tensor for inference
            num_runs: Number of inference runs for timing
            warmup_runs: Number of warmup runs (not counted)
        """
        print(f"\nRunning inference ({num_runs} runs per model)...")

        for name, model in self.models.items():
            print(f"\n  {name}:")
            model.eval()

            # Warmup
            with torch.no_grad():
                for _ in range(warmup_runs):
                    _ = model(input_data)

            # Timed runs
            timings = []
            with torch.no_grad():
                for run_idx in range(num_runs):
                    start_time = time.perf_counter()
                    output = model(input_data)
                    end_time = time.perf_counter()

                    elapsed_ms = (end_time - start_time) * 1000
                    timings.append(elapsed_ms)

                    if (run_idx + 1) % 5 == 0:
                        print(f"    Run {run_idx + 1}/{num_runs}: {elapsed_ms:.2f}ms")

            # Collect results
            self.results[name] = self.managers[name].get_results()

            # Store metrics
            self.metrics[name] = {
                "mean_latency_ms": np.mean(timings),
                "std_latency_ms": np.std(timings),
                "min_latency_ms": np.min(timings),
                "max_latency_ms": np.max(timings),
                "throughput_samples_per_sec": 1000
                / np.mean(timings)
                * input_data.size(0),
            }

        print("\n✓ Inference complete for all models")

    def collect_model_info(self):
        """Collect static information about models."""
        print("\nCollecting model information...")

        for name, model in self.models.items():
            num_params = count_parameters(model)
            num_trainable = count_parameters(model, trainable_only=True)
            model_size_mb = get_model_size(model, unit="MB")

            self.metrics[name].update(
                {
                    "num_parameters": num_params,
                    "num_trainable_parameters": num_trainable,
                    "model_size_mb": model_size_mb,
                }
            )

        print("✓ Model information collected")

    def analyze_activations(self):
        """Analyze activation patterns across models."""
        print("\nAnalyzing activations...")

        for name, results in self.results.items():
            dead_neuron_counts = []
            sparsity_values = []

            for result in results.results:
                if result.hook_type == "activation_monitor":
                    if "dead_neurons" in result.data:
                        dead_neuron_counts.append(result.data["dead_neurons"]["count"])
                    if "sparsity" in result.data:
                        sparsity_values.append(result.data["sparsity"])

            if dead_neuron_counts:
                self.metrics[name]["total_dead_neurons"] = sum(dead_neuron_counts)
                self.metrics[name]["avg_sparsity"] = np.mean(sparsity_values)

        print("✓ Activation analysis complete")

    def print_comparison(self):
        """Print detailed comparison table."""
        print("\n" + "=" * 80)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 80)

        # Define metrics to compare
        metric_labels = {
            "num_parameters": "Parameters",
            "model_size_mb": "Size (MB)",
            "mean_latency_ms": "Latency (ms)",
            "throughput_samples_per_sec": "Throughput (samples/s)",
            "total_dead_neurons": "Dead Neurons",
            "avg_sparsity": "Avg Sparsity",
        }

        # Print header
        model_names = list(self.models.keys())
        print(f"\n{'Metric':<30} " + " ".join(f"{name:>20}" for name in model_names))
        print("-" * 80)

        # Print each metric
        for metric_key, metric_label in metric_labels.items():
            values = []
            for name in model_names:
                value = self.metrics[name].get(metric_key, "N/A")
                if isinstance(value, float):
                    if metric_key == "model_size_mb":
                        values.append(f"{value:>20.2f}")
                    elif "latency" in metric_key or "sparsity" in metric_key:
                        values.append(f"{value:>20.4f}")
                    else:
                        values.append(f"{value:>20.2f}")
                elif isinstance(value, int):
                    values.append(f"{value:>20,}")
                else:
                    values.append(f"{str(value):>20}")

            print(f"{metric_label:<30} " + " ".join(values))

        print("\n" + "=" * 80)

    def visualize_comparison(self, save_path: str = None):
        """Create comparison visualizations."""
        print("\nCreating visualizations...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        model_names = list(self.models.keys())

        # Plot 1: Model Size vs Parameters
        params = [self.metrics[name]["num_parameters"] for name in model_names]
        sizes = [self.metrics[name]["model_size_mb"] for name in model_names]

        axes[0, 0].bar(range(len(model_names)), params, color="steelblue", alpha=0.7)
        axes[0, 0].set_xlabel("Model")
        axes[0, 0].set_ylabel("Parameters")
        axes[0, 0].set_title("Model Size (Parameters)")
        axes[0, 0].set_xticks(range(len(model_names)))
        axes[0, 0].set_xticklabels(model_names, rotation=45, ha="right")
        axes[0, 0].grid(axis="y", alpha=0.3)

        # Plot 2: Inference Latency
        latencies = [self.metrics[name]["mean_latency_ms"] for name in model_names]
        std_latencies = [self.metrics[name]["std_latency_ms"] for name in model_names]

        axes[0, 1].bar(
            range(len(model_names)), latencies, yerr=std_latencies, color="coral", alpha=0.7
        )
        axes[0, 1].set_xlabel("Model")
        axes[0, 1].set_ylabel("Latency (ms)")
        axes[0, 1].set_title("Inference Latency (mean ± std)")
        axes[0, 1].set_xticks(range(len(model_names)))
        axes[0, 1].set_xticklabels(model_names, rotation=45, ha="right")
        axes[0, 1].grid(axis="y", alpha=0.3)

        # Plot 3: Throughput
        throughputs = [
            self.metrics[name]["throughput_samples_per_sec"] for name in model_names
        ]

        axes[1, 0].bar(
            range(len(model_names)), throughputs, color="mediumseagreen", alpha=0.7
        )
        axes[1, 0].set_xlabel("Model")
        axes[1, 0].set_ylabel("Samples/sec")
        axes[1, 0].set_title("Throughput")
        axes[1, 0].set_xticks(range(len(model_names)))
        axes[1, 0].set_xticklabels(model_names, rotation=45, ha="right")
        axes[1, 0].grid(axis="y", alpha=0.3)

        # Plot 4: Dead Neurons (if available)
        dead_neurons = [
            self.metrics[name].get("total_dead_neurons", 0) for name in model_names
        ]

        if any(dead_neurons):
            axes[1, 1].bar(
                range(len(model_names)), dead_neurons, color="crimson", alpha=0.7
            )
            axes[1, 1].set_xlabel("Model")
            axes[1, 1].set_ylabel("Dead Neurons")
            axes[1, 1].set_title("Dead Neuron Count")
            axes[1, 1].set_xticks(range(len(model_names)))
            axes[1, 1].set_xticklabels(model_names, rotation=45, ha="right")
            axes[1, 1].grid(axis="y", alpha=0.3)
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "No dead neuron data available",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
            )
            axes[1, 1].set_xticks([])
            axes[1, 1].set_yticks([])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"  Saved visualization to {save_path}")

        plt.show()
        print("✓ Visualizations created")

    def generate_report(self, output_path: str = "model_comparison_report.json"):
        """Generate comprehensive comparison report."""
        print(f"\nGenerating comparison report...")

        report_data = {
            "models": list(self.models.keys()),
            "metrics": self.metrics,
            "summary": self._generate_summary(),
        }

        # Use ReportGenerator for consistent format
        for name, results in self.results.items():
            report_gen = ReportGenerator(results)
            report_gen.save_report(
                f"detailed_report_{name}.json", report_format="json"
            )

        # Save comparison report
        import json

        with open(output_path, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

        print(f"  Saved comparison report to {output_path}")
        print("✓ Report generation complete")

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary with recommendations."""
        model_names = list(self.models.keys())

        # Find best models for different criteria
        best_speed = min(
            model_names, key=lambda n: self.metrics[n]["mean_latency_ms"]
        )
        best_compact = min(
            model_names, key=lambda n: self.metrics[n]["num_parameters"]
        )
        best_throughput = max(
            model_names, key=lambda n: self.metrics[n]["throughput_samples_per_sec"]
        )

        return {
            "fastest_model": best_speed,
            "most_compact_model": best_compact,
            "highest_throughput_model": best_throughput,
            "recommendations": self._generate_recommendations(),
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        for name in self.models.keys():
            metrics = self.metrics[name]

            # Check for dead neurons
            if metrics.get("total_dead_neurons", 0) > 100:
                recommendations.append(
                    f"{name}: High dead neuron count ({metrics['total_dead_neurons']}). "
                    f"Consider pruning or adjusting architecture."
                )

            # Check model size
            if metrics["model_size_mb"] > 100:
                recommendations.append(
                    f"{name}: Large model size ({metrics['model_size_mb']:.1f}MB). "
                    f"Consider quantization or pruning for deployment."
                )

            # Check sparsity
            if metrics.get("avg_sparsity", 0) > 0.7:
                recommendations.append(
                    f"{name}: High activation sparsity ({metrics['avg_sparsity']:.2%}). "
                    f"Could benefit from sparse matrix operations."
                )

        return recommendations

    def cleanup(self):
        """Clean up all hooks."""
        for manager in self.managers.values():
            manager.remove_all_hooks()
        print("✓ Cleaned up all hooks")


def main():
    """Run model comparison analysis."""
    print("=" * 80)
    print("ADVANCED MODEL COMPARISON ANALYSIS")
    print("=" * 80)

    # Create models
    models = {
        "Baseline": BaselineModel(),
        "Compact": CompactModel(),
        "Deep": DeepModel(),
    }

    # Initialize comparator
    comparator = ModelComparator(models)

    # Setup hooks
    comparator.setup_hooks(hook_types=["forward", "activation"])

    # Collect model information
    comparator.collect_model_info()

    # Create test data
    batch_size = 32
    input_data = torch.randn(batch_size, 128)

    # Run inference and collect metrics
    comparator.run_inference(input_data, num_runs=20, warmup_runs=5)

    # Analyze activations
    comparator.analyze_activations()

    # Print comparison
    comparator.print_comparison()

    # Visualize comparison
    comparator.visualize_comparison(save_path="model_comparison.png")

    # Generate report
    comparator.generate_report("model_comparison_report.json")

    # Cleanup
    comparator.cleanup()

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nFiles generated:")
    print("  - model_comparison.png (visualization)")
    print("  - model_comparison_report.json (metrics)")
    print("  - detailed_report_*.json (per-model details)")


if __name__ == "__main__":
    main()
