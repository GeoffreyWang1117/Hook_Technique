"""
Command-line interface for LLM Hook Analysis Framework.
"""

import click
import sys
import os


@click.group()
@click.version_option(version='0.1.0')
def cli():
    """LLM Hook Analysis Framework - Command Line Interface"""
    pass


@cli.command()
@click.argument('model_path')
@click.option('--hooks', '-h', multiple=True, help='Hooks to use (forward, backward, attention, etc.)')
@click.option('--layers', '-l', help='Layer pattern to hook')
@click.option('--output', '-o', default='./analysis', help='Output directory for results')
@click.option('--format', '-f', type=click.Choice(['text', 'json', 'both']), default='both', help='Report format')
@click.option('--visualize/--no-visualize', default=True, help='Generate visualizations')
def analyze(model_path, hooks, layers, output, format, visualize):
    """Analyze a PyTorch model with hooks."""
    click.echo(f"Analyzing model: {model_path}")
    click.echo(f"Hooks: {hooks if hooks else 'auto-detect'}")
    click.echo(f"Layers: {layers if layers else 'all'}")
    click.echo(f"Output: {output}")

    try:
        import torch
        from llm_hooks import HookManager
        from llm_hooks.analysis import ReportGenerator

        # Load model
        click.echo("Loading model...")
        model = torch.load(model_path)
        model.eval()

        # Setup hooks
        manager = HookManager()

        if not hooks:
            # Auto-detect and use common hooks
            from llm_hooks.pytorch import ForwardHook
            from llm_hooks.attention import AttentionMonitor
            from llm_hooks.activations import ActivationMonitor

            manager.register(ForwardHook(layer_pattern=layers or "*"))
            manager.register(AttentionMonitor())
            manager.register(ActivationMonitor())
        else:
            # Use specified hooks
            for hook_name in hooks:
                hook = _create_hook(hook_name, layers)
                if hook:
                    manager.register(hook)

        manager.apply_to_model(model)
        click.echo(f"Applied {len(manager.hooks)} hooks")

        # Run dummy inference for analysis
        click.echo("Running analysis...")
        dummy_input = torch.randn(1, 10)  # Adjust based on model
        with torch.no_grad():
            _ = model(dummy_input)

        # Generate report
        results = manager.get_results()
        click.echo(f"Collected {len(results)} results")

        os.makedirs(output, exist_ok=True)

        report_gen = ReportGenerator(results)
        if format in ['text', 'both']:
            report_gen.save_report(f"{output}/report.txt", format='text')
            click.echo(f"✓ Text report saved to {output}/report.txt")

        if format in ['json', 'both']:
            report_gen.save_report(f"{output}/report.json", format='json')
            click.echo(f"✓ JSON report saved to {output}/report.json")

        if visualize:
            manager.visualize(results, output_dir=f"{output}/visualizations")
            click.echo(f"✓ Visualizations saved to {output}/visualizations/")

        manager.teardown()
        click.echo("Analysis complete!")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('config_path')
def run(config_path):
    """Run analysis from a configuration file."""
    click.echo(f"Running analysis from config: {config_path}")

    try:
        import yaml
        from llm_hooks.utils.config_loader import load_config, run_from_config

        # Load config
        config = load_config(config_path)
        click.echo(f"Loaded configuration")

        # Run analysis
        run_from_config(config)

        click.echo("Analysis complete!")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('model_path')
@click.option('--output', '-o', default='./benchmark', help='Output directory')
@click.option('--iterations', '-n', default=100, help='Number of iterations')
def benchmark(model_path, output, iterations):
    """Benchmark model performance with and without hooks."""
    click.echo(f"Benchmarking model: {model_path}")

    try:
        import torch
        import time
        from llm_hooks import HookManager
        from llm_hooks.pytorch import ForwardHook

        # Load model
        model = torch.load(model_path)
        model.eval()

        dummy_input = torch.randn(1, 10)

        # Baseline (no hooks)
        click.echo("Running baseline (no hooks)...")
        times_baseline = []
        with torch.no_grad():
            for _ in range(iterations):
                start = time.time()
                _ = model(dummy_input)
                times_baseline.append(time.time() - start)

        baseline_avg = sum(times_baseline) / len(times_baseline)

        # With hooks
        click.echo("Running with hooks...")
        manager = HookManager()
        manager.register(ForwardHook(layer_pattern="*"))
        manager.apply_to_model(model)

        times_hooked = []
        with torch.no_grad():
            for _ in range(iterations):
                start = time.time()
                _ = model(dummy_input)
                times_hooked.append(time.time() - start)

        hooked_avg = sum(times_hooked) / len(times_hooked)
        overhead = (hooked_avg - baseline_avg) / baseline_avg * 100

        manager.teardown()

        # Report
        click.echo("\n" + "="*60)
        click.echo("BENCHMARK RESULTS")
        click.echo("="*60)
        click.echo(f"Baseline: {baseline_avg*1000:.2f} ms")
        click.echo(f"With hooks: {hooked_avg*1000:.2f} ms")
        click.echo(f"Overhead: {overhead:.1f}%")
        click.echo("="*60)

        # Save results
        os.makedirs(output, exist_ok=True)
        with open(f"{output}/benchmark.txt", "w") as f:
            f.write(f"Baseline: {baseline_avg*1000:.2f} ms\n")
            f.write(f"With hooks: {hooked_avg*1000:.2f} ms\n")
            f.write(f"Overhead: {overhead:.1f}%\n")

        click.echo(f"\nResults saved to {output}/benchmark.txt")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def list_hooks():
    """List all available hook types."""
    click.echo("Available Hooks:")
    click.echo("="*60)

    hooks = [
        ("forward", "Forward pass hook", "llm_hooks.pytorch.ForwardHook"),
        ("backward", "Backward pass hook", "llm_hooks.pytorch.BackwardHook"),
        ("attention", "Attention monitoring", "llm_hooks.attention.AttentionMonitor"),
        ("kv_cache", "KV cache monitoring", "llm_hooks.attention.KVCacheMonitor"),
        ("activation", "Activation monitoring", "llm_hooks.activations.ActivationMonitor"),
        ("fisher", "Fisher information", "llm_hooks.fisher.FisherHook"),
        ("gradient", "Gradient tracking", "llm_hooks.gradients.GradientTracker"),
        ("cuda", "CUDA profiling", "llm_hooks.cuda.CUDAProfiler"),
        ("tensorrt", "TensorRT logging", "llm_hooks.tensorrt.TensorRTLogger"),
    ]

    for name, desc, cls in hooks:
        click.echo(f"\n{name}")
        click.echo(f"  Description: {desc}")
        click.echo(f"  Class: {cls}")


@cli.command()
@click.argument('output_path')
def generate_config(output_path):
    """Generate a sample configuration file."""
    config = """# LLM Hook Analysis Configuration

# Model configuration
model:
  path: "path/to/model.pth"
  type: "pytorch"  # pytorch, onnx, tensorrt

# Hooks to use
hooks:
  - type: forward
    layer_pattern: "transformer.layer.*"
    compute_stats: true

  - type: attention
    record_scores: true
    compute_entropy: true

  - type: activation
    dead_neuron_threshold: 0.01

# Analysis configuration
analysis:
  output_dir: "./analysis_output"
  generate_report: true
  report_format: "both"  # text, json, both
  generate_visualizations: true

# Performance options
performance:
  max_samples: 100
  clear_interval: 10
"""

    with open(output_path, 'w') as f:
        f.write(config)

    click.echo(f"Sample configuration saved to {output_path}")


def _create_hook(hook_name, layer_pattern):
    """Helper to create hook from name."""
    from llm_hooks.core.hook_registry import HookRegistry

    hook = HookRegistry.create(hook_name)
    if hook and hasattr(hook, 'layer_pattern'):
        hook.layer_pattern = layer_pattern or "*"

    return hook


def main():
    """Main entry point."""
    cli()


if __name__ == '__main__':
    main()
