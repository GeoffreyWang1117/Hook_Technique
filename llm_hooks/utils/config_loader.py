"""
Configuration file loader for running analysis from YAML configs.
"""

import yaml
import torch
from typing import Dict, Any
from pathlib import Path


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary with configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def run_from_config(config: Dict[str, Any]):
    """
    Run hook analysis from configuration dictionary.

    Args:
        config: Configuration dictionary

    Example config:
        model:
          path: "model.pth"
          type: "pytorch"
        hooks:
          - type: forward
            layer_pattern: "*"
          - type: attention
            record_scores: true
        analysis:
          output_dir: "./output"
          generate_report: true
    """
    from llm_hooks import HookManager
    from llm_hooks.analysis import ReportGenerator

    # Load model
    model_config = config.get('model', {})
    model_path = model_config.get('path')
    model_type = model_config.get('type', 'pytorch')

    print(f"Loading {model_type} model from {model_path}...")
    model = _load_model(model_path, model_type)

    # Setup hooks
    manager = HookManager()
    hooks_config = config.get('hooks', [])

    print(f"Setting up {len(hooks_config)} hooks...")
    for hook_config in hooks_config:
        hook = _create_hook_from_config(hook_config)
        if hook:
            manager.register(hook)

    manager.apply_to_model(model)

    # Run inference/training based on config
    print("Running analysis...")
    data_config = config.get('data', {})
    _run_analysis(model, data_config)

    # Collect results
    results = manager.get_results()
    print(f"Collected {len(results)} results")

    # Generate outputs
    analysis_config = config.get('analysis', {})
    output_dir = analysis_config.get('output_dir', './output')
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if analysis_config.get('generate_report', True):
        report_format = analysis_config.get('report_format', 'both')
        report_gen = ReportGenerator(results)

        if report_format in ['text', 'both']:
            report_gen.save_report(f"{output_dir}/report.txt", format='text')
            print(f"✓ Text report: {output_dir}/report.txt")

        if report_format in ['json', 'both']:
            report_gen.save_report(f"{output_dir}/report.json", format='json')
            print(f"✓ JSON report: {output_dir}/report.json")

    if analysis_config.get('generate_visualizations', True):
        manager.visualize(results, output_dir=f"{output_dir}/visualizations")
        print(f"✓ Visualizations: {output_dir}/visualizations/")

    # Cleanup
    manager.teardown()
    print("Analysis complete!")


def _load_model(model_path: str, model_type: str):
    """Load model based on type."""
    if model_type == 'pytorch':
        return torch.load(model_path)
    elif model_type == 'huggingface':
        from transformers import AutoModel
        return AutoModel.from_pretrained(model_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def _create_hook_from_config(hook_config: Dict[str, Any]):
    """Create hook from configuration."""
    from llm_hooks.core.hook_registry import HookRegistry

    hook_type = hook_config.pop('type')
    hook = HookRegistry.create(hook_type, **hook_config)

    return hook


def _run_analysis(model, data_config: Dict[str, Any]):
    """Run model inference/training for analysis."""
    # Simple dummy inference if no data specified
    if not data_config:
        dummy_input = torch.randn(1, 10)
        with torch.no_grad():
            _ = model(dummy_input)
        return

    # TODO: Add support for custom datasets
    pass
