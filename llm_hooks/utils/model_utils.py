"""
Model utility functions.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Count the number of parameters in a model.

    Args:
        model: PyTorch model
        trainable_only: If True, only count trainable parameters

    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_model_size(model: nn.Module, unit: str = 'MB') -> float:
    """
    Get the size of a model in memory.

    Args:
        model: PyTorch model
        unit: Unit for size ('B', 'KB', 'MB', 'GB')

    Returns:
        Model size in specified unit
    """
    # Calculate parameter size
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()

    # Calculate buffer size
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()

    total_size = param_size + buffer_size

    # Convert to specified unit
    units = {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}
    return total_size / units.get(unit, 1)


def print_model_summary(model: nn.Module, verbose: bool = True):
    """
    Print a summary of the model architecture and parameters.

    Args:
        model: PyTorch model
        verbose: If True, print detailed layer information
    """
    print("=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)

    # Overall statistics
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    model_size = get_model_size(model, unit='MB')

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"Model size: {model_size:.2f} MB")

    if verbose:
        print("\nLayer-wise breakdown:")
        print("-" * 80)
        print(f"{'Layer Name':<40} {'Type':<20} {'Parameters':>15}")
        print("-" * 80)

        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                num_params = sum(p.numel() for p in module.parameters())
                if num_params > 0:
                    module_type = module.__class__.__name__
                    print(f"{name:<40} {module_type:<20} {num_params:>15,}")

    print("=" * 80)


def get_layer_info(model: nn.Module) -> Dict[str, Dict[str, Any]]:
    """
    Get detailed information about each layer in the model.

    Args:
        model: PyTorch model

    Returns:
        Dictionary mapping layer names to their info
    """
    layer_info = {}

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            info = {
                'type': module.__class__.__name__,
                'num_parameters': sum(p.numel() for p in module.parameters()),
                'trainable': any(p.requires_grad for p in module.parameters()),
            }

            # Add layer-specific info
            if isinstance(module, nn.Linear):
                info['in_features'] = module.in_features
                info['out_features'] = module.out_features

            elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                info['in_channels'] = module.in_channels
                info['out_channels'] = module.out_channels
                info['kernel_size'] = module.kernel_size

            layer_info[name] = info

    return layer_info


def compare_models(model1: nn.Module, model2: nn.Module, name1: str = "Model 1", name2: str = "Model 2"):
    """
    Compare two models and print statistics.

    Args:
        model1: First model
        model2: Second model
        name1: Name for first model
        name2: Name for second model
    """
    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    # Get statistics
    stats1 = {
        'params': count_parameters(model1),
        'trainable': count_parameters(model1, trainable_only=True),
        'size_mb': get_model_size(model1, unit='MB'),
    }

    stats2 = {
        'params': count_parameters(model2),
        'trainable': count_parameters(model2, trainable_only=True),
        'size_mb': get_model_size(model2, unit='MB'),
    }

    # Print comparison
    print(f"\n{'':<30} {name1:>20} {name2:>20} {'Difference':>15}")
    print("-" * 90)

    print(f"{'Total Parameters':<30} {stats1['params']:>20,} {stats2['params']:>20,} {stats2['params']-stats1['params']:>15,}")
    print(f"{'Trainable Parameters':<30} {stats1['trainable']:>20,} {stats2['trainable']:>20,} {stats2['trainable']-stats1['trainable']:>15,}")
    print(f"{'Model Size (MB)':<30} {stats1['size_mb']:>20.2f} {stats2['size_mb']:>20.2f} {stats2['size_mb']-stats1['size_mb']:>15.2f}")

    # Percentage differences
    param_diff_pct = ((stats2['params'] - stats1['params']) / stats1['params']) * 100
    size_diff_pct = ((stats2['size_mb'] - stats1['size_mb']) / stats1['size_mb']) * 100

    print("\nPercentage Differences:")
    print(f"  Parameters: {param_diff_pct:+.2f}%")
    print(f"  Size: {size_diff_pct:+.2f}%")

    print("=" * 80)


def estimate_inference_memory(model: nn.Module, input_shape: Tuple[int, ...], batch_size: int = 1) -> Dict[str, float]:
    """
    Estimate memory usage during inference.

    Args:
        model: PyTorch model
        input_shape: Shape of input tensor (without batch dimension)
        batch_size: Batch size

    Returns:
        Dictionary with memory estimates in MB
    """
    # Model parameters
    param_memory = get_model_size(model, unit='MB')

    # Estimate activation memory (rough approximation)
    # This is a simplified estimate
    input_size = batch_size * torch.prod(torch.tensor(input_shape)).item()
    input_memory = input_size * 4 / (1024**2)  # Assume float32

    # Approximate activation memory (typically 2-4x input size)
    activation_memory = input_memory * 3

    return {
        'parameters_mb': param_memory,
        'input_mb': input_memory,
        'activations_mb': activation_memory,
        'total_mb': param_memory + input_memory + activation_memory,
    }
