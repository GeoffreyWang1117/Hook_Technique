# Custom Hook Development Guide

This guide teaches you how to create your own custom hooks to extend the LLM Hook Analysis Framework.

## Table of Contents

1. [Understanding the Hook System](#understanding-the-hook-system)
2. [Basic Hook Structure](#basic-hook-structure)
3. [Creating Your First Custom Hook](#creating-your-first-custom-hook)
4. [Advanced Hook Patterns](#advanced-hook-patterns)
5. [Best Practices](#best-practices)
6. [Testing Custom Hooks](#testing-custom-hooks)
7. [Example Gallery](#example-gallery)

---

## Understanding the Hook System

### Architecture Overview

The framework uses a three-layer architecture:

```
┌─────────────────────────────────────┐
│       HookManager                   │  ← Coordinates all hooks
│  - Registers hooks                  │
│  - Applies to models                │
│  - Collects results                 │
└──────────┬──────────────────────────┘
           │
           ├─► BaseHook (Abstract)     ← Your custom hook extends this
           │    - setup()
           │    - teardown()
           │    - create_result()
           │
           └─► PyTorch Hook APIs       ← Low-level PyTorch hooks
                - register_forward_hook()
                - register_backward_hook()
                - register_full_backward_hook()
```

### Key Components

1. **BaseHook**: Abstract base class all hooks must extend
2. **HookManager**: Orchestrates hook lifecycle
3. **HookResult**: Standardized result format
4. **HookRegistry**: Optional registration for discoverability

---

## Basic Hook Structure

### Minimal Hook Template

```python
from llm_hooks.core import BaseHook, HookResult
from typing import Any, Optional
import torch.nn as nn

class MyCustomHook(BaseHook):
    """Custom hook template."""

    def __init__(self, my_param: str = "default", **kwargs):
        """
        Initialize your custom hook.

        Args:
            my_param: Custom parameter for your hook
            **kwargs: Pass remaining args to BaseHook
        """
        super().__init__(**kwargs)
        self.my_param = my_param
        self._hook_handles = []  # Store hook handles for cleanup

    def setup(self, model: nn.Module) -> None:
        """
        Called when hook is applied to a model.
        Register your PyTorch hooks here.

        Args:
            model: The model to instrument
        """
        # Example: Hook all linear layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                handle = module.register_forward_hook(
                    self._create_hook_fn(name)
                )
                self._hook_handles.append(handle)

    def teardown(self) -> None:
        """
        Called when hooks are removed.
        Clean up resources here.
        """
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def _create_hook_fn(self, layer_name: str):
        """Create a hook function for a specific layer."""
        def hook_fn(module, input, output):
            if not self.enabled:
                return

            # Your custom analysis logic here
            data = self._analyze(output)

            # Create and store result
            result = self.create_result(
                hook_type='my_custom_hook',
                data=data,
                layer_name=layer_name
            )
            self.add_result(result)

        return hook_fn

    def _analyze(self, tensor):
        """Your custom analysis logic."""
        return {
            'my_metric': tensor.mean().item(),
            # Add more metrics...
        }
```

### Using Your Custom Hook

```python
from llm_hooks.core import HookManager

# Create and use your custom hook
manager = HookManager()
custom_hook = MyCustomHook(my_param="value")
manager.register(custom_hook)
manager.apply_to_model(model)

# Run inference
output = model(input_data)

# Get results
results = manager.get_results()
for result in results.results:
    if result.hook_type == 'my_custom_hook':
        print(result.data['my_metric'])
```

---

## Creating Your First Custom Hook

### Example: Weight Distribution Monitor

Let's create a hook that monitors weight distributions.

```python
from llm_hooks.core import BaseHook, register_hook
import torch
import torch.nn as nn
from typing import Dict, Any

@register_hook("weight_distribution")
class WeightDistributionHook(BaseHook):
    """Monitor weight distribution statistics."""

    def __init__(
        self,
        track_mean: bool = True,
        track_std: bool = True,
        track_sparsity: bool = True,
        update_frequency: int = 1,
        **kwargs
    ):
        """
        Initialize weight distribution monitor.

        Args:
            track_mean: Compute mean of weights
            track_std: Compute standard deviation
            track_sparsity: Compute sparsity (% of near-zero weights)
            update_frequency: How often to update (every N forward passes)
        """
        super().__init__(**kwargs)
        self.track_mean = track_mean
        self.track_std = track_std
        self.track_sparsity = track_sparsity
        self.update_frequency = update_frequency
        self._forward_count = 0
        self._hook_handles = []

    def setup(self, model: nn.Module) -> None:
        """Register hooks on all layers with weights."""
        for name, module in model.named_modules():
            if self._has_weights(module):
                # Register forward hook (we'll check weights on forward pass)
                handle = module.register_forward_hook(
                    self._create_hook_fn(name, module)
                )
                self._hook_handles.append(handle)

    def teardown(self) -> None:
        """Remove all hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._forward_count = 0

    def _has_weights(self, module: nn.Module) -> bool:
        """Check if module has weight parameters."""
        return hasattr(module, 'weight') and module.weight is not None

    def _create_hook_fn(self, layer_name: str, module: nn.Module):
        """Create hook function for a layer."""
        def hook_fn(module, input, output):
            if not self.enabled:
                return

            # Update only at specified frequency
            self._forward_count += 1
            if self._forward_count % self.update_frequency != 0:
                return

            # Analyze weights
            weights = module.weight.data
            stats = self._compute_weight_stats(weights)

            # Create result
            result = self.create_result(
                hook_type='weight_distribution',
                data={
                    'layer_name': layer_name,
                    'weight_shape': list(weights.shape),
                    'num_parameters': weights.numel(),
                    **stats
                },
                layer_name=layer_name
            )
            self.add_result(result)

        return hook_fn

    def _compute_weight_stats(self, weights: torch.Tensor) -> Dict[str, Any]:
        """Compute weight distribution statistics."""
        stats = {}

        if self.track_mean:
            stats['mean'] = float(weights.mean())
            stats['abs_mean'] = float(weights.abs().mean())

        if self.track_std:
            stats['std'] = float(weights.std())

        if self.track_sparsity:
            # Count near-zero weights (|w| < 0.01)
            near_zero = (weights.abs() < 0.01).float()
            stats['sparsity'] = float(near_zero.mean())

        # Additional useful stats
        stats['min'] = float(weights.min())
        stats['max'] = float(weights.max())

        return stats
```

### Using the Weight Distribution Hook

```python
# Create hook
weight_hook = WeightDistributionHook(
    track_mean=True,
    track_std=True,
    track_sparsity=True,
    update_frequency=10  # Check every 10 forward passes
)

# Use with manager
with HookManager() as manager:
    manager.register(weight_hook)
    manager.apply_to_model(model)

    # Run multiple forward passes
    for data in dataset:
        output = model(data)

    # Analyze weight distributions
    results = manager.get_results()
    for result in results.results:
        if result.hook_type == 'weight_distribution':
            print(f"\nLayer: {result.layer_name}")
            print(f"  Mean: {result.data['mean']:.4f}")
            print(f"  Std: {result.data['std']:.4f}")
            print(f"  Sparsity: {result.data['sparsity']:.2%}")
```

---

## Advanced Hook Patterns

### Pattern 1: Stateful Hooks

Track state across multiple forward passes.

```python
class RunningStatsHook(BaseHook):
    """Track running statistics across batches."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.running_mean = {}  # Track running mean per layer
        self.count = 0

    def _update_running_stats(self, layer_name: str, value: float):
        """Update running statistics."""
        if layer_name not in self.running_mean:
            self.running_mean[layer_name] = value
        else:
            # Exponential moving average
            alpha = 0.1
            self.running_mean[layer_name] = (
                alpha * value + (1 - alpha) * self.running_mean[layer_name]
            )

    def get_running_stats(self) -> Dict[str, float]:
        """Get current running statistics."""
        return self.running_mean.copy()
```

### Pattern 2: Conditional Hooks

Only activate under certain conditions.

```python
class ConditionalHook(BaseHook):
    """Hook that only activates when condition is met."""

    def __init__(self, condition_fn, **kwargs):
        """
        Args:
            condition_fn: Function that returns True to activate hook
        """
        super().__init__(**kwargs)
        self.condition_fn = condition_fn

    def _create_hook_fn(self, layer_name: str):
        def hook_fn(module, input, output):
            if not self.enabled:
                return

            # Check condition
            if not self.condition_fn(module, input, output):
                return

            # Only proceed if condition met
            data = self._analyze(output)
            result = self.create_result('conditional', data, layer_name)
            self.add_result(result)

        return hook_fn

# Usage
def is_anomalous(module, input, output):
    """Check if output is anomalous."""
    return output.abs().max() > 100  # Large activations

hook = ConditionalHook(condition_fn=is_anomalous)
```

### Pattern 3: Multi-Stage Hooks

Capture data at multiple points.

```python
class MultiStageHook(BaseHook):
    """Hook that captures both forward and backward."""

    def setup(self, model: nn.Module):
        """Register both forward and backward hooks."""
        for name, module in model.named_modules():
            if self._should_hook(module):
                # Forward hook
                fwd_handle = module.register_forward_hook(
                    self._forward_hook(name)
                )
                # Backward hook
                bwd_handle = module.register_full_backward_hook(
                    self._backward_hook(name)
                )
                self._hook_handles.extend([fwd_handle, bwd_handle])

    def _forward_hook(self, layer_name: str):
        def hook_fn(module, input, output):
            # Store output for later comparison with gradients
            self._cache[layer_name + '_output'] = output.detach()
            # ... analysis logic
        return hook_fn

    def _backward_hook(self, layer_name: str):
        def hook_fn(module, grad_input, grad_output):
            # Can access stored outputs from forward pass
            output = self._cache.get(layer_name + '_output')
            # ... combined analysis
        return hook_fn
```

### Pattern 4: Parameterized Layer Selection

Flexible layer selection with patterns.

```python
import re

class SelectiveHook(BaseHook):
    """Hook with flexible layer selection."""

    def __init__(
        self,
        layer_pattern: Optional[str] = None,
        layer_types: Optional[list] = None,
        **kwargs
    ):
        """
        Args:
            layer_pattern: Regex pattern to match layer names
            layer_types: List of module types to hook
        """
        super().__init__(**kwargs)
        self.layer_pattern = layer_pattern
        self.layer_types = layer_types or []

    def _should_hook(self, name: str, module: nn.Module) -> bool:
        """Determine if layer should be hooked."""
        # Check type
        if self.layer_types:
            if not any(isinstance(module, t) for t in self.layer_types):
                return False

        # Check name pattern
        if self.layer_pattern:
            if not re.match(self.layer_pattern, name):
                return False

        return True

# Usage examples
hook1 = SelectiveHook(layer_pattern=r"transformer\.layer\.[0-5]\..*")
hook2 = SelectiveHook(layer_types=[nn.Linear, nn.Conv2d])
hook3 = SelectiveHook(
    layer_pattern=r".*attention.*",
    layer_types=[nn.MultiheadAttention]
)
```

---

## Best Practices

### 1. Resource Management

Always clean up resources in `teardown()`:

```python
def teardown(self):
    """Clean up all resources."""
    # Remove hooks
    for handle in self._hook_handles:
        handle.remove()
    self._hook_handles.clear()

    # Clear caches
    if hasattr(self, '_cache'):
        self._cache.clear()

    # Reset counters
    self._count = 0
```

### 2. Memory Efficiency

Avoid storing large tensors unnecessarily:

```python
# ❌ Bad: Stores full tensor
def bad_hook(module, input, output):
    self.outputs.append(output)  # Memory leak!

# ✅ Good: Stores statistics only
def good_hook(module, input, output):
    stats = {
        'mean': output.mean().item(),
        'std': output.std().item(),
    }
    self.stats.append(stats)  # Much smaller

# ✅ Better: Detach if you must store tensors
def better_hook(module, input, output):
    self.outputs.append(output.detach().cpu())  # Move to CPU, no gradient
```

### 3. Handle Edge Cases

```python
def _analyze(self, tensor):
    """Robust analysis with error handling."""
    try:
        # Handle None
        if tensor is None:
            return {'error': 'None tensor'}

        # Handle tuples/lists (e.g., LSTM outputs)
        if isinstance(tensor, (tuple, list)):
            tensor = tensor[0]

        # Handle different dtypes
        if tensor.dtype not in [torch.float32, torch.float16]:
            tensor = tensor.float()

        # Compute statistics
        stats = {
            'mean': float(tensor.mean()),
            'std': float(tensor.std()),
        }

        return stats

    except Exception as e:
        return {'error': str(e)}
```

### 4. Performance Optimization

```python
class OptimizedHook(BaseHook):
    """Example of performance optimizations."""

    def __init__(self, sample_rate: float = 1.0, **kwargs):
        """
        Args:
            sample_rate: Fraction of forward passes to analyze (0-1)
        """
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self._count = 0

    def _create_hook_fn(self, layer_name: str):
        def hook_fn(module, input, output):
            # Early exit if disabled
            if not self.enabled:
                return

            # Sampling to reduce overhead
            self._count += 1
            if (self._count % int(1/self.sample_rate)) != 0:
                return

            # Use no_grad for analysis (faster)
            with torch.no_grad():
                stats = self._fast_stats(output)

            result = self.create_result('optimized', stats, layer_name)
            self.add_result(result)

        return hook_fn

    def _fast_stats(self, tensor):
        """Compute statistics efficiently."""
        # Flatten for faster computation
        flat = tensor.flatten()

        # Use torch functions (faster than Python)
        return {
            'mean': flat.mean().item(),
            'std': flat.std().item(),
            'min': flat.min().item(),
            'max': flat.max().item(),
        }
```

### 5. Documentation

Document your hooks thoroughly:

```python
class WellDocumentedHook(BaseHook):
    """
    One-line summary of what this hook does.

    Longer description explaining:
    - Purpose and use case
    - What data it captures
    - Performance impact
    - Example usage

    Parameters:
        param1 (type): Description
        param2 (type): Description

    Example:
        ```python
        hook = WellDocumentedHook(param1="value")
        with HookManager() as manager:
            manager.register(hook)
            manager.apply_to_model(model)
            output = model(input)
            results = manager.get_results()
        ```

    Note:
        Important information about usage or limitations.

    Warning:
        Potential issues or gotchas users should know.
    """
    pass
```

---

## Testing Custom Hooks

### Basic Test Structure

```python
import pytest
import torch
import torch.nn as nn
from llm_hooks.core import HookManager

class TestMyCustomHook:
    """Test suite for MyCustomHook."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )

    @pytest.fixture
    def hook(self):
        """Create hook instance."""
        return MyCustomHook(my_param="test")

    def test_hook_setup(self, hook, simple_model):
        """Test hook setup."""
        manager = HookManager()
        manager.register(hook)
        manager.apply_to_model(simple_model)

        # Check hooks were registered
        assert len(hook._hook_handles) > 0

    def test_hook_captures_data(self, hook, simple_model):
        """Test hook captures data during forward pass."""
        with HookManager() as manager:
            manager.register(hook)
            manager.apply_to_model(simple_model)

            # Run inference
            input_data = torch.randn(2, 10)
            output = simple_model(input_data)

            # Check results
            results = manager.get_results()
            assert len(results.results) > 0

            # Check result format
            for result in results.results:
                assert result.hook_type == 'my_custom_hook'
                assert 'my_metric' in result.data

    def test_hook_cleanup(self, hook, simple_model):
        """Test hook cleanup."""
        manager = HookManager()
        manager.register(hook)
        manager.apply_to_model(simple_model)

        # Remove hooks
        manager.remove_all_hooks()

        # Check cleanup
        assert len(hook._hook_handles) == 0

    def test_hook_disabled(self, hook, simple_model):
        """Test hook can be disabled."""
        with HookManager() as manager:
            hook.enabled = False  # Disable
            manager.register(hook)
            manager.apply_to_model(simple_model)

            output = simple_model(torch.randn(2, 10))
            results = manager.get_results()

            # Should have no results when disabled
            assert len(results.results) == 0
```

### Run Tests

```bash
# Run your hook tests
pytest tests/test_my_custom_hook.py -v

# With coverage
pytest tests/test_my_custom_hook.py --cov=llm_hooks.custom
```

---

## Example Gallery

### Example 1: Neuron Correlation Hook

Track correlations between neurons:

```python
class NeuronCorrelationHook(BaseHook):
    """Track neuron activation correlations."""

    def __init__(self, max_samples: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.max_samples = max_samples
        self.activations = {}  # Store activations per layer

    def _create_hook_fn(self, layer_name: str):
        def hook_fn(module, input, output):
            if not self.enabled:
                return

            # Store activations (batch_size, num_neurons)
            act = output.detach().cpu()
            if len(act.shape) > 2:
                act = act.flatten(1)  # Flatten spatial dimensions

            if layer_name not in self.activations:
                self.activations[layer_name] = []

            self.activations[layer_name].append(act)

            # Limit samples
            if len(self.activations[layer_name]) > self.max_samples:
                self.activations[layer_name].pop(0)

        return hook_fn

    def compute_correlations(self):
        """Compute correlation matrices for each layer."""
        correlations = {}
        for layer_name, acts in self.activations.items():
            # Concatenate all samples
            all_acts = torch.cat(acts, dim=0)  # (total_samples, num_neurons)

            # Compute correlation matrix
            corr = torch.corrcoef(all_acts.T)  # (num_neurons, num_neurons)
            correlations[layer_name] = corr

        return correlations
```

### Example 2: Layer Timing Hook

Measure layer-wise execution time:

```python
import time

class LayerTimingHook(BaseHook):
    """Measure execution time for each layer."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.timings = {}

    def _create_hook_fn(self, layer_name: str):
        def hook_fn(module, input, output):
            if not self.enabled:
                return

            # Time is measured using forward pre-hook and regular hook
            # This is a simplified version
            start_time = time.perf_counter()
            # (actual computation happens between hooks)
            end_time = time.perf_counter()

            elapsed_ms = (end_time - start_time) * 1000

            if layer_name not in self.timings:
                self.timings[layer_name] = []
            self.timings[layer_name].append(elapsed_ms)

        return hook_fn

    def get_average_timings(self):
        """Get average timing for each layer."""
        return {
            name: sum(times) / len(times)
            for name, times in self.timings.items()
        }
```

### Example 3: Activation Recording Hook

Record activations for later visualization:

```python
class ActivationRecorder(BaseHook):
    """Record activations for visualization."""

    def __init__(
        self,
        layer_names: list,
        max_samples: int = 10,
        store_on_cpu: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.target_layers = set(layer_names)
        self.max_samples = max_samples
        self.store_on_cpu = store_on_cpu
        self.recorded = {name: [] for name in layer_names}

    def setup(self, model: nn.Module):
        """Hook only specified layers."""
        for name, module in model.named_modules():
            if name in self.target_layers:
                handle = module.register_forward_hook(
                    self._create_hook_fn(name)
                )
                self._hook_handles.append(handle)

    def _create_hook_fn(self, layer_name: str):
        def hook_fn(module, input, output):
            if not self.enabled:
                return

            if len(self.recorded[layer_name]) >= self.max_samples:
                return  # Already have enough samples

            # Store activation
            act = output.detach()
            if self.store_on_cpu:
                act = act.cpu()

            self.recorded[layer_name].append(act)

        return hook_fn

    def get_recordings(self):
        """Get recorded activations."""
        return self.recorded
```

---

## Publishing Your Hook

### 1. Register with Framework

```python
from llm_hooks.core import register_hook

@register_hook("my_custom_hook")
class MyCustomHook(BaseHook):
    pass

# Now discoverable via registry
from llm_hooks.core import HookRegistry
available_hooks = HookRegistry.list_hooks()
print(available_hooks)  # Includes 'my_custom_hook'
```

### 2. Add to Package

If contributing to the framework:

```python
# llm_hooks/custom/my_hook.py
class MyCustomHook(BaseHook):
    pass

# llm_hooks/custom/__init__.py
from .my_hook import MyCustomHook

__all__ = ['MyCustomHook']
```

### 3. Documentation

Add to `docs/api.md`:

```markdown
### MyCustomHook

Description of what it does.

**Parameters:**
- `param1` (type): Description

**Example:**
```python
hook = MyCustomHook(param1="value")
```
```

---

## Getting Help

- **Examples**: See `examples/` directory
- **Source Code**: Read existing hooks in `llm_hooks/`
- **Issues**: Open GitHub issue for questions
- **Discussions**: Share your custom hooks

Happy hook development! 🎣
