# Performance Optimization Guide

This guide provides best practices for minimizing overhead and maximizing efficiency when using the LLM Hook Analysis Framework.

## Table of Contents

1. [Understanding Hook Overhead](#understanding-hook-overhead)
2. [Minimizing Memory Usage](#minimizing-memory-usage)
3. [Reducing Computational Overhead](#reducing-computational-overhead)
4. [Selective Hooking Strategies](#selective-hooking-strategies)
5. [Production Deployment](#production-deployment)
6. [Benchmarking and Profiling](#benchmarking-and-profiling)
7. [Advanced Optimization Techniques](#advanced-optimization-techniques)

---

## Understanding Hook Overhead

### Sources of Overhead

Hooks add overhead through:

1. **Function call overhead**: Each hook function call
2. **Data transfer**: Moving tensors between devices (GPU ↔ CPU)
3. **Computation**: Statistics calculation, analysis
4. **Memory**: Storing results, intermediate data
5. **Synchronization**: GPU sync points for timing

### Typical Overhead Ranges

| Configuration | Overhead | Use Case |
|---------------|----------|----------|
| Statistics only, selective layers | 1-5% | Production monitoring |
| Statistics only, all layers | 5-15% | Development profiling |
| With data capture, selective | 10-30% | Debugging specific issues |
| With data capture, all layers | 30-60% | Comprehensive analysis |
| Full capture + visualization | 60-100%+ | Research, one-time analysis |

---

## Minimizing Memory Usage

### 1. Disable Tensor Storage

**Problem**: Storing full tensors consumes massive memory.

**Solution**: Use statistics instead of raw data.

```python
# ❌ Bad: Stores full tensors
hook = ForwardHook(
    capture_output=True,  # Stores entire activation tensors
    compute_stats=False
)

# ✅ Good: Only statistics
hook = ForwardHook(
    capture_output=False,  # Don't store tensors
    compute_stats=True      # Only compute stats
)
```

**Memory savings**: ~100x reduction for typical layers

### 2. Limit Sample Collection

**Problem**: Accumulating unlimited samples over time.

**Solution**: Set maximum sample counts.

```python
# Limit samples per hook
attention_hook = AttentionMonitor(
    max_samples=100,  # Only keep last 100 samples
    compute_stats=True
)

# Or use sampling
forward_hook = ForwardHook(
    sample_rate=0.1,  # Only process 10% of forward passes
    compute_stats=True
)
```

### 3. Periodic Result Clearing

**Problem**: Results accumulate indefinitely.

**Solution**: Clear results periodically.

```python
# Clear every N batches
batch_count = 0
for data, target in dataloader:
    output = model(data)

    batch_count += 1
    if batch_count % 100 == 0:
        # Analyze current results
        results = manager.get_results()
        analyze(results)

        # Clear to free memory
        manager.clear_results()
```

### 4. Move Data to CPU

**Problem**: Keeping data on GPU wastes VRAM.

**Solution**: Move to CPU when storing.

```python
class MemoryEfficientHook(BaseHook):
    def _create_hook_fn(self, layer_name: str):
        def hook_fn(module, input, output):
            # Detach and move to CPU immediately
            output_cpu = output.detach().cpu()

            # Now safe to store without GPU memory impact
            stats = self._compute_stats(output_cpu)
            # ...
        return hook_fn
```

### 5. Use torch.no_grad()

**Problem**: Hooks creating unnecessary gradient graphs.

**Solution**: Wrap analysis in `torch.no_grad()`.

```python
def _analyze_activation(self, tensor):
    # Prevent gradient tracking during analysis
    with torch.no_grad():
        stats = {
            'mean': tensor.mean().item(),
            'std': tensor.std().item(),
            # ...
        }
    return stats
```

---

## Reducing Computational Overhead

### 1. Hook Only Necessary Layers

**Problem**: Hooking all layers adds unnecessary overhead.

**Solution**: Target specific layers.

```python
# ❌ Bad: Hook everything
hook = ForwardHook()  # Hooks ALL modules

# ✅ Good: Hook specific layers
hook = ForwardHook(
    layer_pattern=r"transformer\.layer\.(0|5|11)\..*"  # Only layers 0, 5, 11
)

# ✅ Better: Hook by type
hook = ForwardHook(
    layer_types=[nn.Linear, nn.Conv2d]  # Only specific types
)
```

### 2. Reduce Statistic Computation

**Problem**: Computing many statistics is slow.

**Solution**: Only compute what you need.

```python
# ❌ Bad: Compute everything
stats = {
    'mean': tensor.mean().item(),
    'std': tensor.std().item(),
    'min': tensor.min().item(),
    'max': tensor.max().item(),
    'median': tensor.median().item(),  # Expensive!
    'quantiles': torch.quantile(tensor, [0.25, 0.75]).tolist(),  # Expensive!
    # ... many more
}

# ✅ Good: Essential stats only
stats = {
    'mean': tensor.mean().item(),
    'std': tensor.std().item(),
}
```

### 3. Batch Statistics Computation

**Problem**: Computing stats per sample is inefficient.

**Solution**: Compute over batches.

```python
def _efficient_stats(self, tensor):
    """Compute stats efficiently over batch dimension."""
    # Single pass for multiple stats
    mean = tensor.mean()
    std = tensor.std()

    # Use existing intermediate results
    variance = std ** 2

    return {
        'mean': mean.item(),
        'std': std.item(),
        'variance': variance.item(),
    }
```

### 4. Lazy Evaluation

**Problem**: Computing stats immediately on every forward pass.

**Solution**: Defer computation until needed.

```python
class LazyHook(BaseHook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.raw_data = []  # Store lightweight references

    def _create_hook_fn(self, layer_name: str):
        def hook_fn(module, input, output):
            # Just store reference, don't compute yet
            self.raw_data.append({
                'layer': layer_name,
                'shape': output.shape,
                'device': output.device,
            })
        return hook_fn

    def compute_statistics(self):
        """Compute stats only when explicitly requested."""
        # Now compute on all collected data
        # ...
```

### 5. Vectorized Operations

**Problem**: Python loops over tensor elements.

**Solution**: Use vectorized PyTorch operations.

```python
# ❌ Bad: Python loop
sparsity = 0
for value in tensor.flatten():
    if abs(value) < threshold:
        sparsity += 1
sparsity /= tensor.numel()

# ✅ Good: Vectorized
sparsity = (tensor.abs() < threshold).float().mean().item()
```

---

## Selective Hooking Strategies

### Strategy 1: Layer Importance Sampling

Hook important layers more frequently.

```python
# Sample rates by layer importance
sample_rates = {
    'attention': 1.0,      # Always hook attention (important)
    'feedforward': 0.5,    # Hook 50% of FFN layers
    'embedding': 0.1,      # Rarely hook embeddings
}

for name, module in model.named_modules():
    layer_type = get_layer_type(name)
    rate = sample_rates.get(layer_type, 0.0)

    if rate > 0:
        hook = ForwardHook(layer_name=name, sample_rate=rate)
        manager.register(hook)
```

### Strategy 2: Progressive Detail

Start coarse, then drill down.

```python
# Phase 1: Coarse analysis (all layers, minimal stats)
with HookManager() as manager:
    hook = ForwardHook(compute_stats=True, capture_output=False)
    manager.apply_to_model(model)
    run_inference()
    results = manager.get_results()

# Identify interesting layers
problematic_layers = identify_issues(results)

# Phase 2: Detailed analysis (only problematic layers)
with HookManager() as manager:
    for layer in problematic_layers:
        detailed_hook = ForwardHook(
            layer_name=layer,
            compute_stats=True,
            capture_output=True,  # Now capture full data
        )
        manager.register(detailed_hook)

    manager.apply_to_model(model)
    run_inference()
    detailed_results = manager.get_results()
```

### Strategy 3: Conditional Activation

Only activate hooks when conditions are met.

```python
class ConditionalHook(BaseHook):
    def __init__(self, trigger_condition, **kwargs):
        super().__init__(**kwargs)
        self.trigger_condition = trigger_condition
        self.triggered = False

    def _create_hook_fn(self, layer_name: str):
        def hook_fn(module, input, output):
            # Check if we should activate
            if not self.triggered:
                if self.trigger_condition():
                    self.triggered = True
                else:
                    return  # Skip this call

            # Normal hook logic
            # ...
        return hook_fn

# Usage: Only activate after 100 iterations
iteration_count = 0
def should_trigger():
    global iteration_count
    iteration_count += 1
    return iteration_count > 100

hook = ConditionalHook(trigger_condition=should_trigger)
```

### Strategy 4: Adaptive Sampling

Increase sampling when anomalies detected.

```python
class AdaptiveHook(BaseHook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_sample_rate = 0.1
        self.current_sample_rate = 0.1
        self.anomaly_count = 0

    def _create_hook_fn(self, layer_name: str):
        def hook_fn(module, input, output):
            # Sample based on current rate
            if random.random() > self.current_sample_rate:
                return

            # Analyze
            stats = self._compute_stats(output)

            # Detect anomalies
            if self._is_anomalous(stats):
                self.anomaly_count += 1
                # Increase sampling when anomalies found
                self.current_sample_rate = min(1.0, self.current_sample_rate * 2)
            else:
                # Decrease sampling when normal
                self.current_sample_rate = max(
                    self.base_sample_rate,
                    self.current_sample_rate * 0.9
                )

            # ...
        return hook_fn
```

---

## Production Deployment

### Best Practices for Production

1. **Minimal overhead configuration**
2. **Graceful degradation**
3. **Feature flags for easy disable**
4. **Separate monitoring from critical path**
5. **Async result processing**

### Production-Ready Hook Configuration

```python
class ProductionHookConfig:
    """Production-safe hook configuration."""

    @staticmethod
    def create_lightweight_monitor(model: nn.Module) -> HookManager:
        """Create minimal overhead monitoring setup."""
        manager = HookManager(name="prod_monitor")

        # Only hook critical layers
        critical_layers = [
            "transformer.layer.0",   # First layer
            "transformer.layer.11",  # Last layer
            "lm_head",               # Output layer
        ]

        for layer_name in critical_layers:
            hook = ForwardHook(
                layer_name=layer_name,
                compute_stats=True,
                capture_output=False,  # No tensor storage
                sample_rate=0.01,      # 1% sampling
            )
            manager.register(hook)

        manager.apply_to_model(model)
        return manager

    @staticmethod
    def create_anomaly_detector(model: nn.Module) -> HookManager:
        """Create anomaly detection with minimal overhead."""
        manager = HookManager(name="anomaly_detector")

        # Monitor for NaN/Inf only (very cheap)
        class AnomalyHook(BaseHook):
            def _create_hook_fn(self, layer_name: str):
                def hook_fn(module, input, output):
                    # Fast check for anomalies
                    if torch.isnan(output).any() or torch.isinf(output).any():
                        # Log anomaly
                        self.add_result(self.create_result(
                            'anomaly',
                            {'layer': layer_name, 'type': 'nan_or_inf'},
                            layer_name
                        ))
                return hook_fn

        manager.register(AnomalyHook())
        manager.apply_to_model(model)
        return manager
```

### Async Result Processing

Process results in background to avoid blocking inference.

```python
import threading
import queue

class AsyncHookManager:
    """Hook manager with async result processing."""

    def __init__(self):
        self.manager = HookManager()
        self.result_queue = queue.Queue()
        self.processor_thread = threading.Thread(
            target=self._process_results,
            daemon=True
        )
        self.processor_thread.start()

    def _process_results(self):
        """Background thread for result processing."""
        while True:
            results = self.result_queue.get()
            if results is None:  # Shutdown signal
                break

            # Process results (analyze, save, etc.)
            self._analyze_results(results)

    def collect_results(self):
        """Non-blocking result collection."""
        results = self.manager.get_results()
        self.manager.clear_results()

        # Queue for background processing
        self.result_queue.put(results)

    def shutdown(self):
        """Graceful shutdown."""
        self.result_queue.put(None)
        self.processor_thread.join()
```

### Feature Flags

Easy enable/disable via environment variables.

```python
import os

class FeatureFlaggedHooks:
    """Hooks with feature flag control."""

    ENABLE_HOOKS = os.getenv('ENABLE_HOOKS', 'false').lower() == 'true'
    HOOK_SAMPLE_RATE = float(os.getenv('HOOK_SAMPLE_RATE', '0.01'))

    @classmethod
    def setup_monitoring(cls, model):
        """Setup monitoring if enabled."""
        if not cls.ENABLE_HOOKS:
            print("Hooks disabled via ENABLE_HOOKS flag")
            return None

        manager = HookManager()
        hook = ForwardHook(
            compute_stats=True,
            sample_rate=cls.HOOK_SAMPLE_RATE
        )
        manager.register(hook)
        manager.apply_to_model(model)

        print(f"Hooks enabled with {cls.HOOK_SAMPLE_RATE:.1%} sampling")
        return manager

# Usage
manager = FeatureFlaggedHooks.setup_monitoring(model)
if manager:  # Only if hooks enabled
    # ... monitoring logic
    pass
```

---

## Benchmarking and Profiling

### Measure Hook Overhead

```python
import time
import numpy as np

def benchmark_hook_overhead(
    model,
    hook_config,
    input_data,
    num_runs=100
):
    """Measure overhead added by hooks."""

    # Baseline (no hooks)
    model.eval()
    baseline_times = []

    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(input_data)

        # Measure
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(input_data)
            end = time.perf_counter()
            baseline_times.append((end - start) * 1000)

    baseline_mean = np.mean(baseline_times)

    # With hooks
    with HookManager() as manager:
        for hook in hook_config:
            manager.register(hook)
        manager.apply_to_model(model)

        hook_times = []
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = model(input_data)

            # Measure
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = model(input_data)
                end = time.perf_counter()
                hook_times.append((end - start) * 1000)

        hook_mean = np.mean(hook_times)

    overhead_pct = ((hook_mean - baseline_mean) / baseline_mean) * 100

    print(f"Baseline: {baseline_mean:.2f}ms")
    print(f"With hooks: {hook_mean:.2f}ms")
    print(f"Overhead: {overhead_pct:.1f}%")

    return {
        'baseline_ms': baseline_mean,
        'with_hooks_ms': hook_mean,
        'overhead_pct': overhead_pct,
    }
```

### Profile Memory Usage

```python
import tracemalloc

def profile_memory_usage(model, manager, input_data):
    """Profile memory usage with hooks."""

    # Start tracing
    tracemalloc.start()

    # Baseline
    snapshot_before = tracemalloc.take_snapshot()

    # Run with hooks
    with torch.no_grad():
        for _ in range(100):
            _ = model(input_data)

    snapshot_after = tracemalloc.take_snapshot()

    # Get results (triggers memory allocation)
    results = manager.get_results()

    snapshot_final = tracemalloc.take_snapshot()

    # Compare
    stats_inference = snapshot_after.compare_to(snapshot_before, 'lineno')
    stats_results = snapshot_final.compare_to(snapshot_after, 'lineno')

    print("Memory usage during inference:")
    for stat in stats_inference[:5]:
        print(f"  {stat}")

    print("\nMemory usage for results:")
    for stat in stats_results[:5]:
        print(f"  {stat}")

    tracemalloc.stop()
```

---

## Advanced Optimization Techniques

### 1. Kernel Fusion

Fuse multiple operations in hooks.

```python
def _fused_statistics(self, tensor):
    """Compute multiple statistics in one pass."""
    # Single pass for mean, variance, min, max
    mean = tensor.mean()
    var = tensor.var()
    min_val = tensor.min()
    max_val = tensor.max()

    return {
        'mean': mean.item(),
        'std': var.sqrt().item(),
        'min': min_val.item(),
        'max': max_val.item(),
    }
```

### 2. Precompute When Possible

Cache computations that don't change.

```python
class OptimizedHook(BaseHook):
    def setup(self, model):
        # Precompute layer information
        self.layer_info = {}
        for name, module in model.named_modules():
            if self._should_hook(module):
                self.layer_info[name] = {
                    'type': type(module).__name__,
                    'num_params': sum(p.numel() for p in module.parameters()),
                    # ... other static info
                }
                # Now register hooks
                # ...
```

### 3. Custom CUDA Kernels

For critical paths, write custom kernels.

```python
# Example: Custom sparse statistics kernel
try:
    from torch.utils.cpp_extension import load

    custom_stats = load(
        name="custom_stats",
        sources=["custom_stats.cpp", "custom_stats.cu"],
        verbose=True
    )

    def _fast_stats(tensor):
        """Use custom CUDA kernel for stats."""
        return custom_stats.compute_stats(tensor)

except Exception:
    # Fallback to PyTorch
    def _fast_stats(tensor):
        return {
            'mean': tensor.mean().item(),
            'std': tensor.std().item(),
        }
```

### 4. Result Aggregation

Aggregate results to reduce memory.

```python
class AggregatingHook(BaseHook):
    """Aggregate results instead of storing individually."""

    def __init__(self, aggregation_window=100, **kwargs):
        super().__init__(**kwargs)
        self.window = aggregation_window
        self.buffer = defaultdict(list)

    def _create_hook_fn(self, layer_name: str):
        def hook_fn(module, input, output):
            stats = self._compute_stats(output)

            # Buffer results
            self.buffer[layer_name].append(stats)

            # Aggregate when buffer full
            if len(self.buffer[layer_name]) >= self.window:
                aggregated = self._aggregate(self.buffer[layer_name])
                result = self.create_result(
                    'aggregated',
                    aggregated,
                    layer_name
                )
                self.add_result(result)

                # Clear buffer
                self.buffer[layer_name].clear()

        return hook_fn

    def _aggregate(self, stats_list):
        """Aggregate multiple statistics."""
        means = [s['mean'] for s in stats_list]
        return {
            'mean_of_means': np.mean(means),
            'std_of_means': np.std(means),
            # ...
        }
```

---

## Quick Reference: Optimization Checklist

### Before Deployment

- [ ] Measure baseline performance without hooks
- [ ] Test with realistic data and batch sizes
- [ ] Profile memory usage
- [ ] Benchmark different hook configurations
- [ ] Set appropriate sample rates
- [ ] Enable only necessary hooks
- [ ] Add feature flags for easy disable
- [ ] Test graceful degradation

### During Development

- [ ] Use context managers for automatic cleanup
- [ ] Clear results periodically
- [ ] Move data to CPU when storing
- [ ] Use `torch.no_grad()` in hooks
- [ ] Disable hooks when not needed
- [ ] Monitor memory growth over time

### Production Monitoring

- [ ] Minimal overhead configuration (< 5%)
- [ ] Async result processing
- [ ] Anomaly detection only
- [ ] Selective layer sampling
- [ ] Automatic alerts for issues
- [ ] Easy enable/disable mechanism

---

## Summary

**Golden Rules**:
1. Only hook what you need
2. Don't store what you can compute
3. Clear results regularly
4. Use sampling for high-frequency operations
5. Profile before deploying

**Overhead Targets**:
- **Development**: 10-30% acceptable
- **Staging**: < 10% recommended
- **Production**: < 5% critical

For more details, see:
- [Best Practices](best_practices.md)
- [Benchmarking Script](../scripts/benchmark.py)
- [FAQ](faq.md)
