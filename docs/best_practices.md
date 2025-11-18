# Best Practices

Guidelines for effectively using the LLM Hook Analysis Framework.

## General Principles

### 1. Start Small, Scale Up

**Do**:
```python
# Start with specific layers
manager.register(ForwardHook(layer_name="transformer.layer.0"))
```

**Don't**:
```python
# Hook everything at once
manager.register(ForwardHook(layer_pattern="*"))  # Too broad initially
```

**Why**: Hooking too many layers can:
- Slow down execution significantly
- Generate overwhelming amounts of data
- Make debugging harder

**Recommendation**: Start with 1-3 critical layers, then expand as needed.

---

### 2. Always Clean Up

**Do**:
```python
# Use context manager
with HookManager() as manager:
    manager.register(hook)
    manager.apply_to_model(model)
    # ... your code ...
# Automatic cleanup

# Or explicit cleanup
manager = HookManager()
try:
    # ... your code ...
finally:
    manager.teardown()
```

**Don't**:
```python
manager = HookManager()
manager.apply_to_model(model)
# Forgot to call teardown()!
```

**Why**: Hooks remain attached and can:
- Cause memory leaks
- Slow down future operations
- Interfere with other code

---

### 3. Choose the Right Hook

| Use Case | Recommended Hook | Why |
|----------|-----------------|-----|
| Activation analysis | `ForwardHook` | Captures outputs |
| Gradient debugging | `BackwardHook` or `GradientTracker` | Monitors gradients |
| Attention patterns | `AttentionMonitor` | Specialized for attention |
| Model pruning | `FisherHook` | Estimates importance |
| Performance profiling | `CUDAProfiler` | Measures execution time |
| KV cache optimization | `KVCacheMonitor` | Tracks cache usage |
| Dead neuron detection | `ActivationMonitor` | Analyzes activations |

---

## Hook Configuration

### 1. Layer Pattern Matching

**Specific Layer**:
```python
# Hook one layer
hook = ForwardHook(layer_name="transformer.layer.0.attention")
```

**Pattern Matching**:
```python
# Hook all attention layers
hook = ForwardHook(layer_pattern="*.attention.*")

# Hook all layers in specific range
hook = ForwardHook(layer_pattern="transformer.layer.[0-2].*")
```

**Find Available Layers**:
```python
# List all layer names
for name, module in model.named_modules():
    print(name)

# Or use utility
from llm_hooks.pytorch.utils import get_layer_names
layers = get_layer_names(model, leaf_only=True)
```

---

### 2. Statistics vs Raw Data

**Enable Statistics** (Recommended):
```python
hook = ForwardHook(
    layer_name="fc1",
    compute_stats=True,      # Get mean, std, min, max
    capture_input=False,     # Don't store full tensors
    capture_output=False,    # Don't store full tensors
)
```

**Raw Data** (Use sparingly):
```python
hook = ForwardHook(
    layer_name="fc1",
    compute_stats=True,
    capture_output=True,     # Store full tensor
)
```

**Trade-off**:
- Statistics: Low memory, fast, usually sufficient
- Raw data: High memory, slow, needed for detailed analysis

---

## Performance Optimization

### 1. Minimize Hook Overhead

**Efficient**:
```python
# Hook only what you need
hook = ForwardHook(
    layer_pattern="transformer.layer.*",
    compute_stats=True,
)
```

**Inefficient**:
```python
# Hook everything with full data capture
hook = ForwardHook(
    layer_pattern="*",
    capture_input=True,
    capture_output=True,
    compute_stats=True,
)
```

**Impact**:
- Each hook adds overhead (~1-5% per layer)
- Full data capture can add 50-200% overhead
- Statistics-only adds minimal overhead

---

### 2. Batch Hook Operations

**Good**:
```python
# Register all hooks before applying
manager.register(forward_hook)
manager.register(backward_hook)
manager.register(attention_hook)
manager.apply_to_model(model)  # One-time setup
```

**Bad**:
```python
# Applying multiple times
manager.register(forward_hook)
manager.apply_to_model(model)
manager.register(backward_hook)
manager.apply_to_model(model)  # Redundant
```

---

### 3. Limit Sample Collection

**For Training**:
```python
# Limit samples to avoid memory bloat
attention_hook = AttentionMonitor(
    max_samples=100,  # Stop after 100 samples
)
```

**For Inference**:
```python
# Collect more samples for statistics
attention_hook = AttentionMonitor(
    max_samples=1000,
)
```

**Clear Periodically**:
```python
for epoch in range(num_epochs):
    # Training...

    if epoch % 10 == 0:
        manager.clear_results()  # Free memory
```

---

## Analysis Best Practices

### 1. Interpret Results Contextually

**Gradient Norms**:
```python
grad_norm = 1e-5

# Context matters!
if model_depth > 50:
    # Deep model - this might be normal
    pass
elif learning_rate > 0.1:
    # High LR - consider vanishing gradients
    print("⚠️ Possible vanishing gradient")
```

**Dead Neurons**:
```python
dead_ratio = 0.3

# Consider the activation function
if activation == "ReLU":
    # 30% is concerning for ReLU
    print("⚠️ High dead neuron ratio")
elif activation == "LeakyReLU":
    # 30% is very concerning for LeakyReLU
    print("🚨 Critical issue")
```

---

### 2. Use Appropriate Thresholds

**Customize for Your Model**:
```python
# Default thresholds
gradient_tracker = GradientTracker(
    vanishing_threshold=1e-6,   # May be too strict
    exploding_threshold=100.0,   # May be too loose
)

# Tuned thresholds
gradient_tracker = GradientTracker(
    vanishing_threshold=1e-5,   # Relaxed
    exploding_threshold=10.0,    # Stricter
)
```

**Guidelines**:
- Vanishing: 1e-7 to 1e-5 (stricter for shallow, looser for deep)
- Exploding: 1.0 to 100.0 (stricter for stable, looser for exploratory)
- Dead neurons: 0.001 to 0.1 (based on activation function)

---

### 3. Visualize Before Concluding

```python
# Don't just read numbers
results = manager.get_results()

# Visualize to understand
manager.visualize(results)

# Look for patterns
# - Are issues concentrated in specific layers?
# - Do problems occur at specific times?
# - Are there correlations?
```

---

## Common Patterns

### 1. Debugging Training

```python
def debug_training(model, train_loader):
    """Template for debugging training issues."""
    manager = HookManager()

    # Setup comprehensive monitoring
    manager.register(GradientTracker(detect_issues=True))
    manager.register(ActivationMonitor(dead_neuron_threshold=0.01))
    manager.register(BackwardHook(layer_pattern="*"))

    manager.apply_to_model(model)

    # Training loop
    for epoch in range(num_epochs):
        for batch in train_loader:
            # Training step...

        # Check for issues every epoch
        grad_summary = gradient_tracker.get_gradient_flow_summary()
        if grad_summary['global']['total_issues'] > 0:
            print(f"⚠️ Issues detected in epoch {epoch}")
            # Take action...

    manager.teardown()
```

---

### 2. Model Optimization

```python
def optimize_model(model, val_data):
    """Template for model optimization."""
    manager = HookManager()

    # Setup profiling hooks
    manager.register(AttentionMonitor(compute_sparsity=True))
    manager.register(KVCacheMonitor(track_memory=True))
    manager.register(ActivationMonitor(dead_neuron_threshold=0.01))

    manager.apply_to_model(model)

    # Run inference
    model.eval()
    with torch.no_grad():
        for data in val_data:
            model(data)

    # Generate optimization report
    results = manager.get_results()
    analyzer = BottleneckAnalyzer(results)
    report = ReportGenerator(results)

    report.save_report("optimization_report.txt")
    manager.visualize(results)

    manager.teardown()
```

---

### 3. Research Analysis

```python
def analyze_attention(model, dataset):
    """Template for attention analysis."""
    manager = HookManager()

    # Focused attention monitoring
    attention_hook = AttentionMonitor(
        record_scores=True,
        compute_entropy=True,
        max_samples=500,
    )

    manager.register(attention_hook)
    manager.apply_to_model(model)

    # Collect data
    model.eval()
    with torch.no_grad():
        for data in dataset:
            model(data)

    # Detailed analysis
    summary = attention_hook.get_attention_summary()

    # Export for paper/visualization
    import json
    with open("attention_analysis.json", "w") as f:
        json.dump(summary, f, indent=2)

    manager.teardown()
    return summary
```

---

## Anti-Patterns

### ❌ Don't: Hook Everything

```python
# Bad: Too broad, too slow
manager.register(ForwardHook(layer_pattern="*"))
manager.register(BackwardHook(layer_pattern="*"))
manager.register(ActivationMonitor(layer_pattern="*"))
```

### ✅ Do: Hook Selectively

```python
# Good: Target specific layers
manager.register(ForwardHook(layer_pattern="transformer.layer.[0-2].*"))
manager.register(BackwardHook(layer_name="transformer.layer.0"))
```

---

### ❌ Don't: Ignore Memory Usage

```python
# Bad: Unbounded collection
for epoch in range(100):
    for batch in loader:
        # Collecting results...
        pass
# Memory explosion!
```

### ✅ Do: Manage Memory

```python
# Good: Clear periodically
for epoch in range(100):
    for batch in loader:
        # Collecting results...
        pass

    if epoch % 10 == 0:
        # Process and clear
        results = manager.get_results()
        process_results(results)
        manager.clear_results()
```

---

### ❌ Don't: Over-rely on Defaults

```python
# Bad: Using defaults without thought
hook = GradientTracker()  # Default thresholds may not suit your model
```

### ✅ Do: Tune Parameters

```python
# Good: Customize for your use case
hook = GradientTracker(
    vanishing_threshold=1e-5,  # Tuned for deep model
    exploding_threshold=50.0,   # Based on observations
)
```

---

## Checklist

Before running analysis:
- [ ] Identified which layers to monitor
- [ ] Chosen appropriate hooks for use case
- [ ] Configured thresholds appropriately
- [ ] Set sample limits to prevent memory issues
- [ ] Planned when to clear results
- [ ] Have cleanup strategy (teardown/context manager)

During analysis:
- [ ] Monitoring memory usage
- [ ] Checking intermediate results
- [ ] Adjusting parameters as needed
- [ ] Saving important findings

After analysis:
- [ ] Generated visualizations
- [ ] Saved reports
- [ ] Cleaned up hooks
- [ ] Documented insights
- [ ] Planned next steps

---

## Further Reading

- [API Documentation](api.md) - Detailed parameter descriptions
- [Tutorials](tutorials.md) - Step-by-step guides
- [Troubleshooting](troubleshooting.md) - Common issues
- [Architecture](architecture.md) - Framework design
