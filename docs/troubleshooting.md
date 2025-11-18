# Troubleshooting Guide

Common issues and their solutions when using the LLM Hook Analysis Framework.

## Installation Issues

### Issue: Import Error

```python
ImportError: No module named 'llm_hooks'
```

**Solutions**:

1. **Install in development mode**:
   ```bash
   cd Hook_Technique
   pip install -e .
   ```

2. **Check Python path**:
   ```python
   import sys
   print(sys.path)
   # Ensure your project directory is in the path
   ```

3. **Virtual environment**:
   ```bash
   # Make sure you're in the correct venv
   which python
   pip list | grep llm-hooks
   ```

---

### Issue: Dependency Errors

```
ModuleNotFoundError: No module named 'torch'
```

**Solution**:
```bash
# Install dependencies
pip install -r requirements.txt

# For CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Hook Setup Issues

### Issue: Hook Not Capturing Data

```python
manager = HookManager()
manager.register(ForwardHook(layer_name="fc1"))
manager.apply_to_model(model)
# ... inference ...
results = manager.get_results()
print(len(results))  # Output: 0 (empty!)
```

**Causes and Solutions**:

1. **Wrong layer name**:
   ```python
   # Check actual layer names
   for name, module in model.named_modules():
       print(name)

   # Use correct name
   hook = ForwardHook(layer_name="network.fc1")  # Not just "fc1"
   ```

2. **Hook applied after inference**:
   ```python
   # Wrong order
   output = model(x)  # ❌ Inference before hooks
   manager.apply_to_model(model)

   # Correct order
   manager.apply_to_model(model)  # ✅ Hooks first
   output = model(x)
   ```

3. **Hook disabled**:
   ```python
   # Check if enabled
   print(hook.enabled)  # Should be True

   # Enable if needed
   hook.enable()
   ```

---

### Issue: Pattern Not Matching Layers

```python
hook = ForwardHook(layer_pattern="transformer.*")
# No layers matched!
```

**Solution**:
```python
# Test pattern matching
import re

pattern = "transformer.*"
pattern_re = pattern.replace('*', '.*')

for name, module in model.named_modules():
    if re.match(f'^{pattern_re}$', name):
        print(f"Matched: {name}")

# Adjust pattern based on results
hook = ForwardHook(layer_pattern="transformer.layers.*")  # More specific
```

---

## Memory Issues

### Issue: Out of Memory

```
RuntimeError: CUDA out of memory. Tried to allocate X MiB
```

**Solutions**:

1. **Limit sample collection**:
   ```python
   hook = AttentionMonitor(
       max_samples=100,  # Stop after 100 samples
   )
   ```

2. **Disable raw data capture**:
   ```python
   hook = ForwardHook(
       capture_input=False,   # Don't store tensors
       capture_output=False,  # Only compute stats
       compute_stats=True,
   )
   ```

3. **Clear results periodically**:
   ```python
   for epoch in range(num_epochs):
       # Training...

       if epoch % 5 == 0:
           manager.clear_results()  # Free memory
   ```

4. **Use smaller batch size**:
   ```python
   # Reduce batch size when profiling
   batch_size = 8  # Instead of 32
   ```

---

### Issue: Memory Leak

```python
# Memory keeps growing over time
```

**Solution**:
```python
# Always cleanup hooks
manager.teardown()

# Or use context manager
with HookManager() as manager:
    # Your code...
    pass  # Automatic cleanup
```

---

## Performance Issues

### Issue: Slow Execution

```python
# Model runs much slower with hooks
```

**Causes and Solutions**:

1. **Too many hooks**:
   ```python
   # Bad: Hooking everything
   hook = ForwardHook(layer_pattern="*")

   # Good: Selective hooking
   hook = ForwardHook(layer_pattern="transformer.layer.[0-2].*")
   ```

2. **Full data capture**:
   ```python
   # Slow: Storing full tensors
   hook = ForwardHook(capture_output=True)

   # Fast: Statistics only
   hook = ForwardHook(compute_stats=True, capture_output=False)
   ```

3. **Too many samples**:
   ```python
   # Limit samples for faster execution
   hook = AttentionMonitor(max_samples=50)
   ```

**Benchmark**:
```python
import time

# Without hooks
start = time.time()
output = model(x)
baseline = time.time() - start

# With hooks
manager.apply_to_model(model)
start = time.time()
output = model(x)
hooked = time.time() - start

overhead = (hooked - baseline) / baseline * 100
print(f"Hook overhead: {overhead:.1f}%")
```

---

## Gradient Issues

### Issue: No Gradients Captured

```python
backward_hook = BackwardHook(layer_name="fc1")
# ... training ...
results = manager.get_results()
# No gradient data!
```

**Solutions**:

1. **Model in eval mode**:
   ```python
   # Wrong: Model in eval mode
   model.eval()
   loss.backward()  # No gradients computed

   # Correct: Model in train mode
   model.train()
   loss.backward()
   ```

2. **Detached tensors**:
   ```python
   # Wrong: Detached from graph
   output = model(x).detach()
   loss = criterion(output, y)
   loss.backward()  # No gradients flow

   # Correct: Keep gradient flow
   output = model(x)
   loss = criterion(output, y)
   loss.backward()
   ```

3. **Frozen parameters**:
   ```python
   # Check if parameters require gradients
   for name, param in model.named_parameters():
       if not param.requires_grad:
           print(f"Frozen: {name}")
   ```

---

### Issue: NaN Gradients

```python
# Getting NaN in gradient statistics
```

**Diagnosis**:
```python
# Use gradient tracker to identify source
grad_tracker = GradientTracker(detect_issues=True)
manager.register(grad_tracker)

# After training
summary = grad_tracker.get_gradient_flow_summary()
for layer, stats in summary.items():
    if stats.get('has_issues'):
        print(f"Issue in {layer}: {stats['issues']}")
```

**Solutions**:
- Check for division by zero
- Reduce learning rate
- Use gradient clipping
- Check input data for NaN/Inf

---

## Data Issues

### Issue: Incorrect Statistics

```python
# Statistics don't match expected values
```

**Debugging**:
```python
# Verify data manually
hook = ForwardHook(
    layer_name="fc1",
    capture_output=True,  # Capture raw tensor
    compute_stats=True,
)

# After inference
results = manager.get_results()
result = results.results[0]

# Manual verification
if 'output' in result.data:
    stats = result.data['output']['stats']
    print(f"Mean: {stats['mean']}")

    # If raw tensor captured, verify
    # tensor = result.data['output']['tensor']
    # actual_mean = tensor.mean().item()
    # print(f"Actual mean: {actual_mean}")
```

---

### Issue: Mismatched Dimensions

```python
# Error: tensor dimensions don't match
```

**Solution**:
```python
# Check tensor shapes in results
for result in results.results:
    if 'output' in result.data:
        shape = result.data['output']['shape']
        print(f"{result.layer_name}: {shape}")

# Ensure hook handles your tensor formats
```

---

## Visualization Issues

### Issue: Plots Not Generated

```python
manager.visualize(results)
# No plots appear or saved
```

**Solutions**:

1. **Missing dependencies**:
   ```bash
   pip install matplotlib seaborn
   ```

2. **Backend issues**:
   ```python
   import matplotlib
   matplotlib.use('Agg')  # For headless environments
   import matplotlib.pyplot as plt
   ```

3. **No data collected**:
   ```python
   # Check if results exist
   print(f"Results: {len(results)}")

   # Ensure correct hook types for visualization
   attention_results = results.filter_by_type('attention')
   if attention_results:
       manager.visualize(results)
   ```

4. **Specify output directory**:
   ```python
   manager.visualize(results, output_dir="./my_plots")
   ```

---

## Fisher Hook Issues

### Issue: Zero Fisher Information

```python
fisher_hook = FisherHook()
# After training, all Fisher values are zero
```

**Cause**: Forgot to call `on_batch_end()`

**Solution**:
```python
for batch in dataloader:
    # Training step
    loss.backward()
    optimizer.step()

    # Important: Notify Fisher hook
    fisher_hook.on_batch_end()  # Don't forget!
```

---

### Issue: Pruning Doesn't Work

```python
masks = fisher_hook.get_pruning_mask(pruning_ratio=0.5)
# Applying masks but model still slow
```

**Solutions**:

1. **Masks not applied correctly**:
   ```python
   # Apply masks to model parameters
   for name, param in model.named_parameters():
       if name in masks:
           param.data *= masks[name].to(param.device)
   ```

2. **Unstructured pruning**:
   ```python
   # Unstructured pruning needs sparse support
   # Use structured pruning instead
   masks = fisher_hook.get_pruning_mask(
       pruning_ratio=0.5,
       granularity='neuron'  # Structured pruning
   )
   ```

---

## CUDA Issues

### Issue: CUDA Profiler Fails

```python
RuntimeError: CUDA error: device-side assert triggered
```

**Solutions**:

1. **CUDA not available**:
   ```python
   if torch.cuda.is_available():
       cuda_profiler = CUDAProfiler()
   else:
       print("CUDA not available, skipping GPU profiling")
   ```

2. **Model not on GPU**:
   ```python
   model = model.cuda()
   x = x.cuda()
   ```

3. **Out of CUDA memory**:
   ```python
   # Clear cache before profiling
   torch.cuda.empty_cache()

   # Use smaller batch size
   batch_size = 16  # Reduced
   ```

---

## API Issues

### Issue: Attribute Error

```python
AttributeError: 'HookManager' object has no attribute 'some_method'
```

**Solution**:
```python
# Check API documentation
from llm_hooks import HookManager
help(HookManager)

# Ensure you're using the correct method name
# Common mistakes:
manager.get_results()  # ✅ Correct
manager.collect_results()  # ❌ Wrong
```

---

### Issue: Type Error

```python
TypeError: __init__() got an unexpected keyword argument 'foo'
```

**Solution**:
```python
# Check hook parameters in documentation
hook = ForwardHook(
    layer_name="fc1",        # ✅ Valid parameter
    compute_stats=True,      # ✅ Valid parameter
    foo=True,                # ❌ Invalid parameter
)

# See docs/api.md for all valid parameters
```

---

## Debugging Tips

### 1. Enable Verbose Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now hooks will print debug information
```

### 2. Check Hook State

```python
# Inspect hook manager state
print(manager.summary())

# Check individual hooks
for hook in manager.hooks:
    print(f"{hook.name}: enabled={hook.enabled}, results={len(hook.get_results())}")
```

### 3. Test with Simple Model

```python
# If issues persist, test with minimal example
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)

model = SimpleModel()
manager = HookManager()
hook = ForwardHook(layer_name="fc")
manager.register(hook)
manager.apply_to_model(model)

x = torch.randn(2, 10)
y = model(x)

results = manager.get_results()
print(f"Results: {len(results)}")  # Should be 1
```

### 4. Check PyTorch Version

```python
import torch
print(f"PyTorch version: {torch.__version__}")

# Framework requires PyTorch >= 2.0
assert torch.__version__ >= "2.0", "Please upgrade PyTorch"
```

---

## Getting More Help

If you can't resolve your issue:

1. **Check Documentation**:
   - API reference: `docs/api.md`
   - Best practices: `docs/best_practices.md`
   - Tutorials: `docs/tutorials.md`

2. **Review Examples**:
   - Basic usage: `examples/basic_usage.py`
   - Use cases: `examples/use_case_*.py`
   - Tutorials: `tutorials/`

3. **Search Issues**:
   - Check GitHub Issues for similar problems
   - Search discussions and closed issues

4. **Create Minimal Reproducible Example**:
   ```python
   # Provide:
   # 1. Python version
   # 2. PyTorch version
   # 3. Framework version
   # 4. Minimal code that reproduces issue
   # 5. Full error traceback
   ```

5. **Report Bug**:
   - Open issue on GitHub
   - Include system info and error trace
   - Provide reproducible example
