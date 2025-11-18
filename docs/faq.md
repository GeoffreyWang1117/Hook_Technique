# Frequently Asked Questions (FAQ)

Common questions about the LLM Hook Analysis Framework.

## General Questions

### What is this framework for?

The LLM Hook Analysis Framework is designed for analyzing, debugging, and optimizing language models and neural networks. It helps you:
- Understand model behavior during inference and training
- Identify performance bottlenecks
- Detect and fix training issues
- Optimize models through pruning
- Analyze attention patterns
- Monitor gradient flow

**Best for**: Model optimization, debugging training, research analysis, production monitoring.

---

### Do I need GPU/CUDA?

**No**, most features work on CPU. However:
- ✅ Works on CPU: All PyTorch hooks, attention monitoring, activation analysis, gradient tracking, Fisher information
- ⚡ Faster on GPU: Everything runs faster
- 🎯 Requires GPU: CUDA profiling (optional feature)

```python
# Framework automatically detects CUDA availability
if torch.cuda.is_available():
    model = model.cuda()
    # CUDA profiling enabled
else:
    # CPU-only features still work
    pass
```

---

### What models are supported?

**Any PyTorch model**, including:
- ✅ Transformers (GPT, BERT, LLaMA, etc.)
- ✅ Vision models (ResNet, ViT, etc.)
- ✅ Custom architectures
- ✅ Hugging Face models
- ✅ timm models

```python
# Works with any nn.Module
from transformers import AutoModel

model = AutoModel.from_pretrained("bert-base-uncased")
manager.apply_to_model(model)  # Works!
```

---

### How much overhead do hooks add?

Typical overhead:
- **Statistics only**: 1-5% slowdown
- **With data capture**: 10-30% slowdown
- **Comprehensive monitoring**: 20-50% slowdown

**Tips to minimize overhead**:
1. Hook only necessary layers
2. Use statistics instead of raw data
3. Limit sample collection
4. Clear results periodically

```python
# Minimal overhead
hook = ForwardHook(
    layer_name="specific_layer",
    compute_stats=True,
    capture_output=False,  # Don't store tensors
)
```

---

### Can I use this in production?

**Yes**, with considerations:

✅ **Suitable for**:
- A/B testing different models
- Monitoring model health
- Detecting anomalies
- Periodic profiling

⚠️ **Not recommended for**:
- Every production inference
- Real-time critical paths
- When milliseconds matter

**Best practice**: Profile in staging, optimize in production.

---

## Technical Questions

### How do hooks work?

PyTorch provides hook APIs that intercept:
1. **Forward hooks**: Called during forward pass, capture activations
2. **Backward hooks**: Called during backprop, capture gradients
3. **Parameter hooks**: Called when gradients are computed

This framework wraps these APIs with:
- Unified interface
- Automatic result collection
- Built-in analysis tools
- Visualization utilities

```python
# Under the hood (simplified)
def forward_hook(module, input, output):
    # Framework captures and analyzes output
    stats = compute_statistics(output)
    store_result(stats)

# You just use high-level API
hook = ForwardHook(layer_name="fc1")
```

---

### What's the difference between hooks?

| Hook Type | Captures | Use Case |
|-----------|----------|----------|
| `ForwardHook` | Activations | Activation analysis, dead neurons |
| `BackwardHook` | Gradients per layer | Layer-wise gradient analysis |
| `GradientTracker` | Gradients per parameter | Parameter-wise gradient tracking |
| `AttentionMonitor` | Attention scores | Attention pattern analysis |
| `KVCacheMonitor` | Cache usage | Memory optimization |
| `ActivationMonitor` | Activation stats | Dead neuron detection |
| `FisherHook` | Fisher information | Model pruning |
| `CUDAProfiler` | Kernel timing | Performance profiling |

**Choose based on your goal**:
- Debugging training → `GradientTracker` + `ActivationMonitor`
- Optimizing inference → `AttentionMonitor` + `CUDAProfiler`
- Model pruning → `FisherHook`
- Research analysis → `AttentionMonitor`

---

### Can I create custom hooks?

**Yes!** Extend `BaseHook`:

```python
from llm_hooks.core import BaseHook

class MyCustomHook(BaseHook):
    def setup(self, model):
        # Register your hooks
        for name, module in model.named_modules():
            if self._should_hook(module):
                handle = module.register_forward_hook(self._my_hook_fn)
                self._hook_handles.append(handle)

    def teardown(self):
        # Cleanup
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def _my_hook_fn(self, module, input, output):
        # Your custom logic
        data = self._analyze(output)
        result = self.create_result('my_hook', data)
        self.add_result(result)

# Use it
manager.register(MyCustomHook())
```

See `docs/custom_hooks.md` for full guide.

---

### How do I hook specific transformer layers?

```python
# Method 1: Specific layer by name
hook = ForwardHook(layer_name="transformer.layer.5")

# Method 2: Range of layers
hook = ForwardHook(layer_pattern="transformer.layer.[0-5].*")

# Method 3: All layers of a type
hook = ForwardHook(layer_pattern="*.attention.*")

# Find layer names first
for name, module in model.named_modules():
    if 'attention' in name.lower():
        print(name)
```

---

### Can I hook pre-trained models?

**Yes**, works with any PyTorch model:

```python
# Hugging Face
from transformers import AutoModel
model = AutoModel.from_pretrained("bert-base-uncased")

# timm
import timm
model = timm.create_model('resnet50', pretrained=True)

# Custom loaded model
model = torch.load('my_model.pth')

# All work with framework
manager.apply_to_model(model)
```

---

### Does this work with model.eval()?

**Yes**, but with differences:

```python
# Inference mode (eval)
model.eval()
manager.apply_to_model(model)

# ✅ Works: ForwardHook, AttentionMonitor, ActivationMonitor
# ❌ No gradients: BackwardHook, GradientTracker (no backprop in eval)

# Training mode
model.train()
# ✅ Everything works
```

**For gradient analysis, use training mode**.

---

## Usage Questions

### How do I analyze just one layer?

```python
# Create manager
manager = HookManager()

# Hook single layer
hook = ForwardHook(layer_name="transformer.layer.0.attention")
manager.register(hook)
manager.apply_to_model(model)

# Run inference
output = model(input)

# Get results for that layer
results = manager.get_results()
for result in results.results:
    if result.layer_name == "transformer.layer.0.attention":
        print(result.data)
```

---

### How often should I clear results?

Depends on use case:

**Short inference sessions** (< 1000 samples):
```python
# Collect all, analyze at end
results = manager.get_results()
analyze(results)
manager.clear_results()
```

**Long training** (many epochs):
```python
for epoch in range(num_epochs):
    # Training...

    if epoch % 10 == 0:
        results = manager.get_results()
        analyze(results)
        manager.clear_results()  # Free memory
```

**Streaming/production**:
```python
# Clear every N samples
sample_count = 0
for data in stream:
    model(data)
    sample_count += 1

    if sample_count % 100 == 0:
        results = manager.get_results()
        analyze(results)
        manager.clear_results()
```

---

### Can I hook multiple models?

**Yes**, use separate managers:

```python
# Model 1
manager1 = HookManager(name="model1")
manager1.register(ForwardHook())
manager1.apply_to_model(model1)

# Model 2
manager2 = HookManager(name="model2")
manager2.register(ForwardHook())
manager2.apply_to_model(model2)

# Use independently
output1 = model1(x)
output2 = model2(x)

results1 = manager1.get_results()
results2 = manager2.get_results()
```

---

### How do I save results for later analysis?

```python
# Option 1: JSON report
from llm_hooks.analysis import ReportGenerator

report_gen = ReportGenerator(results)
report_gen.save_report("analysis.json", format='json')

# Later: load and analyze
import json
with open("analysis.json") as f:
    data = json.load(f)

# Option 2: Pickle results
import pickle

results = manager.get_results()
with open("results.pkl", "wb") as f:
    pickle.dump(results, f)

# Later: load
with open("results.pkl", "rb") as f:
    results = pickle.load(f)
```

---

### Can I use this with DataParallel/DDP?

**Partial support**:

✅ **Works with DataParallel**:
```python
model = nn.DataParallel(model)
manager.apply_to_model(model.module)  # Hook the underlying model
```

⚠️ **DDP limitations**:
```python
# Hooks work but results only from main process
model = DDP(model)
if torch.distributed.get_rank() == 0:
    manager.apply_to_model(model.module)
```

---

## Optimization Questions

### How do I reduce memory usage?

1. **Disable data capture**:
   ```python
   hook = ForwardHook(
       compute_stats=True,
       capture_output=False,  # Don't store tensors
   )
   ```

2. **Limit samples**:
   ```python
   hook = AttentionMonitor(max_samples=100)
   ```

3. **Clear periodically**:
   ```python
   manager.clear_results()
   ```

4. **Hook selectively**:
   ```python
   # Instead of all layers
   hook = ForwardHook(layer_pattern="transformer.layer.0")
   ```

---

### How do I find performance bottlenecks?

```python
from llm_hooks.cuda import CUDAProfiler
from llm_hooks.analysis import BottleneckAnalyzer

# Setup profiling
manager = HookManager()
manager.register(CUDAProfiler())
manager.apply_to_model(model)

# Run inference
output = model(input)

# Analyze
results = manager.get_results()
analyzer = BottleneckAnalyzer(results)
slow_layers = analyzer.identify_slow_layers()

print("Bottlenecks:", slow_layers)
```

---

### What's the best pruning strategy?

Depends on deployment:

**For research** (max compression):
```python
# Weight-level pruning
masks = fisher_hook.get_pruning_mask(
    pruning_ratio=0.5,
    granularity='weight'
)
# Requires sparse matrix support
```

**For production** (hardware-friendly):
```python
# Structured pruning
masks = fisher_hook.get_pruning_mask(
    pruning_ratio=0.3,
    granularity='neuron'  # or 'channel'
)
# Faster on standard hardware
```

**Workflow**:
1. Start with 10-20% pruning
2. Fine-tune for 1-2 epochs
3. Evaluate accuracy
4. Gradually increase if acceptable

---

## Comparison Questions

### vs TensorBoard?

| Feature | LLM Hooks | TensorBoard |
|---------|-----------|-------------|
| Real-time monitoring | ✅ | ✅ |
| Gradient analysis | ✅ Deep | ✅ Basic |
| Attention analysis | ✅ Specialized | ❌ |
| Fisher information | ✅ | ❌ |
| Dead neuron detection | ✅ | ❌ |
| Programmatic access | ✅ Easy | ⚠️ Limited |
| Visualization | ✅ Good | ✅ Excellent |

**Best practice**: Use both!
- TensorBoard for training monitoring
- LLM Hooks for deep analysis

---

### vs PyTorch Profiler?

| Feature | LLM Hooks | PyTorch Profiler |
|---------|-----------|------------------|
| Performance profiling | ✅ | ✅ |
| Model-specific analysis | ✅ | ❌ |
| Attention patterns | ✅ | ❌ |
| Gradient tracking | ✅ | ❌ |
| Easy to use | ✅ | ⚠️ |

**LLM Hooks includes PyTorch Profiler** (CUDAProfiler) plus model-specific features.

---

### vs Captum?

| Feature | LLM Hooks | Captum |
|---------|-----------|--------|
| Interpretability | ⚠️ Some | ✅ Deep |
| Attribution | ❌ | ✅ |
| Training analysis | ✅ | ❌ |
| Performance profiling | ✅ | ❌ |
| Pruning | ✅ | ❌ |

**Different focus**:
- Captum: Model interpretability, feature attribution
- LLM Hooks: Training debugging, performance optimization

---

## Troubleshooting Questions

See [Troubleshooting Guide](troubleshooting.md) for detailed solutions.

**Quick answers**:

Q: *No results collected?*
A: Check hook applied before inference, correct layer names

Q: *High memory usage?*
A: Disable data capture, limit samples, clear results

Q: *Slow execution?*
A: Hook fewer layers, use statistics only

Q: *NaN gradients?*
A: Use GradientTracker to identify source, reduce LR

Q: *Import errors?*
A: Run `pip install -e .` in project directory

---

## Contributing Questions

### How can I contribute?

We welcome contributions!

1. **Bug reports**: Open GitHub issue
2. **Feature requests**: Open GitHub discussion
3. **Code contributions**: Submit pull request
4. **Documentation**: Improve docs, add examples
5. **Use cases**: Share your applications

See `CONTRIBUTING.md` for guidelines.

---

### Can I add new hook types?

**Yes!** We encourage custom hooks:

1. Extend `BaseHook`
2. Implement `setup()` and `teardown()`
3. Add tests
4. Submit PR with documentation

Example use cases:
- Weight distribution analysis
- Neuron correlation tracking
- Custom metric calculation
- Domain-specific analysis

---

### How do I cite this framework?

```bibtex
@software{llm_hooks_2024,
  title = {LLM Hook Analysis Framework},
  author = {Your Team},
  year = {2024},
  url = {https://github.com/GeoffreyWang1117/Hook_Technique}
}
```

---

## Additional Resources

- **Documentation**: `docs/` directory
- **Tutorials**: `tutorials/` directory
- **Examples**: `examples/` directory
- **API Reference**: `docs/api.md`
- **Best Practices**: `docs/best_practices.md`

**Need more help?** Check the [Troubleshooting Guide](troubleshooting.md) or open a GitHub issue.
