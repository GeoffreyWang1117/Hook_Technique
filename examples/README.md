# Examples

Practical examples demonstrating real-world usage of the LLM Hook Analysis Framework.

## 📁 Available Examples

### 🎯 Basic Examples

#### [Basic Usage](basic_usage.py)
**⏱️ 15 minutes** | **🎯 Difficulty: Easy**

A simple, complete example demonstrating core functionality.

**Features**:
- Creating a simple transformer model
- Setting up multiple hook types
- Running forward and backward passes
- Generating reports and visualizations

**When to use**: Learning the framework, quick prototyping

**Run it**:
```bash
python examples/basic_usage.py
```

---

#### [Attention Analysis](attention_analysis.py)
**⏱️ 20 minutes** | **🎯 Difficulty: Easy**

Focused example on analyzing attention patterns.

**Features**:
- Setting up AttentionMonitor
- Computing entropy and sparsity
- Understanding head behaviors
- Visualizing attention patterns

**When to use**: Analyzing transformer attention mechanisms

**Run it**:
```bash
python examples/attention_analysis.py
```

---

#### [Pruning Example](pruning_example.py)
**⏱️ 25 minutes** | **🎯 Difficulty: Medium**

Demonstrates Fisher information-based pruning.

**Features**:
- Accumulating Fisher information
- Analyzing parameter importance
- Generating pruning masks
- Evaluating pruned models

**When to use**: Model compression, reducing model size

**Run it**:
```bash
python examples/pruning_example.py
```

---

### 🚀 Use Case Examples

#### [LLM Inference Optimization](use_case_llm_optimization.py)
**⏱️ 45 minutes** | **🎯 Difficulty: Advanced**

Complete end-to-end workflow for optimizing a language model.

**Features**:
- Comprehensive profiling (attention, KV cache, activations, CUDA)
- Bottleneck identification
- Memory usage analysis
- Optimization recommendations
- Automated report generation

**What you'll learn**:
- Setting up multiple hooks
- Coordinating analysis pipeline
- Interpreting complex results
- Generating actionable insights

**When to use**:
- Optimizing LLM inference speed
- Reducing memory footprint
- Preparing models for deployment

**Output**:
- Text report: `llm_optimization_report.txt`
- JSON report: `llm_optimization_report.json`
- Visualizations: `./llm_optimization_viz/`

**Run it**:
```bash
python examples/use_case_llm_optimization.py
```

**Expected results**:
- Performance bottlenecks identified
- Dead neuron detection
- KV cache optimization opportunities
- Memory usage breakdown
- Specific optimization recommendations

---

#### [Model Debugging](use_case_model_debugging.py)
**⏱️ 30 minutes** | **🎯 Difficulty: Medium**

Systematic approach to debugging training issues.

**Features**:
- Gradient flow monitoring
- Dead neuron detection
- NaN/Inf tracking
- Issue diagnosis
- Solution recommendations

**What you'll learn**:
- Identifying training problems
- Understanding gradient issues
- Detecting activation problems
- Implementing fixes

**When to use**:
- Training isn't converging
- Loss becomes NaN
- Gradients vanishing/exploding
- Model not learning

**Output**:
- Diagnostic report with issues found
- Detailed analysis per problem
- Code examples for solutions
- Prevention checklist

**Run it**:
```bash
python examples/use_case_model_debugging.py
```

**Covers issues**:
- Vanishing gradients
- Exploding gradients
- Dead neurons
- NaN/Inf values
- Learning instability

---

## 🎯 Usage Patterns

### Pattern 1: Quick Analysis

For rapid model inspection:

```python
from llm_hooks import HookManager
from llm_hooks.pytorch import ForwardHook

# Minimal setup
manager = HookManager()
manager.register(ForwardHook(layer_name="critical_layer"))
manager.apply_to_model(model)

# Run inference
output = model(input)

# Quick check
results = manager.get_results()
for r in results.results:
    print(r.data['stats'])

manager.teardown()
```

---

### Pattern 2: Comprehensive Profiling

For deep analysis:

```python
from llm_hooks import HookManager
from llm_hooks.pytorch import ForwardHook, BackwardHook
from llm_hooks.attention import AttentionMonitor
from llm_hooks.activations import ActivationMonitor
from llm_hooks.analysis import BottleneckAnalyzer, ReportGenerator

# Setup all hooks
manager = HookManager()
manager.register(ForwardHook(layer_pattern="*"))
manager.register(BackwardHook(layer_pattern="*"))
manager.register(AttentionMonitor())
manager.register(ActivationMonitor())
manager.apply_to_model(model)

# Run workload
for batch in dataloader:
    output = model(batch)
    loss.backward()

# Comprehensive analysis
results = manager.get_results()
analyzer = BottleneckAnalyzer(results)
summary = analyzer.generate_summary()

report_gen = ReportGenerator(results)
report_gen.save_report("analysis.txt")
manager.visualize(results)

manager.teardown()
```

---

### Pattern 3: Continuous Monitoring

For long-running training:

```python
manager = HookManager()
manager.register(GradientTracker(detect_issues=True))
manager.register(ActivationMonitor())
manager.apply_to_model(model)

for epoch in range(num_epochs):
    for batch in dataloader:
        # Training step
        output = model(batch)
        loss.backward()
        optimizer.step()

    # Periodic check
    if epoch % 10 == 0:
        results = manager.get_results()
        check_for_issues(results)
        manager.clear_results()  # Free memory

manager.teardown()
```

---

## 📊 Expected Outputs

### Console Output

All examples print detailed, educational output:
- Step-by-step progress
- Statistics and metrics
- Analysis and interpretation
- Recommendations

Example:
```
================================================================================
Tutorial 1: Hello World - Your First Hook
================================================================================

Step 1: Creating a simple neural network...
Model created: SimpleNet(...)
Model parameters: 450 total

Step 2: Creating a hook manager...
Hook manager created: HookManager(name=tutorial_1, hooks=0, setup=False)
...
```

---

### Generated Files

Examples may generate:
- **Reports**: `.txt` and `.json` formats
- **Visualizations**: `.png` plots
- **Traces**: Chrome trace files for profiling

Check each example's output section for details.

---

## 🔧 Customization

### Modify Parameters

All examples are designed to be easily customizable:

```python
# Change model size
model = SimpleTransformer(d_model=256, num_layers=4)  # Smaller

# Adjust hook sensitivity
hook = GradientTracker(
    vanishing_threshold=1e-5,  # More lenient
    exploding_threshold=50.0,   # Stricter
)

# Limit sample collection
hook = AttentionMonitor(max_samples=50)  # Fewer samples
```

---

### Use Your Own Model

Replace example models with your own:

```python
# Instead of
model = SimpleTransformer()

# Use your model
from transformers import AutoModel
model = AutoModel.from_pretrained("bert-base-uncased")

# Or your custom model
model = MyCustomModel()

# Framework works the same!
manager.apply_to_model(model)
```

---

## 🎓 Learning Approach

### For Beginners

1. Start with [basic_usage.py](basic_usage.py)
2. Read through the code comments
3. Run it and examine output
4. Modify parameters and re-run
5. Try [attention_analysis.py](attention_analysis.py)

### For Intermediate Users

1. Study [use_case_llm_optimization.py](use_case_llm_optimization.py)
2. Understand the multi-hook coordination
3. Apply pattern to your use case
4. Experiment with different hook combinations

### For Advanced Users

1. Review all examples
2. Extract patterns for your workflow
3. Extend with custom hooks
4. Integrate into your training pipeline

---

## 🐛 Troubleshooting

### Example Won't Run

```bash
# Check dependencies
pip install -r requirements.txt

# Check installation
pip list | grep llm-hooks

# Reinstall if needed
pip install -e .
```

---

### Insufficient Memory

```python
# Reduce batch size
batch_size = 8  # Instead of 32

# Limit samples
hook = AttentionMonitor(max_samples=20)

# Disable data capture
hook = ForwardHook(capture_output=False)
```

---

### Slow Execution

```python
# Hook fewer layers
hook = ForwardHook(layer_name="specific_layer")

# Use statistics only
hook = ForwardHook(compute_stats=True, capture_output=False)

# Reduce number of iterations
num_iterations = 10  # Instead of 100
```

---

## 📚 Related Documentation

- [API Reference](../docs/api.md) - Detailed parameter descriptions
- [Tutorials](../tutorials/README.md) - Step-by-step learning
- [Best Practices](../docs/best_practices.md) - Usage guidelines
- [Troubleshooting](../docs/troubleshooting.md) - Common issues

---

## 🤝 Contributing Examples

Have a cool use case? Share it!

1. Create a new example file
2. Follow the existing format
3. Add detailed comments
4. Include expected output
5. Update this README
6. Submit a pull request

---

## 💡 Ideas for More Examples

Looking for examples to create:

- [ ] Vision transformer analysis
- [ ] Multi-modal model profiling
- [ ] Distributed training monitoring
- [ ] Quantization-aware training
- [ ] Model architecture search
- [ ] Continuous integration testing
- [ ] Production monitoring setup

---

**Ready to explore?** Start with [basic_usage.py](basic_usage.py)!
