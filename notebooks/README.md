# Interactive Jupyter Notebooks

This directory contains interactive tutorials for learning the LLM Hook Analysis Framework hands-on.

## Notebooks Overview

### 1. Getting Started (`01_getting_started.ipynb`)
**Level**: Beginner
**Duration**: 15-20 minutes
**Topics**:
- Setting up hooks on PyTorch models
- Capturing forward pass activations
- Basic analysis and visualization
- Understanding hook results
- Proper cleanup with context managers

**What you'll build**: A simple activation analyzer for feedforward networks

---

### 2. Attention Analysis (`02_attention_analysis.ipynb`)
**Level**: Intermediate
**Duration**: 25-30 minutes
**Topics**:
- Monitoring multi-head attention patterns
- Computing attention entropy and sparsity
- Visualizing attention heatmaps
- KV cache analysis for autoregressive models
- Identifying attention bottlenecks

**What you'll build**: An attention pattern analyzer for transformers

---

### 3. Gradient Debugging (`03_gradient_debugging.ipynb`)
**Level**: Intermediate
**Duration**: 30-35 minutes
**Topics**:
- Detecting vanishing/exploding gradients
- Tracking gradient flow through layers
- Monitoring gradient norms during training
- Comparing different architectures
- Solutions for common gradient issues

**What you'll build**: A gradient health monitor for deep networks

---

### 4. Model Pruning (`04_model_pruning.ipynb`)
**Level**: Advanced
**Duration**: 35-40 minutes
**Topics**:
- Understanding Fisher information
- Collecting Fisher scores during training
- Generating pruning masks (weight/neuron/channel level)
- Applying pruning and fine-tuning
- Comparing pruned vs original models

**What you'll build**: An intelligent model compression pipeline

---

## Getting Started

### Prerequisites

```bash
# Install the framework
cd /path/to/Hook_Technique
pip install -e .

# Install Jupyter
pip install jupyter notebook

# Optional: Install JupyterLab for better experience
pip install jupyterlab
```

### Running the Notebooks

**Option 1: Jupyter Notebook**
```bash
cd notebooks
jupyter notebook
```

**Option 2: JupyterLab**
```bash
cd notebooks
jupyter lab
```

**Option 3: VS Code**
- Open the notebook file in VS Code
- Install the Jupyter extension
- Click "Run All" or run cells individually

### Recommended Learning Path

**For Beginners**:
1. Start with `01_getting_started.ipynb`
2. Try the basic examples in `../examples/`
3. Move to `02_attention_analysis.ipynb` or `03_gradient_debugging.ipynb`

**For Researchers**:
1. Review `01_getting_started.ipynb` quickly
2. Deep dive into `02_attention_analysis.ipynb`
3. Explore `04_model_pruning.ipynb`
4. Check advanced examples in `../examples/`

**For ML Engineers**:
1. Skim `01_getting_started.ipynb`
2. Focus on `03_gradient_debugging.ipynb`
3. Study `04_model_pruning.ipynb`
4. Review production best practices in `../docs/best_practices.md`

---

## Tips for Using These Notebooks

### 1. Interactive Exploration
- **Experiment**: Modify parameters and see what happens
- **Visualize**: All notebooks include visualization examples
- **Compare**: Try different configurations side-by-side

### 2. Adapt to Your Models
- Replace dummy models with your own architectures
- Use real datasets instead of random data
- Adjust hyperparameters for your use case

### 3. Performance Considerations
- Start with small models and short runs
- Clear results periodically to save memory
- Use `max_samples` parameter to limit data collection

### 4. Troubleshooting
- If a cell fails, check you ran all previous cells
- Restart kernel if you encounter memory issues
- See `../docs/troubleshooting.md` for common issues

---

## Example Workflows

### Workflow 1: Debug Training Issues
```python
# 1. Run 03_gradient_debugging.ipynb
# 2. Identify problematic layers
# 3. Try solutions (BatchNorm, gradient clipping, etc.)
# 4. Compare before/after
```

### Workflow 2: Optimize Transformer Inference
```python
# 1. Run 02_attention_analysis.ipynb
# 2. Identify sparse attention heads
# 3. Analyze KV cache usage
# 4. Apply optimizations based on findings
```

### Workflow 3: Compress Model for Deployment
```python
# 1. Run 04_model_pruning.ipynb
# 2. Collect Fisher information
# 3. Generate structured pruning masks
# 4. Fine-tune and evaluate
```

---

## Notebook Features

All notebooks include:
- ✅ Clear learning objectives
- ✅ Step-by-step explanations
- ✅ Runnable code examples
- ✅ Visualizations
- ✅ Best practices
- ✅ Common pitfalls to avoid
- ✅ Next steps and exercises

---

## Additional Resources

### Documentation
- **API Reference**: `../docs/api.md`
- **Best Practices**: `../docs/best_practices.md`
- **Architecture**: `../docs/architecture.md`
- **FAQ**: `../docs/faq.md`

### Examples
- **Basic Examples**: `../examples/basic_*.py`
- **Use Cases**: `../examples/use_case_*.py`
- **Tutorials**: `../tutorials/`

### Getting Help
- **Troubleshooting Guide**: `../docs/troubleshooting.md`
- **GitHub Issues**: Report bugs or request features
- **Discussions**: Ask questions and share use cases

---

## Contributing

Found an issue or have an improvement?
- Report notebook bugs via GitHub issues
- Suggest new topics for future notebooks
- Share your own notebook examples

See `../CONTRIBUTING.md` for guidelines.

---

## Quick Reference

### Common Imports
```python
import torch
import torch.nn as nn
from llm_hooks.core import HookManager
from llm_hooks.pytorch import ForwardHook, BackwardHook
from llm_hooks.attention import AttentionMonitor
from llm_hooks.fisher import FisherHook
from llm_hooks.gradients import GradientTracker
```

### Basic Usage Pattern
```python
# 1. Create manager
with HookManager() as manager:
    # 2. Register hooks
    manager.register(ForwardHook(compute_stats=True))

    # 3. Apply to model
    manager.apply_to_model(model)

    # 4. Run inference/training
    output = model(input_data)

    # 5. Get results
    results = manager.get_results()

    # 6. Analyze
    for result in results.results:
        print(result.data)
# 7. Hooks auto-removed on exit
```

---

## Keyboard Shortcuts (Jupyter)

- `Shift + Enter`: Run cell and move to next
- `Ctrl + Enter`: Run cell and stay
- `A`: Insert cell above
- `B`: Insert cell below
- `D + D`: Delete cell
- `M`: Convert to markdown
- `Y`: Convert to code
- `Z`: Undo cell deletion

---

Happy learning! 🚀
