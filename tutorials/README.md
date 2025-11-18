# Tutorials

Step-by-step tutorials for learning the LLM Hook Analysis Framework.

## 📚 Tutorial Series

### 🌟 Beginner Level

Perfect for first-time users who want to understand the basics.

#### [1. Hello World - Your First Hook](01_hello_world.py)
**⏱️ 10 minutes** | **🎯 Difficulty: Easy**

Your first introduction to the framework.

**What you'll learn**:
- Creating a HookManager
- Registering a basic ForwardHook
- Running inference with hooks
- Viewing captured results

**Prerequisites**: None

**Run it**:
```bash
python tutorials/01_hello_world.py
```

---

#### [2. Understanding Gradient Flow](02_gradient_flow.py)
**⏱️ 15 minutes** | **🎯 Difficulty: Easy**

Learn how to track gradients during training.

**What you'll learn**:
- Setting up BackwardHook
- Using GradientTracker
- Detecting vanishing/exploding gradients
- Interpreting gradient statistics

**Prerequisites**: Tutorial 1

**Run it**:
```bash
python tutorials/02_gradient_flow.py
```

---

### 🚀 Intermediate Level

For users comfortable with basics, ready to dive deeper.

#### [3. Analyzing Attention Patterns](03_attention_analysis.py)
**⏱️ 20 minutes** | **🎯 Difficulty: Medium**

Master attention mechanism analysis.

**What you'll learn**:
- Using AttentionMonitor
- Computing entropy and sparsity
- Understanding attention head behaviors
- Identifying optimization opportunities

**Prerequisites**: Tutorials 1-2

**Run it**:
```bash
python tutorials/03_attention_analysis.py
```

---

#### [4. Detecting Dead Neurons](04_dead_neurons.py)
**⏱️ 20 minutes** | **🎯 Difficulty: Medium**

Learn to identify and fix activation issues.

**What you'll learn**:
- Monitoring neuron activations
- Detecting dead neurons
- Analyzing activation distributions
- Implementing solutions

**Prerequisites**: Tutorials 1-2

**Run it**:
```bash
python tutorials/04_dead_neurons.py
```

---

### 🎓 Advanced Level

For experienced users ready for complex scenarios.

#### [5. Model Pruning with Fisher Information](05_fisher_pruning.py)
**⏱️ 30 minutes** | **🎯 Difficulty: Hard**

Master intelligent model compression.

**What you'll learn**:
- Estimating Fisher information
- Parameter importance analysis
- Generating pruning masks (weight/neuron/channel)
- Evaluating pruned models

**Prerequisites**: Tutorials 1-4

**Run it**:
```bash
python tutorials/05_fisher_pruning.py
```

---

## 🎯 Learning Paths

Choose a path based on your goals:

### 🏃 Quick Start (1 hour)
Just want to get started quickly?

1. [Tutorial 1: Hello World](01_hello_world.py) - 10 min
2. [Tutorial 2: Gradient Flow](02_gradient_flow.py) - 15 min
3. [Tutorial 3: Attention Analysis](03_attention_analysis.py) - 20 min
4. [Basic Usage Example](../examples/basic_usage.py) - 15 min

**✅ You'll be able to**: Use basic hooks and understand results

---

### 🎓 Complete Course (3 hours)
Want comprehensive understanding?

1. [Tutorial 1: Hello World](01_hello_world.py) - 10 min
2. [Tutorial 2: Gradient Flow](02_gradient_flow.py) - 15 min
3. [Tutorial 3: Attention Analysis](03_attention_analysis.py) - 20 min
4. [Tutorial 4: Dead Neurons](04_dead_neurons.py) - 20 min
5. [Tutorial 5: Fisher Pruning](05_fisher_pruning.py) - 30 min
6. [Use Case: LLM Optimization](../examples/use_case_llm_optimization.py) - 45 min
7. [Use Case: Model Debugging](../examples/use_case_model_debugging.py) - 30 min
8. Read [Best Practices](../docs/best_practices.md) - 15 min

**✅ You'll be able to**: Apply framework to real projects, extend with custom hooks

---

### 🔬 Research Focus (2 hours)
Focused on research and analysis?

1. [Tutorial 3: Attention Analysis](03_attention_analysis.py) - 20 min
2. [Tutorial 5: Fisher Pruning](05_fisher_pruning.py) - 30 min
3. [Use Case: LLM Optimization](../examples/use_case_llm_optimization.py) - 45 min
4. Read [Architecture Documentation](../docs/architecture.md) - 15 min
5. Review [API Reference](../docs/api.md) - 10 min

**✅ You'll be able to**: Conduct research, publish results, extend framework

---

## 📖 Additional Resources

### Documentation
- [Quick Start Guide](../docs/quickstart.md) - Get started in 5 minutes
- [API Reference](../docs/api.md) - Complete API documentation
- [Best Practices](../docs/best_practices.md) - Tips and guidelines
- [Troubleshooting](../docs/troubleshooting.md) - Common issues and solutions
- [FAQ](../docs/faq.md) - Frequently asked questions
- [Architecture](../docs/architecture.md) - Framework design

### Examples
- [Basic Usage](../examples/basic_usage.py) - Simple example
- [Attention Analysis](../examples/attention_analysis.py) - Attention patterns
- [Pruning Example](../examples/pruning_example.py) - Fisher-based pruning
- [LLM Optimization](../examples/use_case_llm_optimization.py) - Complete workflow
- [Model Debugging](../examples/use_case_model_debugging.py) - Debug training issues

---

## 💡 Tips for Learning

### 1. Run the Code
Don't just read - execute each tutorial. The code is designed to be self-contained and educational.

```bash
# Run any tutorial
python tutorials/01_hello_world.py
python tutorials/02_gradient_flow.py
# ... etc
```

### 2. Experiment
Modify the code to experiment:
- Change layer patterns
- Adjust thresholds
- Try different models
- Add your own analysis

### 3. Read the Output
Each tutorial prints detailed explanations. Read carefully to understand:
- What data is captured
- What the statistics mean
- How to interpret results
- What actions to take

### 4. Check the Documentation
When you encounter something new:
- Check [API docs](../docs/api.md) for parameter details
- Read [best practices](../docs/best_practices.md) for guidelines
- Consult [troubleshooting](../docs/troubleshooting.md) for issues

---

## 🎯 What's Next?

After completing the tutorials:

1. **Apply to Your Project**
   - Start with one hook type
   - Analyze your specific model
   - Implement optimizations based on findings

2. **Explore Advanced Features**
   - Create custom hooks
   - Build multi-hook analysis pipelines
   - Develop domain-specific analyzers

3. **Contribute**
   - Share your use cases
   - Contribute new hooks or features
   - Improve documentation

---

## 🤝 Getting Help

- **Questions**: Check [FAQ](../docs/faq.md) first
- **Issues**: See [Troubleshooting Guide](../docs/troubleshooting.md)
- **Bugs**: Report on GitHub Issues
- **Discussions**: Join GitHub Discussions

---

## 📝 Tutorial Checklist

Track your progress:

**Beginner**:
- [ ] Tutorial 1: Hello World
- [ ] Tutorial 2: Gradient Flow

**Intermediate**:
- [ ] Tutorial 3: Attention Analysis
- [ ] Tutorial 4: Dead Neurons

**Advanced**:
- [ ] Tutorial 5: Fisher Pruning

**Bonus**:
- [ ] Use Case: LLM Optimization
- [ ] Use Case: Model Debugging
- [ ] Read all documentation
- [ ] Apply to your own project

---

## 🌟 Success Stories

After completing these tutorials, you'll be able to:

✅ **Understand** your model's internal behavior
✅ **Identify** performance bottlenecks and training issues
✅ **Optimize** model inference and memory usage
✅ **Debug** training problems systematically
✅ **Analyze** attention patterns and model interpretability
✅ **Prune** models intelligently using Fisher information
✅ **Extend** the framework with custom hooks

---

**Ready to start?** Begin with [Tutorial 1: Hello World](01_hello_world.py)!

Happy learning! 🚀
