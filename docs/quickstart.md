# Quick Start Guide

## Installation

```bash
# Clone the repository
git clone https://github.com/GeoffreyWang1117/Hook_Technique.git
cd Hook_Technique

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Basic Usage

### 1. Import the Framework

```python
from llm_hooks import HookManager
from llm_hooks.pytorch import ForwardHook, BackwardHook
from llm_hooks.attention import AttentionMonitor
```

### 2. Create a Hook Manager

```python
manager = HookManager(name="my_analysis")
```

### 3. Register Hooks

```python
# Monitor forward pass
manager.register(ForwardHook(layer_pattern="transformer.*"))

# Monitor backward pass
manager.register(BackwardHook(layer_pattern="transformer.*"))

# Monitor attention patterns
manager.register(AttentionMonitor(record_scores=True))
```

### 4. Apply to Model

```python
import torch.nn as nn

model = YourTransformerModel()
manager.apply_to_model(model)
```

### 5. Run Inference/Training

```python
# Forward pass
output = model(input_ids)

# Backward pass (if training)
loss = criterion(output, target)
loss.backward()
```

### 6. Analyze Results

```python
# Get all results
results = manager.get_results()

# Generate report
from llm_hooks.analysis import ReportGenerator
report_gen = ReportGenerator(results)
print(report_gen.generate_text_report())

# Visualize
manager.visualize(results)
```

### 7. Cleanup

```python
manager.teardown()
```

## Common Use Cases

### Attention Analysis

```python
from llm_hooks.attention import AttentionMonitor

attention_hook = AttentionMonitor(
    record_scores=True,
    compute_entropy=True,
    compute_sparsity=True
)
manager.register(attention_hook)

# After inference
summary = attention_hook.get_attention_summary()
print(f"Average entropy: {summary['entropy']['mean']}")
```

### Dead Neuron Detection

```python
from llm_hooks.activations import ActivationMonitor

activation_hook = ActivationMonitor(
    dead_neuron_threshold=0.01,
    track_distribution=True
)
manager.register(activation_hook)

# After inference
dead_report = activation_hook.get_dead_neuron_report()
for layer, stats in dead_report.items():
    print(f"{layer}: {stats['dead_ratio']*100:.2f}% dead neurons")
```

### Gradient Flow Analysis

```python
from llm_hooks.gradients import GradientTracker

grad_hook = GradientTracker(
    detect_issues=True,
    vanishing_threshold=1e-6,
    exploding_threshold=100.0
)
manager.register(grad_hook)

# After backward pass
summary = grad_hook.get_gradient_flow_summary()
problematic = grad_hook.get_problematic_layers()
print(f"Problematic layers: {problematic}")
```

### Fisher-based Pruning

```python
from llm_hooks.fisher import FisherHook

fisher_hook = FisherHook(accumulate_samples=100)
manager.register(fisher_hook)

# Train for several iterations
for batch in dataloader:
    # ... training code ...
    fisher_hook.on_batch_end()

# Generate pruning masks
masks = fisher_hook.get_pruning_mask(
    pruning_ratio=0.3,
    granularity='neuron'
)
```

## Examples

Check the `examples/` directory for complete working examples:

- `basic_usage.py` - Basic framework usage
- `attention_analysis.py` - Attention pattern analysis
- `pruning_example.py` - Fisher-based pruning

## Next Steps

- Read the [API Documentation](api.md)
- Learn about [Visualization Options](visualization.md)
- Explore [Custom Hook Development](custom_hooks.md)
