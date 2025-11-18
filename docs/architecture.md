# Architecture Design

Comprehensive guide to the LLM Hook Analysis Framework architecture.

## Overview

The framework follows a modular, pluggable architecture designed for extensibility and ease of use.

```
┌─────────────────────────────────────────────────────────┐
│                    User Application                      │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                     HookManager                          │
│  • Coordinates multiple hooks                            │
│  • Manages hook lifecycle                                │
│  • Collects and aggregates results                       │
└─────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  PyTorch     │   │  Attention   │   │  Activation  │
│  Hooks       │   │  Monitors    │   │  Monitors    │
└──────────────┘   └──────────────┘   └──────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────┐
│                  PyTorch Model                           │
│  • Forward pass                                          │
│  • Backward pass                                         │
│  • Parameter updates                                     │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              Analysis & Visualization                    │
│  • BottleneckAnalyzer                                    │
│  • ReportGenerator                                       │
│  • Visualization tools                                   │
└─────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Base Classes

#### BaseHook

The abstract foundation for all hooks.

```python
class BaseHook(ABC):
    """
    Abstract base class providing:
    - Lifecycle management (setup/teardown)
    - Result collection
    - Enable/disable functionality
    """

    @abstractmethod
    def setup(self, model: Any) -> None:
        """Attach hooks to model."""

    @abstractmethod
    def teardown(self) -> None:
        """Remove hooks and cleanup."""
```

**Design principles**:
- Single Responsibility: Each hook type handles one aspect
- Open/Closed: Easy to extend without modifying core
- Liskov Substitution: All hooks interchangeable through base class

---

#### HookManager

Central coordinator for all hooks.

```python
class HookManager:
    """
    Responsibilities:
    - Register/unregister hooks
    - Apply hooks to models
    - Collect aggregated results
    - Coordinate teardown
    """
```

**Key features**:
- **Hook Registry**: Maintains list of active hooks
- **Lazy Application**: Hooks applied when needed
- **Result Aggregation**: Collects from all hooks
- **Context Manager**: Automatic cleanup

**Lifecycle**:
```
Create → Register Hooks → Apply to Model → Run → Collect Results → Teardown
```

---

### 2. Hook Types

#### PyTorch Hooks (`llm_hooks/pytorch/`)

Fundamental building blocks using PyTorch's hook API.

**ForwardHook**:
```python
# Captures activations during forward pass
module.register_forward_hook(hook_fn)
```

**BackwardHook**:
```python
# Captures gradients during backpropagation
module.register_full_backward_hook(hook_fn)
```

**Design pattern**: Observer pattern
- PyTorch modules are subjects
- Hooks are observers
- Notified on forward/backward events

---

#### Specialized Hooks

Built on PyTorch hooks with domain-specific logic.

**AttentionMonitor** (`llm_hooks/attention/`):
- Identifies attention layers
- Extracts attention scores
- Computes entropy, sparsity
- Analyzes head diversity

**ActivationMonitor** (`llm_hooks/activations/`):
- Tracks activation statistics
- Detects dead neurons
- Monitors sparsity
- Analyzes distributions

**FisherHook** (`llm_hooks/fisher/`):
- Accumulates squared gradients
- Estimates parameter importance
- Generates pruning masks

**GradientTracker** (`llm_hooks/gradients/`):
- Monitors gradient flow
- Detects vanishing/exploding
- Tracks NaN/Inf values

---

### 3. Data Structures

#### HookResult

Container for captured data.

```python
@dataclass
class HookResult:
    hook_name: str          # Which hook captured this
    hook_type: str          # Type of hook (forward, backward, etc.)
    timestamp: float        # When captured
    layer_name: str         # Which layer
    data: Dict[str, Any]    # Captured data
    metadata: Dict[str, Any]  # Additional context
```

**Design pattern**: Value Object
- Immutable after creation
- Self-contained
- Serializable

---

#### HookResultCollection

Aggregates and queries results.

```python
class HookResultCollection:
    """
    Features:
    - Filter by hook, type, layer
    - Sort by timestamp
    - Iterate over results
    """
```

**Design pattern**: Collection + Strategy
- Encapsulates list of results
- Provides query strategies
- Maintains order

---

### 4. Analysis Layer

#### BottleneckAnalyzer

Identifies performance and training issues.

```python
class BottleneckAnalyzer:
    """
    Analyzes:
    - Slow layers (performance)
    - Gradient issues (training)
    - Dead neurons (efficiency)
    - Memory usage (resources)
    """
```

**Algorithm**:
```
1. Aggregate data from all hooks
2. Compute statistics per layer
3. Apply thresholds for issues
4. Rank by severity
5. Generate recommendations
```

---

#### ReportGenerator

Creates human-readable reports.

```python
class ReportGenerator:
    """
    Generates:
    - Text reports (human-readable)
    - JSON reports (programmatic)
    - Structured summaries
    """
```

**Design pattern**: Template Method
- Common report structure
- Customizable sections
- Multiple output formats

---

### 5. Visualization

Converts numerical results to visual insights.

```python
# Module: llm_hooks/visualization/

def visualize_results(results: HookResultCollection):
    """
    Creates:
    - Attention heatmaps
    - Gradient flow charts
    - Activation distributions
    - Performance timelines
    """
```

**Technologies**:
- Matplotlib: Static plots
- Seaborn: Statistical visualizations
- NumPy: Data processing

---

## Data Flow

### Forward Pass Monitoring

```
Input → Model Forward Pass
         ↓
    [ForwardHook]
         ↓
    Capture Output
         ↓
    Compute Statistics
         ↓
    Create HookResult
         ↓
    Store in Collection
         ↓
    Continue Forward Pass
         ↓
       Output
```

---

### Backward Pass Monitoring

```
Loss ← Output
  ↓
Backward Pass
  ↓
[BackwardHook] ← Gradient
  ↓
Capture Gradient
  ↓
Detect Issues
  ↓
Create HookResult
  ↓
Store in Collection
  ↓
Continue Backward
  ↓
Update Parameters
```

---

### Attention Monitoring

```
Attention Layer Forward
         ↓
[AttentionMonitor]
         ↓
Extract Scores (batch, heads, seq, seq)
         ↓
Compute per-head:
  • Entropy
  • Sparsity
  • Patterns
         ↓
Aggregate Across Heads
         ↓
Create HookResult
         ↓
Store Statistics
```

---

## Design Patterns

### 1. Strategy Pattern

Different hooks implement different strategies:

```python
class BaseHook(ABC):
    @abstractmethod
    def setup(self, model):
        """Strategy for hooking"""

class ForwardHook(BaseHook):
    def setup(self, model):
        # Strategy: Hook forward pass

class BackwardHook(BaseHook):
    def setup(self, model):
        # Strategy: Hook backward pass
```

---

### 2. Observer Pattern

Hooks observe model events:

```python
# Subject: PyTorch Module
# Observers: Hooks
# Event: Forward/Backward pass

module.register_forward_hook(observer_callback)
```

---

### 3. Factory Pattern

Hook registry creates hooks:

```python
class HookRegistry:
    @classmethod
    def create(cls, name: str, **kwargs):
        """Factory for creating hooks by name"""
        hook_class = cls.get(name)
        return hook_class(**kwargs)
```

---

### 4. Decorator Pattern

Hooks decorate model behavior:

```python
# Original model behavior
output = model(input)

# Decorated with hooks
output = model(input)  # Hooks execute transparently
```

---

### 5. Template Method Pattern

ReportGenerator defines template:

```python
class ReportGenerator:
    def generate_report(self):
        self._write_header()      # Template method
        self._write_overview()    # Template method
        self._write_analysis()    # Template method
        self._write_footer()      # Template method
```

---

## Extensibility Points

### 1. Custom Hooks

Extend `BaseHook` for new functionality:

```python
class MyCustomHook(BaseHook):
    def setup(self, model):
        # Your hook logic

    def teardown(self):
        # Cleanup
```

---

### 2. Custom Analyzers

Implement new analysis algorithms:

```python
class MyAnalyzer:
    def __init__(self, results: HookResultCollection):
        self.results = results

    def analyze(self):
        # Your analysis logic
```

---

### 3. Custom Visualizations

Add new visualization types:

```python
def plot_my_metric(results, save_path=None):
    # Your visualization logic
    plt.figure()
    # ... plotting ...
    if save_path:
        plt.savefig(save_path)
```

---

### 4. Custom Reports

Create specialized reports:

```python
class MyReportGenerator(ReportGenerator):
    def generate_my_report(self):
        # Your report format
```

---

## Performance Considerations

### Memory Management

**Challenge**: Hooks can accumulate large amounts of data.

**Solutions**:
1. **Lazy Statistics**: Compute on-demand rather than store
2. **Sample Limiting**: `max_samples` parameter
3. **Periodic Clearing**: `manager.clear_results()`
4. **Streaming Analysis**: Process results incrementally

```python
# Good: Limited samples
hook = AttentionMonitor(max_samples=100)

# Good: Periodic cleanup
if batch_idx % 100 == 0:
    manager.clear_results()
```

---

### Computation Overhead

**Challenge**: Each hook adds overhead.

**Optimizations**:
1. **Selective Hooking**: Target specific layers
2. **Statistics Only**: Avoid storing full tensors
3. **Conditional Execution**: Enable/disable dynamically
4. **Batch Processing**: Process multiple results together

```python
# Efficient: Statistics only
hook = ForwardHook(
    compute_stats=True,
    capture_output=False,  # Don't store tensor
)
```

---

### Thread Safety

**Current state**: Single-threaded design.

**For multi-threading**:
- Use separate `HookManager` per thread
- Or implement locking in custom hooks

---

## Testing Strategy

### Unit Tests

Test individual hooks in isolation:

```python
def test_forward_hook():
    model = SimpleModel()
    hook = ForwardHook(layer_name="fc")

    manager = HookManager()
    manager.register(hook)
    manager.apply_to_model(model)

    x = torch.randn(2, 10)
    output = model(x)

    results = manager.get_results()
    assert len(results) == 1
```

---

### Integration Tests

Test multiple hooks together:

```python
def test_multi_hook_integration():
    model = SimpleModel()
    manager = HookManager()

    manager.register(ForwardHook())
    manager.register(BackwardHook())

    # Test interaction
```

---

### End-to-End Tests

Test complete workflows:

```python
def test_optimization_workflow():
    # Complete use case from start to finish
    model = create_model()
    manager = setup_hooks()
    results = run_profiling()
    report = generate_report()
    # Verify complete pipeline
```

---

## Future Enhancements

### Planned Features

1. **Distributed Support**: Better DDP integration
2. **Streaming Analysis**: Real-time monitoring
3. **Model Comparison**: Side-by-side analysis
4. **Time-series Analysis**: Track metrics over time
5. **Interactive Visualization**: Web-based dashboards
6. **AutoML Integration**: Automatic optimization
7. **Export Formats**: More output formats (CSV, Parquet)

### Plugin System

```python
# Future API
from llm_hooks.plugins import Plugin

class MyPlugin(Plugin):
    def on_result(self, result):
        # Custom processing
```

---

## Best Practices

### For Framework Users

1. **Start Small**: Hook 1-3 layers initially
2. **Profile First**: Understand your model before optimizing
3. **Use Context Managers**: Ensure cleanup
4. **Monitor Memory**: Watch for memory growth
5. **Clear Periodically**: Free resources regularly

### For Framework Developers

1. **Follow Patterns**: Use established design patterns
2. **Document Thoroughly**: Every public API documented
3. **Test Extensively**: Unit + integration tests
4. **Minimize Dependencies**: Keep core lightweight
5. **Optimize Carefully**: Profile before optimizing

---

## Conclusion

The LLM Hook Analysis Framework provides:
- ✅ Modular, extensible architecture
- ✅ Comprehensive hooking capabilities
- ✅ Rich analysis and visualization
- ✅ Easy to use and extend
- ✅ Production-ready design

**Core philosophy**: Make model analysis accessible without sacrificing power.

For implementation details, see source code with inline documentation.
