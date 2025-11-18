# API Documentation

## Core Classes

### HookManager

The main interface for managing hooks.

```python
class HookManager:
    def __init__(self, name: str = "default")
    def register(self, hook: BaseHook) -> HookManager
    def unregister(self, hook: BaseHook) -> HookManager
    def apply_to_model(self, model: nn.Module) -> HookManager
    def teardown(self) -> None
    def get_results(self) -> HookResultCollection
    def visualize(self, results: Optional[HookResultCollection] = None) -> None
```

**Methods:**

- `register(hook)` - Register a hook with the manager
- `unregister(hook)` - Remove a hook from the manager
- `apply_to_model(model)` - Apply all hooks to a PyTorch model
- `teardown()` - Remove all hooks and cleanup
- `get_results()` - Get all collected results
- `visualize(results)` - Generate visualizations

### BaseHook

Abstract base class for all hooks.

```python
class BaseHook(ABC):
    def __init__(self, name: Optional[str] = None, enabled: bool = True)
    def setup(self, model: Any) -> None
    def teardown(self) -> None
    def enable(self) -> None
    def disable(self) -> None
    def get_results(self) -> List[HookResult]
    def clear_results(self) -> None
```

## PyTorch Hooks

### ForwardHook

Captures forward pass activations.

```python
class ForwardHook(BaseHook):
    def __init__(
        self,
        layer_name: Optional[str] = None,
        layer_pattern: Optional[str] = None,
        capture_input: bool = True,
        capture_output: bool = True,
        compute_stats: bool = True,
        name: Optional[str] = None,
    )
```

**Parameters:**

- `layer_name` - Specific layer to hook (e.g., "transformer.layer.0")
- `layer_pattern` - Pattern to match layers (e.g., "transformer.*")
- `capture_input` - Whether to capture input tensors
- `capture_output` - Whether to capture output tensors
- `compute_stats` - Whether to compute statistics

### BackwardHook

Captures backward pass gradients.

```python
class BackwardHook(BaseHook):
    def __init__(
        self,
        layer_name: Optional[str] = None,
        layer_pattern: Optional[str] = None,
        capture_input_grad: bool = True,
        capture_output_grad: bool = True,
        compute_stats: bool = True,
        detect_vanishing: bool = True,
        detect_exploding: bool = True,
        name: Optional[str] = None,
    )
```

## Attention Hooks

### AttentionMonitor

Monitors attention patterns and statistics.

```python
class AttentionMonitor(BaseHook):
    def __init__(
        self,
        layer_pattern: str = "*attention*",
        record_scores: bool = True,
        record_patterns: bool = True,
        compute_entropy: bool = True,
        compute_sparsity: bool = True,
        max_samples: int = 100,
        name: Optional[str] = None,
    )

    def get_attention_summary(self) -> Dict[str, Any]
```

**Methods:**

- `get_attention_summary()` - Get aggregated attention statistics

### KVCacheMonitor

Monitors KV cache usage and efficiency.

```python
class KVCacheMonitor(BaseHook):
    def __init__(
        self,
        layer_pattern: str = "*attention*",
        track_size: bool = True,
        track_memory: bool = True,
        track_reuse: bool = True,
        name: Optional[str] = None,
    )

    def get_cache_summary(self) -> Dict[str, Any]
```

## Activation Hooks

### ActivationMonitor

Monitors activation statistics and dead neurons.

```python
class ActivationMonitor(BaseHook):
    def __init__(
        self,
        layer_pattern: str = "*",
        dead_neuron_threshold: float = 0.01,
        track_distribution: bool = True,
        track_sparsity: bool = True,
        name: Optional[str] = None,
    )

    def get_dead_neuron_report(self) -> Dict[str, Any]
```

## Fisher Information

### FisherHook

Estimates Fisher information for pruning.

```python
class FisherHook(BaseHook):
    def __init__(
        self,
        layer_pattern: str = "*",
        accumulate_samples: int = 100,
        name: Optional[str] = None,
    )

    def on_batch_end(self) -> None
    def get_fisher_importance(self, normalize: bool = True) -> Dict[str, torch.Tensor]
    def get_pruning_mask(self, pruning_ratio: float, granularity: str = "weight") -> Dict[str, torch.Tensor]
    def get_sensitivity_report(self) -> Dict[str, Any]
```

**Methods:**

- `on_batch_end()` - Call after each training batch
- `get_fisher_importance()` - Get Fisher information matrices
- `get_pruning_mask()` - Generate pruning masks
- `get_sensitivity_report()` - Get parameter sensitivity report

## Gradient Tracking

### GradientTracker

Tracks gradient flow and detects issues.

```python
class GradientTracker(BaseHook):
    def __init__(
        self,
        layer_pattern: str = "*",
        detect_issues: bool = True,
        vanishing_threshold: float = 1e-6,
        exploding_threshold: float = 100.0,
        name: Optional[str] = None,
    )

    def get_gradient_flow_summary(self) -> Dict[str, Any]
    def get_problematic_layers(self) -> List[str]
```

## CUDA Profiling

### CUDAProfiler

Profiles CUDA kernel execution.

```python
class CUDAProfiler(BaseHook):
    def __init__(
        self,
        with_stack: bool = True,
        with_flops: bool = True,
        profile_memory: bool = True,
        record_shapes: bool = True,
        name: Optional[str] = None,
    )

    def start_profiling(self) -> None
    def stop_profiling(self) -> None
    def get_performance_summary(self) -> Dict[str, Any]
    def export_trace(self, filename: str) -> None
```

## Analysis Tools

### BottleneckAnalyzer

Analyzes performance bottlenecks.

```python
class BottleneckAnalyzer:
    def __init__(self, results: HookResultCollection)

    def identify_slow_layers(self, threshold_percentile: float = 90) -> List[str]
    def detect_gradient_issues(self) -> Dict[str, List[str]]
    def find_dead_neurons(self) -> Dict[str, int]
    def analyze_memory_usage(self) -> Dict[str, Any]
    def generate_summary(self) -> Dict[str, Any]
```

### ReportGenerator

Generates analysis reports.

```python
class ReportGenerator:
    def __init__(self, results: HookResultCollection)

    def generate_text_report(self) -> str
    def generate_json_report(self) -> str
    def save_report(self, filepath: str, format: str = 'text') -> None
```

## Visualization

### Visualization Functions

```python
def visualize_results(results: HookResultCollection, output_dir: str = "./visualizations")
def plot_attention(results: List, save_path: Optional[str] = None)
def plot_gradients(results: List, save_path: Optional[str] = None)
def plot_activations(results: List, save_path: Optional[str] = None)
```
