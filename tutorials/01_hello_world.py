"""
Tutorial 1: Hello World - Your First Hook

This tutorial introduces the basics of the LLM Hook framework.
You'll learn how to:
1. Create a simple model
2. Set up a basic hook
3. Collect and view results
"""

import torch
import torch.nn as nn
from llm_hooks import HookManager
from llm_hooks.pytorch import ForwardHook


print("=" * 80)
print("Tutorial 1: Hello World - Your First Hook")
print("=" * 80)
print()

# ============================================================================
# Step 1: Create a Simple Model
# ============================================================================
print("Step 1: Creating a simple neural network...")
print()

class SimpleNet(nn.Module):
    """A simple 3-layer neural network."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleNet()
print(f"Model created: {model}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters())} total")
print()

# ============================================================================
# Step 2: Create a Hook Manager
# ============================================================================
print("Step 2: Creating a hook manager...")
print()

# The HookManager coordinates all hooks
manager = HookManager(name="tutorial_1")
print(f"Hook manager created: {manager}")
print()

# ============================================================================
# Step 3: Register a Forward Hook
# ============================================================================
print("Step 3: Registering a forward hook...")
print()

# ForwardHook captures activations during forward pass
forward_hook = ForwardHook(
    layer_name="fc1",  # Hook the first linear layer
    capture_input=True,
    capture_output=True,
    compute_stats=True,
)

manager.register(forward_hook)
print(f"Registered hook: {forward_hook}")
print()

# ============================================================================
# Step 4: Apply Hooks to Model
# ============================================================================
print("Step 4: Applying hooks to model...")
print()

manager.apply_to_model(model)
print("Hooks applied successfully!")
print()

# ============================================================================
# Step 5: Run Inference
# ============================================================================
print("Step 5: Running inference...")
print()

# Create dummy input
batch_size = 4
input_dim = 10
x = torch.randn(batch_size, input_dim)

print(f"Input shape: {x.shape}")

# Forward pass
with torch.no_grad():
    output = model(x)

print(f"Output shape: {output.shape}")
print()

# ============================================================================
# Step 6: Collect and View Results
# ============================================================================
print("Step 6: Collecting results...")
print()

results = manager.get_results()
print(f"Collected {len(results)} hook results")
print()

# Examine the first result
if results:
    result = results.results[0]
    print("Result details:")
    print(f"  Hook name: {result.hook_name}")
    print(f"  Hook type: {result.hook_type}")
    print(f"  Layer name: {result.layer_name}")
    print(f"  Timestamp: {result.timestamp}")
    print()

    # View captured data
    if 'output' in result.data:
        output_data = result.data['output']
        print("Output statistics:")
        print(f"  Shape: {output_data['shape']}")
        print(f"  Dtype: {output_data['dtype']}")

        if 'stats' in output_data:
            stats = output_data['stats']
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Std: {stats['std']:.4f}")
            print(f"  Min: {stats['min']:.4f}")
            print(f"  Max: {stats['max']:.4f}")
            print(f"  Sparsity: {stats['sparsity']*100:.2f}%")

print()

# ============================================================================
# Step 7: Cleanup
# ============================================================================
print("Step 7: Cleaning up...")
manager.teardown()
print("Hooks removed. Done!")
print()

# ============================================================================
# Summary
# ============================================================================
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
You just learned:
1. How to create a HookManager
2. How to register a ForwardHook on a specific layer
3. How to apply hooks to a model
4. How to run inference and collect results
5. How to view the captured data

Next steps:
- Try Tutorial 2 to learn about backward hooks
- Experiment with different layers and hook types
- Check out the API documentation for more options
""")
