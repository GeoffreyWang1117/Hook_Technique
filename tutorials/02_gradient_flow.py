"""
Tutorial 2: Understanding Gradient Flow

This tutorial teaches you how to track gradients during backpropagation.
You'll learn how to:
1. Set up backward hooks
2. Track gradient statistics
3. Detect gradient issues (vanishing/exploding)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from llm_hooks import HookManager
from llm_hooks.pytorch import BackwardHook
from llm_hooks.gradients import GradientTracker


print("=" * 80)
print("Tutorial 2: Understanding Gradient Flow")
print("=" * 80)
print()

# ============================================================================
# Step 1: Create a Deeper Model
# ============================================================================
print("Step 1: Creating a deeper neural network...")
print()

class DeepNet(nn.Module):
    """A deeper network to demonstrate gradient flow."""
    def __init__(self, num_layers=5):
        super().__init__()
        layers = []
        in_features = 50

        for i in range(num_layers):
            layers.append(nn.Linear(in_features, in_features))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(in_features, 10))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

model = DeepNet(num_layers=5)
model.train()  # Set to training mode
print(f"Created a deep network with {len(list(model.parameters()))} parameter tensors")
print()

# ============================================================================
# Step 2: Set Up Hooks
# ============================================================================
print("Step 2: Setting up gradient tracking hooks...")
print()

manager = HookManager()

# Option 1: BackwardHook - captures gradients at specific layers
backward_hook = BackwardHook(
    layer_pattern="network.*",  # Hook all layers in the network
    detect_vanishing=True,
    detect_exploding=True,
    vanishing_threshold=1e-6,
    exploding_threshold=100.0,
)

# Option 2: GradientTracker - tracks gradients on parameters
gradient_tracker = GradientTracker(
    layer_pattern="*weight",  # Track all weight parameters
    detect_issues=True,
)

manager.register(backward_hook)
manager.register(gradient_tracker)
manager.apply_to_model(model)

print("Hooks registered:")
print(f"  - {backward_hook}")
print(f"  - {gradient_tracker}")
print()

# ============================================================================
# Step 3: Training Loop
# ============================================================================
print("Step 3: Running training iterations...")
print()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

num_iterations = 10

for i in range(num_iterations):
    # Generate random data
    x = torch.randn(32, 50)
    y = torch.randint(0, 10, (32,))

    # Forward pass
    output = model(x)
    loss = criterion(output, y)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i + 1) % 5 == 0:
        print(f"Iteration {i+1}/{num_iterations}, Loss: {loss.item():.4f}")

print()

# ============================================================================
# Step 4: Analyze Gradient Flow
# ============================================================================
print("Step 4: Analyzing gradient flow...")
print()

# Get gradient flow summary
gradient_summary = gradient_tracker.get_gradient_flow_summary()

print("Gradient Flow Analysis:")
print(f"  Total parameters tracked: {gradient_summary['global']['total_parameters']}")
print(f"  Total issues detected: {gradient_summary['global']['total_issues']}")
print()

# Check for gradient issues
if gradient_summary['global']['total_issues'] > 0:
    print("⚠️  Gradient issues detected:")
    issue_types = gradient_summary['global']['issue_types']
    for issue_type in issue_types:
        print(f"  - {issue_type}")
    print()

    # Get problematic layers
    problematic = gradient_tracker.get_problematic_layers()
    print(f"Problematic layers: {len(problematic)}")
    for layer in problematic[:5]:  # Show first 5
        print(f"  - {layer}")
else:
    print("✅ No gradient issues detected!")

print()

# ============================================================================
# Step 5: Visualize Gradient Statistics
# ============================================================================
print("Step 5: Gradient statistics per parameter...")
print()

# Show statistics for each parameter
for param_name, stats in list(gradient_summary.items())[:5]:
    if param_name == 'global':
        continue

    print(f"{param_name}:")
    print(f"  Avg gradient norm: {stats['stats']['norm_l2']:.6f}")
    print(f"  Avg abs mean: {stats['stats']['abs_mean']:.6f}")
    print(f"  Max abs value: {stats['stats']['abs_max']:.6f}")

    if stats['has_issues']:
        print(f"  ⚠️  Issues: {[issue['type'] for issue in stats['issues']]}")

    print()

# ============================================================================
# Step 6: Best Practices
# ============================================================================
print("=" * 80)
print("BEST PRACTICES")
print("=" * 80)
print("""
Gradient Flow Tips:

1. **Vanishing Gradients**:
   - Common in deep networks
   - Solutions: Use ReLU/GELU, batch normalization, residual connections
   - Consider gradient clipping

2. **Exploding Gradients**:
   - Can cause training instability
   - Solutions: Gradient clipping, lower learning rate
   - Check weight initialization

3. **Monitoring**:
   - Track gradient norms across layers
   - Watch for sudden changes
   - Monitor throughout training

4. **Thresholds**:
   - Vanishing: < 1e-6 (adjustable)
   - Exploding: > 100.0 (adjustable)
   - Tune based on your model

Next Tutorial: Learn about attention analysis!
""")

manager.teardown()
