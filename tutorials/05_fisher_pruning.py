"""
Tutorial 5: Model Pruning with Fisher Information

This tutorial teaches you how to use Fisher information for intelligent pruning.
You'll learn how to:
1. Estimate Fisher information during training
2. Identify important vs unimportant parameters
3. Generate pruning masks at different granularities
4. Evaluate pruned model performance
"""

import torch
import torch.nn as nn
import torch.optim as optim
from llm_hooks import HookManager
from llm_hooks.fisher import FisherHook
import copy


print("=" * 80)
print("Tutorial 5: Model Pruning with Fisher Information")
print("=" * 80)
print()

# ============================================================================
# Step 1: Create and Train a Model
# ============================================================================
print("Step 1: Creating a model for pruning...")
print()

class PruningTestNet(nn.Module):
    """A simple network for demonstrating pruning."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 200)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(200, 200)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

model = PruningTestNet()
model.train()

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Model created with {total_params:,} parameters")
print()

# ============================================================================
# Step 2: Set Up Fisher Hook
# ============================================================================
print("Step 2: Setting up Fisher information tracking...")
print()

manager = HookManager()

fisher_hook = FisherHook(
    layer_pattern="*",          # Track all layers
    accumulate_samples=100,     # Number of batches to accumulate
)

manager.register(fisher_hook)
manager.apply_to_model(model)

print("Fisher hook registered!")
print()

# ============================================================================
# Step 3: Train and Accumulate Fisher Information
# ============================================================================
print("Step 3: Training model and accumulating Fisher information...")
print()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_batches = 100

print(f"Training for {num_batches} batches...")
for i in range(num_batches):
    # Generate random data
    x = torch.randn(32, 100)
    y = torch.randint(0, 10, (32,))

    # Training step
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    # Important: Notify Fisher hook after each batch
    fisher_hook.on_batch_end()

    if (i + 1) % 20 == 0:
        print(f"  Batch {i+1}/{num_batches}, Loss: {loss.item():.4f}")

print()

# ============================================================================
# Step 4: Analyze Parameter Importance
# ============================================================================
print("Step 4: Analyzing parameter importance...")
print()

# Get Fisher information (normalized)
fisher_importance = fisher_hook.get_fisher_importance(normalize=True)

print("Parameter Importance (Fisher Information):")
print("=" * 60)

importance_summary = []
for name, fisher in fisher_importance.items():
    mean_importance = fisher.mean().item()
    max_importance = fisher.max().item()
    total_importance = fisher.sum().item()

    importance_summary.append({
        'name': name,
        'mean': mean_importance,
        'max': max_importance,
        'total': total_importance,
    })

    print(f"\n{name}:")
    print(f"  Mean importance: {mean_importance:.6f}")
    print(f"  Max importance:  {max_importance:.6f}")
    print(f"  Total importance: {total_importance:.6f}")

print()

# Sort by total importance
importance_summary.sort(key=lambda x: x['total'], reverse=True)

print("Most Important Parameters:")
for i, item in enumerate(importance_summary[:3], 1):
    print(f"  {i}. {item['name']} (Total: {item['total']:.6f})")

print()

# ============================================================================
# Step 5: Generate Pruning Masks
# ============================================================================
print("Step 5: Generating pruning masks at different ratios...")
print()

pruning_ratios = [0.1, 0.3, 0.5, 0.7]

print("Pruning Analysis:")
print("=" * 60)

for ratio in pruning_ratios:
    print(f"\nPruning Ratio: {ratio*100:.0f}%")

    # Generate masks for weight-level pruning
    masks_weight = fisher_hook.get_pruning_mask(
        pruning_ratio=ratio,
        granularity='weight'
    )

    # Count pruned parameters
    total = 0
    pruned = 0
    for name, mask in masks_weight.items():
        param_total = mask.numel()
        param_pruned = (mask == 0).sum().item()

        total += param_total
        pruned += param_pruned

    remaining = total - pruned
    compression = (1 - ratio) * 100

    print(f"  Total parameters: {total:,}")
    print(f"  Pruned: {pruned:,}")
    print(f"  Remaining: {remaining:,} ({compression:.0f}%)")
    print(f"  Actual compression: {(pruned/total)*100:.1f}%")

print()

# ============================================================================
# Step 6: Apply Pruning and Evaluate
# ============================================================================
print("Step 6: Applying pruning and evaluating...")
print()

# Choose a pruning ratio
target_ratio = 0.3
print(f"Applying {target_ratio*100:.0f}% pruning...")

# Get masks
masks = fisher_hook.get_pruning_mask(
    pruning_ratio=target_ratio,
    granularity='weight'
)

# Apply masks to model
pruned_model = copy.deepcopy(model)
for name, param in pruned_model.named_parameters():
    if name in masks:
        mask = masks[name].to(param.device)
        param.data *= mask

print("Pruning applied!")
print()

# Evaluate both models
def evaluate(model, num_samples=1000):
    """Simple evaluation."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for _ in range(num_samples // 32):
            x = torch.randn(32, 100)
            y = torch.randint(0, 10, (32,))

            output = model(x)
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / total

print("Evaluating models...")
original_acc = evaluate(model)
pruned_acc = evaluate(pruned_model)

print(f"Original model accuracy: {original_acc*100:.2f}%")
print(f"Pruned model accuracy: {pruned_acc*100:.2f}%")
print(f"Accuracy drop: {(original_acc - pruned_acc)*100:.2f}%")
print()

# ============================================================================
# Step 7: Different Pruning Granularities
# ============================================================================
print("Step 7: Comparing pruning granularities...")
print()

granularities = ['weight', 'neuron', 'channel']

print("Granularity Comparison:")
print("=" * 60)

for granularity in granularities:
    print(f"\n{granularity.upper()} Pruning:")

    try:
        masks = fisher_hook.get_pruning_mask(
            pruning_ratio=0.3,
            granularity=granularity
        )

        total = sum(m.numel() for m in masks.values())
        pruned = sum((m == 0).sum().item() for m in masks.values())

        print(f"  Parameters pruned: {pruned:,} / {total:,}")
        print(f"  Pruning ratio: {(pruned/total)*100:.1f}%")

        # Characteristics
        if granularity == 'weight':
            print("  💡 Unstructured pruning - fine-grained but needs sparse support")
        elif granularity == 'neuron':
            print("  💡 Structured pruning - removes entire neurons")
        elif granularity == 'channel':
            print("  💡 Channel pruning - for convolutional layers")

    except Exception as e:
        print(f"  ⚠️  Not applicable for this model: {e}")

print()

# ============================================================================
# Step 8: Best Practices
# ============================================================================
print("=" * 80)
print("FISHER PRUNING BEST PRACTICES")
print("=" * 80)
print("""
Understanding Fisher Information:
- Estimates parameter importance using gradient information
- Higher Fisher value = more important parameter
- Accumulated over multiple training samples

Pruning Strategies:

1. **Weight-Level (Unstructured)**:
   - Finest granularity
   - Best compression potential
   - Requires sparse matrix support
   - Good for: Research, sparse accelerators

2. **Neuron-Level (Structured)**:
   - Removes entire neurons/units
   - No special hardware needed
   - Easier to accelerate
   - Good for: Production deployment

3. **Channel-Level (Structured)**:
   - For convolutional layers
   - Hardware-friendly
   - Maintains spatial structure
   - Good for: CNN optimization

Workflow:
1. Train model normally
2. Accumulate Fisher (100-1000 batches)
3. Start with low pruning ratio (10-20%)
4. Fine-tune after pruning
5. Gradually increase ratio if needed
6. Monitor accuracy drop (<5% acceptable)

Tips:
- Accumulate Fisher on representative data
- Different layers have different sensitivity
- Combine with quantization for max compression
- Iterative pruning often works better
- Always fine-tune after pruning

Advanced:
- Layer-wise pruning ratios
- Sensitivity-based thresholds
- Gradual magnitude pruning
- Lottery ticket hypothesis

Next Tutorial: Learn about end-to-end model optimization!
""")

manager.teardown()
