"""
Tutorial 4: Detecting Dead Neurons and Activation Issues

This tutorial teaches you how to monitor neuron activations.
You'll learn how to:
1. Track activation statistics
2. Detect dead neurons
3. Analyze activation distributions
4. Identify optimization opportunities
"""

import torch
import torch.nn as nn
import torch.optim as optim
from llm_hooks import HookManager
from llm_hooks.activations import ActivationMonitor


print("=" * 80)
print("Tutorial 4: Detecting Dead Neurons")
print("=" * 80)
print()

# ============================================================================
# Step 1: Create a Model with Potential Dead Neurons
# ============================================================================
print("Step 1: Creating a neural network...")
print()

class NeuronTestNet(nn.Module):
    """A network that might develop dead neurons."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(50, 100)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(100, 100)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

model = NeuronTestNet()
model.train()

print(f"Created network with {sum(p.numel() for p in model.parameters())} parameters")
print()

# ============================================================================
# Step 2: Set Up Activation Monitoring
# ============================================================================
print("Step 2: Setting up activation monitoring...")
print()

manager = HookManager()

activation_monitor = ActivationMonitor(
    layer_pattern="*",              # Monitor all layers
    dead_neuron_threshold=0.01,     # Threshold for dead neuron detection
    track_distribution=True,        # Track activation distributions
    track_sparsity=True,           # Track sparsity
)

manager.register(activation_monitor)
manager.apply_to_model(model)

print("Activation monitoring enabled!")
print()

# ============================================================================
# Step 3: Train the Model
# ============================================================================
print("Step 3: Training the model...")
print()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_batches = 50

print("Training for 50 batches...")
for i in range(num_batches):
    # Generate random data
    x = torch.randn(32, 50)
    y = torch.randint(0, 10, (32,))

    # Training step
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    if (i + 1) % 10 == 0:
        print(f"  Batch {i+1}/{num_batches}, Loss: {loss.item():.4f}")

print()

# ============================================================================
# Step 4: Analyze Dead Neurons
# ============================================================================
print("Step 4: Analyzing dead neurons...")
print()

# Get dead neuron report
dead_report = activation_monitor.get_dead_neuron_report()

print("Dead Neuron Analysis:")
print("=" * 60)

total_neurons = 0
total_dead = 0

for layer_name, stats in dead_report.items():
    num_neurons = stats['total_neurons']
    num_dead = stats['dead_neurons']
    dead_ratio = stats['dead_ratio']

    total_neurons += num_neurons
    total_dead += num_dead

    print(f"\n{layer_name}:")
    print(f"  Total neurons: {num_neurons}")
    print(f"  Dead neurons: {num_dead}")
    print(f"  Dead ratio: {dead_ratio*100:.2f}%")
    print(f"  Mean activation: {stats['mean_activation']:.6f}")

    # Severity indication
    if dead_ratio > 0.5:
        print("  ⚠️  CRITICAL: >50% neurons are dead!")
        print("     → Consider: Lower learning rate, change initialization")
    elif dead_ratio > 0.3:
        print("  ⚠️  WARNING: >30% neurons are dead")
        print("     → Consider: Adjust learning rate, use LeakyReLU")
    elif dead_ratio > 0.1:
        print("  ⚡ NOTICE: Some dead neurons detected")
        print("     → Monitor during training")
    else:
        print("  ✅ Healthy activation pattern")

print()
print("=" * 60)
print(f"Overall Statistics:")
print(f"  Total neurons: {total_neurons}")
print(f"  Total dead: {total_dead}")
print(f"  Global dead ratio: {(total_dead/total_neurons)*100:.2f}%")
print()

# ============================================================================
# Step 5: Analyze Activation Distributions
# ============================================================================
print("Step 5: Analyzing activation distributions...")
print()

results = manager.get_results()

# Find layers with interesting patterns
print("Activation Statistics by Layer:")
print()

layer_stats = {}
for result in results.results:
    layer_name = result.layer_name
    if layer_name not in layer_stats:
        layer_stats[layer_name] = []

    if 'sparsity' in result.data:
        layer_stats[layer_name].append(result.data['sparsity'])

# Analyze sparsity
for layer_name, sparsity_list in list(layer_stats.items())[:5]:
    if sparsity_list:
        avg_sparsity = sum(s['zero_ratio'] for s in sparsity_list) / len(sparsity_list)
        avg_active = sum(s['active_ratio'] for s in sparsity_list) / len(sparsity_list)

        print(f"{layer_name}:")
        print(f"  Zero ratio: {avg_sparsity*100:.2f}%")
        print(f"  Active ratio: {avg_active*100:.2f}%")

        # Interpretation
        if avg_sparsity > 0.8:
            print("  💡 Very sparse activations → Good for pruning")
        elif avg_sparsity > 0.5:
            print("  💡 Moderate sparsity → Efficient")
        else:
            print("  💡 Dense activations → High compute cost")
        print()

# ============================================================================
# Step 6: Recommendations
# ============================================================================
print("=" * 80)
print("DEAD NEURON SOLUTIONS")
print("=" * 80)
print("""
Why Dead Neurons Occur:
1. ReLU zeroes out negative values
2. Poor weight initialization
3. High learning rate
4. Vanishing gradients
5. Dataset bias

Solutions:

1. **Change Activation Function**:
   - Use LeakyReLU instead of ReLU
   - Try GELU, SiLU, or Swish
   - Example:
     self.relu = nn.LeakyReLU(0.01)

2. **Adjust Learning Rate**:
   - Lower learning rate
   - Use learning rate scheduling
   - Example:
     optimizer = optim.Adam(model.parameters(), lr=0.0001)

3. **Better Initialization**:
   - Use Xavier or He initialization
   - Example:
     nn.init.kaiming_normal_(layer.weight)

4. **Add Batch Normalization**:
   - Normalizes activations
   - Helps gradient flow
   - Example:
     self.bn = nn.BatchNorm1d(100)

5. **Monitor During Training**:
   - Check dead neuron ratio every epoch
   - Adjust if ratio increases
   - Stop if >50% dead

6. **Pruning Strategy**:
   - Remove consistently dead neurons
   - Can reduce model size by 20-50%
   - Use Fisher information for importance

Prevention:
- Start with small learning rate
- Use proper initialization
- Monitor from epoch 0
- Compare different activation functions

Next Tutorial: Learn about Fisher-based pruning!
""")

manager.teardown()
