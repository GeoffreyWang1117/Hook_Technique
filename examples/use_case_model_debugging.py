"""
Use Case: Debugging Training Issues

This example shows how to use hooks to debug common training problems:
1. Vanishing/exploding gradients
2. Dead neurons
3. NaN/Inf values
4. Learning instability
"""

import torch
import torch.nn as nn
import torch.optim as optim
from llm_hooks import HookManager
from llm_hooks.pytorch import BackwardHook
from llm_hooks.gradients import GradientTracker
from llm_hooks.activations import ActivationMonitor


print("=" * 80)
print("Use Case: Debugging Training Issues")
print("=" * 80)
print()

# ============================================================================
# Scenario: A Model with Training Problems
# ============================================================================
print("Scenario: Debugging a problematic model...")
print()

class ProblematicNet(nn.Module):
    """A network with potential training issues."""
    def __init__(self):
        super().__init__()
        # Deep network without proper initialization
        layers = []
        for i in range(10):  # Very deep!
            layers.append(nn.Linear(128, 128))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(128, 10))
        self.network = nn.Sequential(*layers)

        # Poor initialization (will cause issues)
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=0.1)

    def forward(self, x):
        return self.network(x)

model = ProblematicNet()
model.train()

print("Created a deep network (10 layers) with potential issues")
print()

# ============================================================================
# Set Up Debugging Hooks
# ============================================================================
print("Setting up debugging hooks...")
print()

manager = HookManager(name="debugger")

# 1. Gradient tracker for gradient issues
gradient_tracker = GradientTracker(
    layer_pattern="*",
    detect_issues=True,
    vanishing_threshold=1e-7,
    exploding_threshold=10.0,
)

# 2. Backward hook for detailed gradient analysis
backward_hook = BackwardHook(
    layer_pattern="network.*",
    detect_vanishing=True,
    detect_exploding=True,
)

# 3. Activation monitor for dead neurons
activation_monitor = ActivationMonitor(
    layer_pattern="*",
    dead_neuron_threshold=0.01,
)

manager.register(gradient_tracker)
manager.register(backward_hook)
manager.register(activation_monitor)
manager.apply_to_model(model)

print("Debugging hooks activated!")
print()

# ============================================================================
# Training Loop with Monitoring
# ============================================================================
print("Starting training...")
print("=" * 60)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)  # High LR on purpose

num_epochs = 5
issues_detected = []

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    print("-" * 60)

    epoch_loss = 0
    num_batches = 10

    for batch_idx in range(num_batches):
        # Generate data
        x = torch.randn(32, 128)
        y = torch.randint(0, 10, (32,))

        # Training step
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)

        # Check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  ⚠️  CRITICAL: NaN/Inf loss detected at batch {batch_idx}!")
            issues_detected.append({
                'epoch': epoch + 1,
                'batch': batch_idx,
                'type': 'nan_loss',
            })
            break

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / num_batches
    print(f"  Average Loss: {avg_loss:.4f}")

    # Check for gradient issues after each epoch
    grad_summary = gradient_tracker.get_gradient_flow_summary()

    if grad_summary['global']['total_issues'] > 0:
        print(f"  ⚠️  {grad_summary['global']['total_issues']} gradient issues detected!")

        issue_types = grad_summary['global']['issue_types']
        for issue_type in issue_types:
            print(f"    - {issue_type}")

        issues_detected.append({
            'epoch': epoch + 1,
            'type': 'gradient_issues',
            'details': issue_types,
        })

    # Check for dead neurons
    dead_report = activation_monitor.get_dead_neuron_report()
    total_dead = sum(stats['dead_neurons'] for stats in dead_report.values())

    if total_dead > 0:
        print(f"  ⚠️  {total_dead} dead neurons detected")
        issues_detected.append({
            'epoch': epoch + 1,
            'type': 'dead_neurons',
            'count': total_dead,
        })

print()

# ============================================================================
# Diagnostic Report
# ============================================================================
print("=" * 80)
print("DIAGNOSTIC REPORT")
print("=" * 80)
print()

if issues_detected:
    print(f"⚠️  ISSUES FOUND: {len(issues_detected)} problems detected\n")

    # Categorize issues
    issue_types = {}
    for issue in issues_detected:
        itype = issue['type']
        if itype not in issue_types:
            issue_types[itype] = []
        issue_types[itype].append(issue)

    # Report each issue type
    for itype, issues in issue_types.items():
        print(f"{itype.upper().replace('_', ' ')}:")
        print(f"  Occurrences: {len(issues)}")

        if itype == 'gradient_issues':
            # Collect all gradient issue types
            all_grad_issues = set()
            for issue in issues:
                all_grad_issues.update(issue.get('details', []))

            print(f"  Types: {', '.join(all_grad_issues)}")

        elif itype == 'dead_neurons':
            max_dead = max(issue['count'] for issue in issues)
            print(f"  Peak dead neurons: {max_dead}")

        print()

else:
    print("✅ No major issues detected!")
    print()

# ============================================================================
# Detailed Analysis
# ============================================================================
print("=" * 80)
print("DETAILED ANALYSIS")
print("=" * 80)
print()

# Gradient flow analysis
print("Gradient Flow:")
print("-" * 60)
grad_summary = gradient_tracker.get_gradient_flow_summary()
problematic_layers = gradient_tracker.get_problematic_layers()

if problematic_layers:
    print(f"Problematic layers: {len(problematic_layers)}")
    print("\nTop 5 problematic layers:")
    for layer in problematic_layers[:5]:
        if layer in grad_summary:
            stats = grad_summary[layer]['stats']
            print(f"  {layer}:")
            print(f"    Gradient norm: {stats['norm_l2']:.6f}")
            print(f"    Abs mean: {stats['abs_mean']:.6f}")

            issues = grad_summary[layer].get('issues', [])
            if issues:
                print(f"    Issues: {[i['type'] for i in issues]}")
else:
    print("✅ All gradients are healthy")

print()

# Dead neuron analysis
print("Dead Neuron Analysis:")
print("-" * 60)
dead_report = activation_monitor.get_dead_neuron_report()

if dead_report:
    total_neurons = sum(stats['total_neurons'] for stats in dead_report.values())
    total_dead = sum(stats['dead_neurons'] for stats in dead_report.values())
    dead_ratio = total_dead / total_neurons if total_neurons > 0 else 0

    print(f"Total neurons: {total_neurons}")
    print(f"Dead neurons: {total_dead}")
    print(f"Dead ratio: {dead_ratio*100:.2f}%")
    print()

    print("Most affected layers:")
    sorted_layers = sorted(
        dead_report.items(),
        key=lambda x: x[1]['dead_ratio'],
        reverse=True
    )

    for layer_name, stats in sorted_layers[:5]:
        print(f"  {layer_name}:")
        print(f"    Dead: {stats['dead_neurons']}/{stats['total_neurons']}")
        print(f"    Ratio: {stats['dead_ratio']*100:.2f}%")

print()

# ============================================================================
# Solutions and Recommendations
# ============================================================================
print("=" * 80)
print("SOLUTIONS & RECOMMENDATIONS")
print("=" * 80)
print()

solutions = []

# Check for vanishing gradients
if 'vanishing_gradient' in issue_types:
    solutions.append({
        'problem': 'Vanishing Gradients',
        'causes': [
            'Network too deep',
            'Poor weight initialization',
            'Saturating activation functions',
        ],
        'solutions': [
            '1. Use better initialization (Xavier/He)',
            '2. Add batch normalization',
            '3. Use residual connections',
            '4. Try different activations (GELU, SiLU)',
            '5. Reduce network depth',
        ],
        'code_example': '''
# Better initialization
for layer in model.modules():
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight, mode='fan_out')

# Add batch normalization
self.bn = nn.BatchNorm1d(128)
x = self.bn(x)

# Add residual connections
x = x + self.layer(x)
        ''',
    })

# Check for exploding gradients
if 'exploding_gradient' in issue_types:
    solutions.append({
        'problem': 'Exploding Gradients',
        'causes': [
            'Learning rate too high',
            'Poor weight initialization',
            'Unstable architecture',
        ],
        'solutions': [
            '1. Reduce learning rate',
            '2. Use gradient clipping',
            '3. Better weight initialization',
            '4. Add layer normalization',
        ],
        'code_example': '''
# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Lower learning rate
optimizer = optim.SGD(model.parameters(), lr=0.001)  # Was 0.1

# Layer normalization
self.norm = nn.LayerNorm(128)
        ''',
    })

# Check for dead neurons
if 'dead_neurons' in issue_types:
    solutions.append({
        'problem': 'Dead Neurons',
        'causes': [
            'ReLU kills negative values',
            'Learning rate too high',
            'Poor initialization',
        ],
        'solutions': [
            '1. Use LeakyReLU instead of ReLU',
            '2. Lower learning rate',
            '3. Better initialization',
            '4. Monitor from start',
        ],
        'code_example': '''
# Use LeakyReLU
self.activation = nn.LeakyReLU(0.01)

# Or use GELU (smooth activation)
self.activation = nn.GELU()

# Better initialization
nn.init.kaiming_normal_(layer.weight, nonlinearity='leaky_relu')
        ''',
    })

# Display solutions
for i, sol in enumerate(solutions, 1):
    print(f"{i}. {sol['problem']}")
    print("-" * 60)
    print("\nPossible Causes:")
    for cause in sol['causes']:
        print(f"  • {cause}")

    print("\nRecommended Solutions:")
    for solution in sol['solutions']:
        print(f"  {solution}")

    print(f"\nCode Example:")
    print(sol['code_example'])
    print()

# General recommendations
print("=" * 60)
print("GENERAL RECOMMENDATIONS")
print("=" * 60)
print("""
1. Always monitor training from epoch 0
2. Start with small learning rate (1e-4 to 1e-3)
3. Use proper weight initialization
4. Add normalization layers (BatchNorm/LayerNorm)
5. Consider using modern activations (GELU, SiLU)
6. Implement gradient clipping for stability
7. Use residual connections for deep networks
8. Monitor gradients, activations, and loss regularly

Prevention Checklist:
☑ Proper weight initialization
☑ Appropriate learning rate
☑ Gradient clipping enabled
☑ Normalization layers added
☑ Suitable activation functions
☑ Monitoring hooks in place
☑ Validation checks implemented
""")

manager.teardown()
print("\nDebugging session complete!")
