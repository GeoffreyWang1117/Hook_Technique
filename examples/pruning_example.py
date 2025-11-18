"""
Example: Using Fisher information for structured pruning.
"""

import torch
import torch.nn as nn
from llm_hooks import HookManager
from llm_hooks.fisher import FisherHook


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def main():
    print("Fisher-based Pruning Example")
    print("=" * 80)
    print()

    # Create model
    model = SimpleModel()
    model.train()

    # Setup Fisher hook
    manager = HookManager()
    fisher_hook = FisherHook(layer_pattern="*", accumulate_samples=100)
    manager.register(fisher_hook)
    manager.apply_to_model(model)

    # Train for a few iterations to accumulate Fisher information
    print("Accumulating Fisher information...")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for i in range(100):
        # Dummy data
        x = torch.randn(32, 100)
        y = torch.randint(0, 10, (32,))

        # Forward
        output = model(x)
        loss = criterion(output, y)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update Fisher hook
        fisher_hook.on_batch_end()

        if (i + 1) % 20 == 0:
            print(f"  Iteration {i+1}/100, Loss: {loss.item():.4f}")

    print()

    # Get Fisher importance
    print("Analyzing parameter importance...")
    importance = fisher_hook.get_fisher_importance(normalize=True)

    for name, fisher in importance.items():
        print(f"{name}:")
        print(f"  Mean importance: {fisher.mean().item():.6f}")
        print(f"  Max importance:  {fisher.max().item():.6f}")
        print(f"  Min importance:  {fisher.min().item():.6f}")
        print()

    # Generate pruning masks
    print("Generating pruning masks...")
    pruning_ratios = [0.1, 0.3, 0.5]

    for ratio in pruning_ratios:
        masks = fisher_hook.get_pruning_mask(pruning_ratio=ratio, granularity='weight')

        total_params = 0
        pruned_params = 0

        for name, mask in masks.items():
            total = mask.numel()
            kept = mask.sum().item()
            pruned = total - kept

            total_params += total
            pruned_params += pruned

        print(f"\nPruning ratio {ratio*100:.0f}%:")
        print(f"  Total parameters: {total_params}")
        print(f"  Pruned parameters: {pruned_params}")
        print(f"  Remaining: {total_params - pruned_params} ({(1-ratio)*100:.0f}%)")

    # Get sensitivity report
    print("\nSensitivity Report:")
    report = fisher_hook.get_sensitivity_report()

    for name, stats in report.items():
        if name == 'global':
            continue
        print(f"{name}:")
        print(f"  Total importance: {stats['total_importance']:.6f}")
        print(f"  Mean importance:  {stats['mean_importance']:.6f}")

    manager.teardown()
    print("\nDone!")


if __name__ == "__main__":
    main()
