"""
Basic usage example of LLM Hook Analysis Framework.
"""

import torch
import torch.nn as nn
from llm_hooks import HookManager
from llm_hooks.pytorch import ForwardHook, BackwardHook
from llm_hooks.attention import AttentionMonitor
from llm_hooks.activations import ActivationMonitor
from llm_hooks.analysis import ReportGenerator


# Define a simple transformer-like model
class SimpleTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(1000, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 1000)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc_out(x.mean(dim=1))
        return x


def main():
    print("=" * 80)
    print("LLM Hook Analysis Framework - Basic Usage Example")
    print("=" * 80)
    print()

    # Create model
    print("Creating model...")
    model = SimpleTransformer(d_model=256, nhead=4, num_layers=2)
    model.train()

    # Create hook manager
    print("Setting up hooks...")
    manager = HookManager(name="example_manager")

    # Register hooks
    manager.register(ForwardHook(layer_pattern="transformer.*", compute_stats=True))
    manager.register(BackwardHook(layer_pattern="transformer.*", detect_vanishing=True))
    manager.register(ActivationMonitor(layer_pattern="*fc*", track_distribution=True))

    # Apply hooks to model
    manager.apply_to_model(model)

    # Create dummy data
    print("Running forward and backward pass...")
    batch_size = 4
    seq_len = 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    target = torch.randint(0, 1000, (batch_size,))

    # Forward pass
    output = model(input_ids)

    # Backward pass
    loss = nn.CrossEntropyLoss()(output, target)
    loss.backward()

    print(f"Loss: {loss.item():.4f}")
    print()

    # Get results
    print("Collecting results...")
    results = manager.get_results()
    print(f"Collected {len(results)} hook results")
    print()

    # Generate report
    print("Generating analysis report...")
    report_gen = ReportGenerator(results)
    report = report_gen.generate_text_report()
    print(report)

    # Save report
    report_gen.save_report("analysis_report.txt", format='text')
    report_gen.save_report("analysis_report.json", format='json')

    # Visualize results
    print("\nGenerating visualizations...")
    manager.visualize(results, output_dir="./visualizations")

    # Cleanup
    print("\nCleaning up...")
    manager.teardown()

    print("Done!")


if __name__ == "__main__":
    main()
