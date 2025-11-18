"""
Example: Attention pattern analysis for transformer models.
"""

import torch
import torch.nn as nn
from llm_hooks import HookManager
from llm_hooks.attention import AttentionMonitor, KVCacheMonitor


class AttentionLayer(nn.Module):
    """Simple multi-head attention layer."""
    def __init__(self, d_model=512, nhead=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # Self-attention with residual
        attn_out, attn_weights = self.attention(x, x, x, need_weights=True)
        x = self.norm(x + attn_out)
        return x, attn_weights


def main():
    print("Attention Analysis Example")
    print("=" * 80)
    print()

    # Create model
    model = AttentionLayer(d_model=256, nhead=4)

    # Setup hooks
    manager = HookManager()
    attention_hook = AttentionMonitor(
        layer_pattern="*attention*",
        record_scores=True,
        compute_entropy=True,
        compute_sparsity=True,
    )
    manager.register(attention_hook)
    manager.apply_to_model(model)

    # Run inference
    batch_size = 2
    seq_len = 16
    d_model = 256

    x = torch.randn(batch_size, seq_len, d_model)

    print("Running attention forward pass...")
    with torch.no_grad():
        output, weights = model(x)

    # Get attention statistics
    print("\nAttention Statistics:")
    summary = attention_hook.get_attention_summary()

    print(f"Total samples: {summary.get('total_samples', 0)}")
    print(f"Layers monitored: {summary.get('layers_monitored', [])}")

    if 'entropy' in summary:
        print(f"\nEntropy Statistics:")
        print(f"  Mean: {summary['entropy']['mean']:.4f}")
        print(f"  Std:  {summary['entropy']['std']:.4f}")
        print(f"  Min:  {summary['entropy']['min']:.4f}")
        print(f"  Max:  {summary['entropy']['max']:.4f}")

    if 'sparsity' in summary:
        print(f"\nSparsity Statistics:")
        print(f"  Mean: {summary['sparsity']['mean']:.4f}")
        print(f"  Std:  {summary['sparsity']['std']:.4f}")

    # Visualize
    results = manager.get_results()
    manager.visualize(results)

    manager.teardown()
    print("\nDone!")


if __name__ == "__main__":
    main()
