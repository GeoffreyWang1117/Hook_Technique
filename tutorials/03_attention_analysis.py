"""
Tutorial 3: Analyzing Attention Patterns

This tutorial teaches you how to analyze attention mechanisms.
You'll learn how to:
1. Monitor attention patterns
2. Compute attention entropy and sparsity
3. Identify attention heads with different behaviors
"""

import torch
import torch.nn as nn
from llm_hooks import HookManager
from llm_hooks.attention import AttentionMonitor
import numpy as np


print("=" * 80)
print("Tutorial 3: Analyzing Attention Patterns")
print("=" * 80)
print()

# ============================================================================
# Step 1: Create a Transformer Model
# ============================================================================
print("Step 1: Creating a transformer model...")
print()

class SimpleTransformerBlock(nn.Module):
    """A simple transformer block with multi-head attention."""
    def __init__(self, d_model=128, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model,
            num_heads,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Self-attention with residual
        attn_out, attn_weights = self.attention(x, x, x, need_weights=True)
        x = self.norm1(x + attn_out)

        # Feed-forward with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x, attn_weights

model = SimpleTransformerBlock(d_model=128, num_heads=4)
model.eval()  # Inference mode

print(f"Model created with:")
print(f"  - Embedding dimension: 128")
print(f"  - Number of heads: 4")
print(f"  - Head dimension: 32")
print()

# ============================================================================
# Step 2: Set Up Attention Monitoring
# ============================================================================
print("Step 2: Setting up attention monitoring...")
print()

manager = HookManager()

attention_monitor = AttentionMonitor(
    layer_pattern="*attention*",
    record_scores=True,        # Record raw attention scores
    record_patterns=True,      # Record attention patterns
    compute_entropy=True,      # Compute entropy (diversity)
    compute_sparsity=True,     # Compute sparsity
    max_samples=10,           # Limit samples
)

manager.register(attention_monitor)
manager.apply_to_model(model)

print("Attention monitoring enabled!")
print()

# ============================================================================
# Step 3: Run Inference with Different Inputs
# ============================================================================
print("Step 3: Running inference with different sequence lengths...")
print()

sequence_lengths = [8, 16, 32]

for seq_len in sequence_lengths:
    # Create input sequence
    batch_size = 2
    d_model = 128
    x = torch.randn(batch_size, seq_len, d_model)

    print(f"Processing sequence length: {seq_len}")

    # Forward pass
    with torch.no_grad():
        output, weights = model(x)

    print(f"  Output shape: {output.shape}")

print()

# ============================================================================
# Step 4: Analyze Attention Statistics
# ============================================================================
print("Step 4: Analyzing attention patterns...")
print()

# Get comprehensive summary
summary = attention_monitor.get_attention_summary()

print("Attention Summary:")
print(f"  Total samples: {summary['total_samples']}")
print(f"  Layers monitored: {summary['layers_monitored']}")
print()

# Entropy analysis
if 'entropy' in summary:
    entropy_stats = summary['entropy']
    print("Entropy Statistics (Higher = More Diverse):")
    print(f"  Mean: {entropy_stats['mean']:.4f}")
    print(f"  Std:  {entropy_stats['std']:.4f}")
    print(f"  Min:  {entropy_stats['min']:.4f}")
    print(f"  Max:  {entropy_stats['max']:.4f}")
    print()

    # Interpret entropy
    if entropy_stats['mean'] < 1.0:
        print("  💡 Low entropy → Focused attention")
    elif entropy_stats['mean'] > 3.0:
        print("  💡 High entropy → Diffuse attention")
    else:
        print("  💡 Medium entropy → Balanced attention")
    print()

# Sparsity analysis
if 'sparsity' in summary:
    sparsity_stats = summary['sparsity']
    print("Sparsity Statistics (Higher = More Sparse):")
    print(f"  Mean: {sparsity_stats['mean']:.4f}")
    print(f"  Std:  {sparsity_stats['std']:.4f}")
    print()

    # Interpret sparsity
    sparsity_pct = sparsity_stats['mean'] * 100
    print(f"  💡 {sparsity_pct:.1f}% of attention weights are near zero")

    if sparsity_pct > 50:
        print("  → High sparsity: Attention is selective")
    else:
        print("  → Low sparsity: Attention is distributed")
    print()

# ============================================================================
# Step 5: Per-Head Analysis
# ============================================================================
print("Step 5: Analyzing individual attention heads...")
print()

results = manager.get_results()
if results.results:
    # Get the first result for detailed analysis
    result = results.results[0]
    head_stats = result.data.get('head_stats', [])

    print(f"Analyzing {len(head_stats)} attention heads:")
    print()

    for head_stat in head_stats:
        head_id = head_stat['head_id']
        entropy = head_stat.get('entropy_mean', 0)
        sparsity = head_stat.get('sparsity', 0)
        diagonal_mean = head_stat.get('diagonal_mean', 0)

        print(f"Head {head_id}:")
        print(f"  Entropy: {entropy:.4f}")
        print(f"  Sparsity: {sparsity:.4f}")
        print(f"  Self-attention strength: {diagonal_mean:.4f}")

        # Characterize the head
        if diagonal_mean > 0.5:
            print("  🎯 Type: Self-focused (looks at current token)")
        elif sparsity > 0.7:
            print("  🎯 Type: Selective (focuses on few tokens)")
        else:
            print("  🎯 Type: Distributed (looks at many tokens)")

        print()

# ============================================================================
# Step 6: Practical Tips
# ============================================================================
print("=" * 80)
print("UNDERSTANDING ATTENTION METRICS")
print("=" * 80)
print("""
Key Metrics Explained:

1. **Entropy**:
   - Measures attention distribution diversity
   - Low entropy (0-1): Focused on few tokens
   - High entropy (>3): Distributed across many tokens
   - Use case: Identify specialized vs general heads

2. **Sparsity**:
   - Percentage of near-zero attention weights
   - High sparsity (>70%): Selective attention
   - Low sparsity (<30%): Broad attention
   - Use case: KV cache optimization

3. **Diagonal Attention**:
   - How much each token attends to itself
   - High diagonal: Self-focused heads
   - Low diagonal: Context-gathering heads
   - Use case: Understanding head roles

4. **Head Diversity**:
   - Variation between different heads
   - High diversity: Specialized heads
   - Low diversity: Redundant heads
   - Use case: Model pruning

Applications:
- Optimize KV cache based on sparsity
- Prune redundant attention heads
- Debug attention collapse
- Understand model behavior

Next Tutorial: Learn about activation analysis and dead neurons!
""")

manager.teardown()
