"""
Use Case: LLM Inference Optimization

This example demonstrates a complete workflow for optimizing a small LLM:
1. Profile the model to identify bottlenecks
2. Analyze attention patterns for KV cache optimization
3. Detect dead neurons for potential pruning
4. Generate optimization recommendations
"""

import torch
import torch.nn as nn
from llm_hooks import HookManager
from llm_hooks.pytorch import ForwardHook
from llm_hooks.attention import AttentionMonitor, KVCacheMonitor
from llm_hooks.activations import ActivationMonitor
from llm_hooks.cuda import CUDAProfiler
from llm_hooks.analysis import BottleneckAnalyzer, ReportGenerator


print("=" * 80)
print("Use Case: LLM Inference Optimization")
print("=" * 80)
print()

# ============================================================================
# Step 1: Define a Small LLM Model
# ============================================================================
print("Step 1: Creating a small language model...")
print()

class SmallLLM(nn.Module):
    """A small transformer-based language model."""
    def __init__(self, vocab_size=10000, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Embedding(512, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape

        # Embeddings
        x = self.embedding(input_ids)

        # Positional encoding
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = x + self.pos_encoding(positions)

        # Transformer
        x = self.transformer(x)

        # Output projection
        logits = self.fc_out(x)

        return logits

# Create model
model = SmallLLM(vocab_size=10000, d_model=512, nhead=8, num_layers=6)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"Model created:")
print(f"  Vocabulary: 10,000")
print(f"  Hidden size: 512")
print(f"  Attention heads: 8")
print(f"  Layers: 6")
print(f"  Total parameters: {total_params:,}")
print()

# ============================================================================
# Step 2: Set Up Comprehensive Monitoring
# ============================================================================
print("Step 2: Setting up comprehensive monitoring...")
print()

manager = HookManager(name="llm_optimization")

# 1. Forward hooks for activation analysis
forward_hook = ForwardHook(
    layer_pattern="transformer.layers.*",
    compute_stats=True,
)

# 2. Attention monitoring
attention_monitor = AttentionMonitor(
    layer_pattern="*attn*",
    record_scores=True,
    compute_entropy=True,
    compute_sparsity=True,
    max_samples=20,
)

# 3. KV cache monitoring
kv_monitor = KVCacheMonitor(
    layer_pattern="*attn*",
    track_size=True,
    track_memory=True,
)

# 4. Activation monitoring for dead neurons
activation_monitor = ActivationMonitor(
    layer_pattern="transformer.layers.*",
    dead_neuron_threshold=0.01,
    track_distribution=True,
)

# 5. CUDA profiling (if available)
if torch.cuda.is_available():
    cuda_profiler = CUDAProfiler(
        with_stack=True,
        profile_memory=True,
    )
    manager.register(cuda_profiler)

# Register all hooks
manager.register(forward_hook)
manager.register(attention_monitor)
manager.register(kv_monitor)
manager.register(activation_monitor)

manager.apply_to_model(model)

print("Registered hooks:")
for hook in manager.hooks:
    print(f"  - {hook.name}")
print()

# ============================================================================
# Step 3: Run Inference Benchmarks
# ============================================================================
print("Step 3: Running inference benchmarks...")
print()

# Test with different sequence lengths
sequence_lengths = [32, 64, 128, 256]

print("Running inference with different sequence lengths...")
for seq_len in sequence_lengths:
    batch_size = 4
    input_ids = torch.randint(0, 10000, (batch_size, seq_len))

    print(f"  Sequence length: {seq_len}", end="")

    # Inference
    with torch.no_grad():
        if torch.cuda.is_available():
            cuda_profiler.start_profiling()

        output = model(input_ids)

        if torch.cuda.is_available():
            cuda_profiler.stop_profiling()

    print(f" → Output shape: {output.shape}")

print()

# ============================================================================
# Step 4: Analyze Results
# ============================================================================
print("Step 4: Analyzing optimization opportunities...")
print()

results = manager.get_results()
analyzer = BottleneckAnalyzer(results)

# Bottleneck analysis
print("=" * 60)
print("BOTTLENECK ANALYSIS")
print("=" * 60)

bottleneck_summary = analyzer.generate_summary()

# 1. Performance bottlenecks
slow_layers = bottleneck_summary.get('slow_layers', [])
if slow_layers:
    print(f"\n⚠️  Performance Bottlenecks ({len(slow_layers)} layers):")
    for layer in slow_layers[:5]:
        print(f"  - {layer}")
else:
    print("\n✅ No significant performance bottlenecks detected")

# 2. Dead neurons
dead_neurons = bottleneck_summary.get('dead_neurons', {})
if dead_neurons:
    total_dead = sum(dead_neurons.values())
    print(f"\n⚠️  Dead Neurons Detected:")
    print(f"  Total dead neurons: {total_dead}")
    print(f"  Affected layers: {len(dead_neurons)}")

    print("\n  Top affected layers:")
    sorted_dead = sorted(dead_neurons.items(), key=lambda x: x[1], reverse=True)
    for layer, count in sorted_dead[:3]:
        print(f"    - {layer}: {count} dead neurons")
else:
    print("\n✅ No dead neurons detected")

# 3. Memory usage
mem_analysis = bottleneck_summary.get('memory_analysis', {})
if mem_analysis:
    print(f"\n📊 Memory Analysis:")
    print(f"  Total memory: {mem_analysis.get('total_memory_mb', 0):.2f} MB")
    print(f"  Peak memory: {mem_analysis.get('max_memory_mb', 0):.2f} MB")
    print(f"  Avg cache size: {mem_analysis.get('avg_cache_size', 0):.0f} tokens")

print()

# ============================================================================
# Step 5: Attention Analysis
# ============================================================================
print("=" * 60)
print("ATTENTION ANALYSIS")
print("=" * 60)

attention_summary = attention_monitor.get_attention_summary()

if attention_summary:
    print(f"\nTotal samples: {attention_summary.get('total_samples', 0)}")

    # Entropy analysis
    if 'entropy' in attention_summary:
        entropy = attention_summary['entropy']
        print(f"\nAttention Entropy:")
        print(f"  Mean: {entropy['mean']:.3f}")
        print(f"  Std: {entropy['std']:.3f}")

        if entropy['mean'] > 3.0:
            print("  💡 High entropy → Consider multi-query attention")
        elif entropy['mean'] < 1.0:
            print("  💡 Low entropy → Good for KV cache pruning")

    # Sparsity analysis
    if 'sparsity' in attention_summary:
        sparsity = attention_summary['sparsity']
        sparsity_pct = sparsity['mean'] * 100
        print(f"\nAttention Sparsity:")
        print(f"  Mean: {sparsity_pct:.1f}%")

        if sparsity_pct > 70:
            print("  💡 High sparsity → Optimize KV cache storage")
        elif sparsity_pct > 50:
            print("  💡 Moderate sparsity → Use sparse attention")

print()

# ============================================================================
# Step 6: KV Cache Analysis
# ============================================================================
print("=" * 60)
print("KV CACHE ANALYSIS")
print("=" * 60)

kv_summary = kv_monitor.get_cache_summary()

if kv_summary:
    print(f"\nLayers monitored: {len(kv_summary.get('layers_monitored', []))}")

    if 'size_stats' in kv_summary:
        size_stats = kv_summary['size_stats']
        print(f"\nCache Size Statistics:")
        print(f"  Mean: {size_stats['mean']:.1f} tokens")
        print(f"  Max: {size_stats['max']} tokens")
        print(f"  Std: {size_stats['std']:.1f}")

    if 'memory_stats' in kv_summary:
        mem_stats = kv_summary['memory_stats']
        print(f"\nMemory Statistics:")
        print(f"  Total: {mem_stats['total_mb']:.2f} MB")
        print(f"  Mean per layer: {mem_stats['mean_mb']:.2f} MB")
        print(f"  Peak: {mem_stats['max_mb']:.2f} MB")

print()

# ============================================================================
# Step 7: Generate Optimization Report
# ============================================================================
print("=" * 60)
print("OPTIMIZATION RECOMMENDATIONS")
print("=" * 60)

report_gen = ReportGenerator(results)

# Save detailed report
report_gen.save_report("llm_optimization_report.txt", format='text')
report_gen.save_report("llm_optimization_report.json", format='json')

print("\n📝 Detailed reports saved:")
print("  - llm_optimization_report.txt")
print("  - llm_optimization_report.json")
print()

# Generate recommendations
print("Optimization Recommendations:\n")

recommendations = []

# Based on dead neurons
if dead_neurons and sum(dead_neurons.values()) > 100:
    recommendations.append({
        'priority': 'HIGH',
        'category': 'Model Pruning',
        'action': f'Remove {sum(dead_neurons.values())} dead neurons',
        'expected_benefit': 'Reduce model size by 10-20%',
    })

# Based on attention sparsity
if attention_summary and 'sparsity' in attention_summary:
    if attention_summary['sparsity']['mean'] > 0.7:
        recommendations.append({
            'priority': 'MEDIUM',
            'category': 'KV Cache',
            'action': 'Implement sparse KV cache',
            'expected_benefit': 'Reduce memory usage by 30-50%',
        })

# Based on memory usage
if mem_analysis and mem_analysis.get('total_memory_mb', 0) > 1000:
    recommendations.append({
        'priority': 'HIGH',
        'category': 'Memory',
        'action': 'Optimize KV cache with quantization',
        'expected_benefit': 'Reduce memory by 40-60%',
        })

# Based on attention entropy
if attention_summary and 'entropy' in attention_summary:
    if attention_summary['entropy']['mean'] > 3.5:
        recommendations.append({
            'priority': 'MEDIUM',
            'category': 'Architecture',
            'action': 'Consider Multi-Query Attention (MQA)',
            'expected_benefit': 'Faster inference, less memory',
        })

# Display recommendations
if recommendations:
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. [{rec['priority']}] {rec['category']}")
        print(f"   Action: {rec['action']}")
        print(f"   Expected: {rec['expected_benefit']}")
        print()
else:
    print("✅ Model is already well-optimized!")
    print()

# ============================================================================
# Step 8: Visualize Results
# ============================================================================
print("=" * 60)
print("VISUALIZATION")
print("=" * 60)

print("\nGenerating visualizations...")
manager.visualize(results, output_dir="./llm_optimization_viz")
print("✅ Visualizations saved to: ./llm_optimization_viz/")
print()

# ============================================================================
# Summary
# ============================================================================
print("=" * 80)
print("OPTIMIZATION WORKFLOW COMPLETE")
print("=" * 80)
print("""
Next Steps:

1. Review Reports:
   - Check text report for detailed analysis
   - Examine JSON report for programmatic access
   - View visualizations for insights

2. Implement Optimizations:
   - Apply recommended pruning
   - Optimize KV cache based on sparsity
   - Consider architectural changes

3. Validate:
   - Benchmark inference speed
   - Measure memory usage
   - Test model accuracy

4. Iterate:
   - Re-run analysis after changes
   - Compare before/after metrics
   - Fine-tune as needed

Tools Used:
✓ Forward hooks for activation analysis
✓ Attention monitoring for pattern analysis
✓ KV cache monitoring for memory optimization
✓ Dead neuron detection for pruning
✓ CUDA profiling for performance analysis
✓ Automated report generation
✓ Visualization tools

For more information, see docs/optimization_guide.md
""")

manager.teardown()
