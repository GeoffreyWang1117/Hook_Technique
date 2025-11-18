"""
Visualization utilities for hook results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional, List, Dict, Any
from llm_hooks.core.hook_result import HookResultCollection


def visualize_results(results: HookResultCollection, output_dir: str = "./visualizations"):
    """
    Automatically visualize hook results based on their types.

    Args:
        results: Collection of hook results.
        output_dir: Directory to save visualizations.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Group results by type
    attention_results = results.filter_by_type('attention')
    gradient_results = results.filter_by_type('gradient')
    activation_results = results.filter_by_type('activation')

    # Visualize each type
    if attention_results:
        plot_attention(attention_results, save_path=f"{output_dir}/attention.png")

    if gradient_results:
        plot_gradients(gradient_results, save_path=f"{output_dir}/gradients.png")

    if activation_results:
        plot_activations(activation_results, save_path=f"{output_dir}/activations.png")

    print(f"Visualizations saved to {output_dir}")


def plot_attention(results: List, save_path: Optional[str] = None):
    """
    Visualize attention patterns.

    Args:
        results: List of attention results.
        save_path: Optional path to save the figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Attention Analysis', fontsize=16)

    # Extract attention statistics
    entropies = []
    sparsities = []
    layer_names = []

    for result in results:
        data = result.data
        layer_names.append(data.get('layer_name', 'unknown'))

        for head_stat in data.get('head_stats', []):
            if 'entropy_mean' in head_stat:
                entropies.append(head_stat['entropy_mean'])
            if 'sparsity' in head_stat:
                sparsities.append(head_stat['sparsity'])

    # Plot entropy distribution
    if entropies:
        axes[0, 0].hist(entropies, bins=30, alpha=0.7, color='blue')
        axes[0, 0].set_xlabel('Entropy')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Attention Entropy Distribution')
        axes[0, 0].axvline(np.mean(entropies), color='red', linestyle='--', label=f'Mean: {np.mean(entropies):.2f}')
        axes[0, 0].legend()

    # Plot sparsity distribution
    if sparsities:
        axes[0, 1].hist(sparsities, bins=30, alpha=0.7, color='green')
        axes[0, 1].set_xlabel('Sparsity')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Attention Sparsity Distribution')
        axes[0, 1].axvline(np.mean(sparsities), color='red', linestyle='--', label=f'Mean: {np.mean(sparsities):.2f}')
        axes[0, 1].legend()

    # Plot attention scores heatmap (if available)
    if results and 'scores_sample' in results[0].data:
        scores = np.array(results[0].data['scores_sample'][0])  # First head
        sns.heatmap(scores, ax=axes[1, 0], cmap='viridis', cbar=True)
        axes[1, 0].set_title('Sample Attention Pattern (Head 0)')
        axes[1, 0].set_xlabel('Key Position')
        axes[1, 0].set_ylabel('Query Position')

    # Summary statistics
    summary_text = f"""
    Attention Summary:
    - Total samples: {len(results)}
    - Layers: {len(set(layer_names))}
    - Avg Entropy: {np.mean(entropies):.3f} ± {np.std(entropies):.3f}
    - Avg Sparsity: {np.mean(sparsities):.3f} ± {np.std(sparsities):.3f}
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center')
    axes[1, 1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Attention plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_gradients(results: List, save_path: Optional[str] = None):
    """
    Visualize gradient flow.

    Args:
        results: List of gradient results.
        save_path: Optional path to save the figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Gradient Flow Analysis', fontsize=16)

    # Extract gradient statistics
    layer_names = []
    gradient_norms = []
    gradient_means = []

    for result in results:
        data = result.data
        layer_names.append(data.get('param_name', 'unknown'))

        if 'stats' in data:
            gradient_norms.append(data['stats'].get('norm_l2', 0))
            gradient_means.append(data['stats'].get('abs_mean', 0))

    # Plot gradient norms
    if gradient_norms:
        axes[0, 0].plot(gradient_norms, marker='o', linestyle='-', alpha=0.7)
        axes[0, 0].set_xlabel('Layer Index')
        axes[0, 0].set_ylabel('Gradient Norm (L2)')
        axes[0, 0].set_title('Gradient Magnitude Across Layers')
        axes[0, 0].set_yscale('log')
        axes[0, 0].grid(True, alpha=0.3)

    # Plot gradient means
    if gradient_means:
        axes[0, 1].plot(gradient_means, marker='s', linestyle='-', alpha=0.7, color='orange')
        axes[0, 1].set_xlabel('Layer Index')
        axes[0, 1].set_ylabel('Mean Absolute Gradient')
        axes[0, 1].set_title('Average Gradient Magnitude')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)

    # Histogram of gradient norms
    if gradient_norms:
        axes[1, 0].hist(gradient_norms, bins=30, alpha=0.7, color='purple')
        axes[1, 0].set_xlabel('Gradient Norm')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Gradient Norms')
        axes[1, 0].set_xscale('log')

    # Summary
    num_issues = sum(1 for r in results if r.data.get('issues', []))
    summary_text = f"""
    Gradient Summary:
    - Total layers: {len(results)}
    - Layers with issues: {num_issues}
    - Avg norm: {np.mean(gradient_norms):.3e}
    - Max norm: {np.max(gradient_norms):.3e}
    - Min norm: {np.min(gradient_norms):.3e}
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center')
    axes[1, 1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Gradient plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_activations(results: List, save_path: Optional[str] = None):
    """
    Visualize activation statistics.

    Args:
        results: List of activation results.
        save_path: Optional path to save the figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Activation Analysis', fontsize=16)

    # Extract statistics
    layer_names = []
    sparsity_ratios = []
    dead_neuron_ratios = []

    for result in results:
        data = result.data
        layer_names.append(result.layer_name or 'unknown')

        if 'sparsity' in data:
            sparsity_ratios.append(data['sparsity'].get('zero_ratio', 0))

        if 'dead_neurons' in data:
            dead_neuron_ratios.append(data['dead_neurons'].get('ratio', 0))

    # Plot sparsity across layers
    if sparsity_ratios:
        axes[0, 0].bar(range(len(sparsity_ratios)), sparsity_ratios, alpha=0.7)
        axes[0, 0].set_xlabel('Layer Index')
        axes[0, 0].set_ylabel('Sparsity Ratio')
        axes[0, 0].set_title('Activation Sparsity Across Layers')
        axes[0, 0].grid(True, alpha=0.3)

    # Plot dead neurons
    if dead_neuron_ratios:
        axes[0, 1].bar(range(len(dead_neuron_ratios)), dead_neuron_ratios, alpha=0.7, color='red')
        axes[0, 1].set_xlabel('Layer Index')
        axes[0, 1].set_ylabel('Dead Neuron Ratio')
        axes[0, 1].set_title('Dead Neurons Across Layers')
        axes[0, 1].grid(True, alpha=0.3)

    # Distribution histogram
    if sparsity_ratios:
        axes[1, 0].hist(sparsity_ratios, bins=20, alpha=0.7, color='green')
        axes[1, 0].set_xlabel('Sparsity Ratio')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Sparsity Distribution')

    # Summary
    summary_text = f"""
    Activation Summary:
    - Total layers: {len(results)}
    - Avg sparsity: {np.mean(sparsity_ratios):.3f}
    - Avg dead neurons: {np.mean(dead_neuron_ratios):.3f}
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center')
    axes[1, 1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Activation plot saved to {save_path}")
    else:
        plt.show()

    plt.close()
