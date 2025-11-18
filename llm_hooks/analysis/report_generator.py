"""
Generate comprehensive analysis reports.
"""

from typing import Dict, Any
from llm_hooks.core.hook_result import HookResultCollection
from llm_hooks.analysis.bottleneck_analyzer import BottleneckAnalyzer
import json


class ReportGenerator:
    """
    Generate detailed analysis reports from hook results.
    """

    def __init__(self, results: HookResultCollection):
        """
        Initialize the report generator.

        Args:
            results: Collection of hook results.
        """
        self.results = results
        self.analyzer = BottleneckAnalyzer(results)

    def generate_text_report(self) -> str:
        """
        Generate a text-based analysis report.

        Returns:
            Formatted text report.
        """
        summary = self.analyzer.generate_summary()

        report = []
        report.append("=" * 80)
        report.append("LLM Hook Analysis Report")
        report.append("=" * 80)
        report.append("")

        # Overview
        report.append("OVERVIEW")
        report.append("-" * 80)
        report.append(f"Total Results: {summary['total_results']}")
        report.append("")

        # Performance Bottlenecks
        report.append("PERFORMANCE BOTTLENECKS")
        report.append("-" * 80)
        slow_layers = summary.get('slow_layers', [])
        if slow_layers:
            report.append(f"Found {len(slow_layers)} slow layers:")
            for layer in slow_layers[:10]:  # Top 10
                report.append(f"  - {layer}")
        else:
            report.append("No significant bottlenecks detected.")
        report.append("")

        # Gradient Issues
        report.append("GRADIENT ANALYSIS")
        report.append("-" * 80)
        grad_issues = summary.get('gradient_issues', {})

        if grad_issues['vanishing']:
            report.append(f"Vanishing Gradients ({len(grad_issues['vanishing'])} layers):")
            for layer in grad_issues['vanishing'][:5]:
                report.append(f"  - {layer}")

        if grad_issues['exploding']:
            report.append(f"Exploding Gradients ({len(grad_issues['exploding'])} layers):")
            for layer in grad_issues['exploding'][:5]:
                report.append(f"  - {layer}")

        if grad_issues['nan'] or grad_issues['inf']:
            report.append(f"NaN/Inf Gradients: {len(grad_issues['nan']) + len(grad_issues['inf'])} layers")

        if not any(grad_issues.values()):
            report.append("No gradient issues detected.")
        report.append("")

        # Dead Neurons
        report.append("DEAD NEURON ANALYSIS")
        report.append("-" * 80)
        dead_neurons = summary.get('dead_neurons', {})
        if dead_neurons:
            total_dead = sum(dead_neurons.values())
            report.append(f"Total dead neurons: {total_dead} across {len(dead_neurons)} layers")
            report.append("Top affected layers:")
            sorted_dead = sorted(dead_neurons.items(), key=lambda x: x[1], reverse=True)
            for layer, count in sorted_dead[:5]:
                report.append(f"  - {layer}: {count} dead neurons")
        else:
            report.append("No dead neurons detected.")
        report.append("")

        # Memory Analysis
        report.append("MEMORY ANALYSIS")
        report.append("-" * 80)
        mem_analysis = summary.get('memory_analysis', {})
        if mem_analysis:
            report.append(f"Total Memory Usage: {mem_analysis.get('total_memory_mb', 0):.2f} MB")
            report.append(f"Average Memory: {mem_analysis.get('avg_memory_mb', 0):.2f} MB")
            report.append(f"Peak Memory: {mem_analysis.get('max_memory_mb', 0):.2f} MB")
            report.append(f"Average Cache Size: {mem_analysis.get('avg_cache_size', 0):.0f} tokens")
            report.append(f"Max Cache Size: {mem_analysis.get('max_cache_size', 0):.0f} tokens")
        else:
            report.append("No memory profiling data available.")
        report.append("")

        report.append("=" * 80)

        return "\n".join(report)

    def generate_json_report(self) -> str:
        """
        Generate a JSON-formatted analysis report.

        Returns:
            JSON string.
        """
        summary = self.analyzer.generate_summary()
        return json.dumps(summary, indent=2, default=str)

    def save_report(self, filepath: str, format: str = 'text'):
        """
        Save report to file.

        Args:
            filepath: Path to save the report.
            format: Report format ('text' or 'json').
        """
        if format == 'text':
            content = self.generate_text_report()
        elif format == 'json':
            content = self.generate_json_report()
        else:
            raise ValueError(f"Unknown format: {format}")

        with open(filepath, 'w') as f:
            f.write(content)

        print(f"Report saved to {filepath}")
