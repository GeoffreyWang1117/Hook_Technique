"""
TensorRT logger for capturing engine build and inference logs.
"""

from typing import Optional, List, Dict, Any
from llm_hooks.core.base_hook import BaseHook
from llm_hooks.core.hook_registry import register_hook
import logging
import time


@register_hook("tensorrt")
class TensorRTLogger(BaseHook):
    """
    Hook for capturing TensorRT logs and profiling data.

    This hook integrates with TensorRT to capture:
    - Engine build logs
    - Layer execution times
    - Memory usage
    - Optimization decisions
    """

    def __init__(
        self,
        log_severity: str = "WARNING",
        capture_build_logs: bool = True,
        capture_inference_logs: bool = True,
        profile_layers: bool = True,
        name: Optional[str] = None,
    ):
        """
        Initialize TensorRT logger.

        Args:
            log_severity: Minimum severity level (VERBOSE, INFO, WARNING, ERROR)
            capture_build_logs: Whether to capture engine build logs
            capture_inference_logs: Whether to capture inference logs
            profile_layers: Whether to profile individual layers
            name: Optional name for the hook
        """
        super().__init__(name=name or "TensorRTLogger")
        self.log_severity = log_severity
        self.capture_build_logs = capture_build_logs
        self.capture_inference_logs = capture_inference_logs
        self.profile_layers = profile_layers

        # Internal state
        self.build_logs: List[Dict[str, Any]] = []
        self.inference_logs: List[Dict[str, Any]] = []
        self.layer_profiles: Dict[str, List[float]] = {}

        # TensorRT logger instance
        self._trt_logger = None

    def setup(self, model: Any) -> None:
        """
        Setup TensorRT logging.

        Note: This requires the model to be a TensorRT engine or builder.
        For PyTorch models, use torch2trt or similar conversion tools first.
        """
        try:
            import tensorrt as trt

            # Create TensorRT logger
            self._trt_logger = self._create_trt_logger(trt)

            # Check if model has TensorRT components
            if hasattr(model, 'engine'):
                self._setup_engine_profiling(model.engine, trt)
            elif hasattr(model, 'builder'):
                self._setup_builder_logging(model.builder, trt)
            else:
                logging.warning(
                    "Model doesn't have TensorRT engine or builder. "
                    "TensorRT logging may not work correctly."
                )

        except ImportError:
            logging.warning(
                "TensorRT not available. Install with: pip install nvidia-tensorrt"
            )

    def teardown(self) -> None:
        """Cleanup TensorRT logging."""
        self._trt_logger = None

    def _create_trt_logger(self, trt):
        """Create a custom TensorRT logger."""

        class CustomTRTLogger(trt.ILogger):
            def __init__(self, hook_instance):
                trt.ILogger.__init__(self)
                self.hook = hook_instance

            def log(self, severity, msg):
                # Map TensorRT severity to our format
                severity_map = {
                    trt.Logger.INTERNAL_ERROR: "CRITICAL",
                    trt.Logger.ERROR: "ERROR",
                    trt.Logger.WARNING: "WARNING",
                    trt.Logger.INFO: "INFO",
                    trt.Logger.VERBOSE: "VERBOSE",
                }

                log_entry = {
                    'severity': severity_map.get(severity, "UNKNOWN"),
                    'message': msg,
                    'timestamp': time.time(),
                }

                # Store based on context
                if "build" in msg.lower() and self.hook.capture_build_logs:
                    self.hook.build_logs.append(log_entry)
                elif self.hook.capture_inference_logs:
                    self.hook.inference_logs.append(log_entry)

        return CustomTRTLogger(self)

    def _setup_engine_profiling(self, engine, trt):
        """Setup profiling for a TensorRT engine."""
        if not self.profile_layers:
            return

        # Create profiler
        try:
            context = engine.create_execution_context()

            # Enable profiling
            if hasattr(context, 'profiler'):
                profiler = self._create_profiler(trt)
                context.profiler = profiler

        except Exception as e:
            logging.warning(f"Failed to setup engine profiling: {e}")

    def _setup_builder_logging(self, builder, trt):
        """Setup logging for TensorRT builder."""
        if not self.capture_build_logs:
            return

        try:
            # Attach logger to builder
            builder.logger = self._trt_logger

        except Exception as e:
            logging.warning(f"Failed to setup builder logging: {e}")

    def _create_profiler(self, trt):
        """Create a custom TensorRT profiler."""

        class CustomProfiler(trt.IProfiler):
            def __init__(self, hook_instance):
                trt.IProfiler.__init__(self)
                self.hook = hook_instance

            def report_layer_time(self, layer_name, ms):
                if layer_name not in self.hook.layer_profiles:
                    self.hook.layer_profiles[layer_name] = []
                self.hook.layer_profiles[layer_name].append(ms)

        return CustomProfiler(self)

    def get_build_summary(self) -> Dict[str, Any]:
        """Get summary of build logs."""
        if not self.build_logs:
            return {}

        # Count by severity
        severity_counts = {}
        for log in self.build_logs:
            severity = log['severity']
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        return {
            'total_logs': len(self.build_logs),
            'severity_counts': severity_counts,
            'logs': self.build_logs,
        }

    def get_inference_summary(self) -> Dict[str, Any]:
        """Get summary of inference logs."""
        if not self.inference_logs:
            return {}

        return {
            'total_logs': len(self.inference_logs),
            'logs': self.inference_logs,
        }

    def get_layer_profiling(self) -> Dict[str, Any]:
        """Get layer profiling statistics."""
        if not self.layer_profiles:
            return {}

        stats = {}
        for layer_name, times in self.layer_profiles.items():
            if times:
                stats[layer_name] = {
                    'mean_ms': sum(times) / len(times),
                    'min_ms': min(times),
                    'max_ms': max(times),
                    'total_ms': sum(times),
                    'count': len(times),
                }

        # Sort by total time
        sorted_stats = dict(
            sorted(stats.items(), key=lambda x: x[1]['total_ms'], reverse=True)
        )

        return {
            'layers': sorted_stats,
            'total_layers': len(sorted_stats),
        }

    def export_profile(self, filepath: str):
        """Export profiling data to JSON file."""
        import json

        data = {
            'build_summary': self.get_build_summary(),
            'inference_summary': self.get_inference_summary(),
            'layer_profiling': self.get_layer_profiling(),
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Profiling data exported to {filepath}")
