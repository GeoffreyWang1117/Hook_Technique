"""
Combined forward and backward hook for a specific layer.
"""

import torch.nn as nn
from typing import Optional
from llm_hooks.core.base_hook import BaseHook
from llm_hooks.pytorch.forward_hook import ForwardHook
from llm_hooks.pytorch.backward_hook import BackwardHook


class LayerHook(BaseHook):
    """
    Combined hook that captures both forward and backward information for a layer.
    """

    def __init__(
        self,
        layer_name: str,
        capture_forward: bool = True,
        capture_backward: bool = True,
        compute_stats: bool = True,
        name: Optional[str] = None,
    ):
        """
        Initialize the layer hook.

        Args:
            layer_name: Name of the layer to hook.
            capture_forward: Whether to capture forward pass.
            capture_backward: Whether to capture backward pass.
            compute_stats: Whether to compute statistics.
            name: Optional name for the hook.
        """
        super().__init__(name=name or f"LayerHook_{layer_name}")
        self.layer_name = layer_name
        self.capture_forward = capture_forward
        self.capture_backward = capture_backward
        self.compute_stats = compute_stats

        # Sub-hooks
        self.forward_hook: Optional[ForwardHook] = None
        self.backward_hook: Optional[BackwardHook] = None

    def setup(self, model: nn.Module) -> None:
        """Setup hooks on the model."""
        if self.capture_forward:
            self.forward_hook = ForwardHook(
                layer_name=self.layer_name,
                compute_stats=self.compute_stats,
            )
            self.forward_hook.setup(model)

        if self.capture_backward:
            self.backward_hook = BackwardHook(
                layer_name=self.layer_name,
                compute_stats=self.compute_stats,
            )
            self.backward_hook.setup(model)

    def teardown(self) -> None:
        """Remove all hooks."""
        if self.forward_hook:
            self.forward_hook.teardown()

        if self.backward_hook:
            self.backward_hook.teardown()

    def get_results(self):
        """Get results from both forward and backward hooks."""
        results = []

        if self.forward_hook:
            results.extend(self.forward_hook.get_results())

        if self.backward_hook:
            results.extend(self.backward_hook.get_results())

        return results

    def clear_results(self) -> None:
        """Clear results from both hooks."""
        if self.forward_hook:
            self.forward_hook.clear_results()

        if self.backward_hook:
            self.backward_hook.clear_results()
