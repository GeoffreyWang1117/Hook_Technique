"""
Base hook class that all hooks should inherit from.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
import time
from dataclasses import dataclass, field


@dataclass
class HookResult:
    """Container for hook execution results."""
    hook_name: str
    hook_type: str
    timestamp: float
    layer_name: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseHook(ABC):
    """
    Abstract base class for all hooks.

    All custom hooks should inherit from this class and implement the
    required abstract methods.
    """

    def __init__(self, name: Optional[str] = None, enabled: bool = True):
        """
        Initialize the base hook.

        Args:
            name: Optional name for the hook. If not provided, uses class name.
            enabled: Whether the hook is initially enabled.
        """
        self.name = name or self.__class__.__name__
        self.enabled = enabled
        self.results: List[HookResult] = []
        self._hook_handles: List[Any] = []

    @abstractmethod
    def setup(self, model: Any) -> None:
        """
        Setup the hook for a given model.

        Args:
            model: The model to attach hooks to.
        """
        pass

    @abstractmethod
    def teardown(self) -> None:
        """
        Remove all hooks and cleanup resources.
        """
        pass

    def enable(self) -> None:
        """Enable the hook."""
        self.enabled = True

    def disable(self) -> None:
        """Disable the hook."""
        self.enabled = False

    def clear_results(self) -> None:
        """Clear all collected results."""
        self.results.clear()

    def get_results(self) -> List[HookResult]:
        """
        Get all collected results.

        Returns:
            List of HookResult objects.
        """
        return self.results

    def add_result(self, result: HookResult) -> None:
        """
        Add a result to the collection.

        Args:
            result: The HookResult to add.
        """
        self.results.append(result)

    def create_result(
        self,
        hook_type: str,
        data: Dict[str, Any],
        layer_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> HookResult:
        """
        Create a HookResult with current timestamp.

        Args:
            hook_type: Type of hook (e.g., 'forward', 'backward', 'attention').
            data: Data collected by the hook.
            layer_name: Optional layer name where hook was triggered.
            metadata: Optional metadata about the hook execution.

        Returns:
            A new HookResult object.
        """
        return HookResult(
            hook_name=self.name,
            hook_type=hook_type,
            timestamp=time.time(),
            layer_name=layer_name,
            data=data,
            metadata=metadata or {}
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, enabled={self.enabled})"
