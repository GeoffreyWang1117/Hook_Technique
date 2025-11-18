"""
Hook manager for coordinating multiple hooks.
"""

from typing import Any, Dict, List, Optional, Type
from llm_hooks.core.base_hook import BaseHook
from llm_hooks.core.hook_result import HookResult, HookResultCollection
import logging

logger = logging.getLogger(__name__)


class HookManager:
    """
    Manager for coordinating multiple hooks on a model.

    The HookManager provides a central interface for:
    - Registering and managing multiple hooks
    - Applying hooks to models
    - Collecting and analyzing results
    - Visualizing hook data
    """

    def __init__(self, name: str = "default"):
        """
        Initialize the hook manager.

        Args:
            name: Name for this manager instance.
        """
        self.name = name
        self.hooks: List[BaseHook] = []
        self.model: Optional[Any] = None
        self._is_setup = False

    def register(self, hook: BaseHook) -> "HookManager":
        """
        Register a hook with the manager.

        Args:
            hook: The hook to register.

        Returns:
            Self for method chaining.
        """
        if hook not in self.hooks:
            self.hooks.append(hook)
            logger.info(f"Registered hook: {hook.name}")

        # If model is already attached, setup the new hook
        if self._is_setup and self.model is not None:
            hook.setup(self.model)

        return self

    def unregister(self, hook: BaseHook) -> "HookManager":
        """
        Unregister a hook from the manager.

        Args:
            hook: The hook to unregister.

        Returns:
            Self for method chaining.
        """
        if hook in self.hooks:
            hook.teardown()
            self.hooks.remove(hook)
            logger.info(f"Unregistered hook: {hook.name}")

        return self

    def apply_to_model(self, model: Any) -> "HookManager":
        """
        Apply all registered hooks to a model.

        Args:
            model: The model to apply hooks to.

        Returns:
            Self for method chaining.
        """
        self.model = model

        for hook in self.hooks:
            if hook.enabled:
                logger.info(f"Setting up hook: {hook.name}")
                hook.setup(model)

        self._is_setup = True
        logger.info(f"Applied {len(self.hooks)} hooks to model")

        return self

    def teardown(self) -> None:
        """Remove all hooks from the model."""
        for hook in self.hooks:
            hook.teardown()

        self._is_setup = False
        logger.info("Tore down all hooks")

    def enable_all(self) -> None:
        """Enable all hooks."""
        for hook in self.hooks:
            hook.enable()
        logger.info("Enabled all hooks")

    def disable_all(self) -> None:
        """Disable all hooks."""
        for hook in self.hooks:
            hook.disable()
        logger.info("Disabled all hooks")

    def clear_results(self) -> None:
        """Clear results from all hooks."""
        for hook in self.hooks:
            hook.clear_results()
        logger.info("Cleared all hook results")

    def get_results(self) -> HookResultCollection:
        """
        Get all results from all hooks.

        Returns:
            A HookResultCollection containing all results.
        """
        collection = HookResultCollection()

        for hook in self.hooks:
            for result in hook.get_results():
                collection.add(result)

        return collection

    def get_hook_by_name(self, name: str) -> Optional[BaseHook]:
        """
        Get a hook by its name.

        Args:
            name: Name of the hook to retrieve.

        Returns:
            The hook if found, None otherwise.
        """
        for hook in self.hooks:
            if hook.name == name:
                return hook
        return None

    def get_hooks_by_type(self, hook_type: Type[BaseHook]) -> List[BaseHook]:
        """
        Get all hooks of a specific type.

        Args:
            hook_type: The class type to filter by.

        Returns:
            List of hooks matching the type.
        """
        return [hook for hook in self.hooks if isinstance(hook, hook_type)]

    def summary(self) -> Dict[str, Any]:
        """
        Get a summary of the hook manager state.

        Returns:
            Dictionary containing manager summary.
        """
        results = self.get_results()

        return {
            "name": self.name,
            "num_hooks": len(self.hooks),
            "hooks": [
                {
                    "name": hook.name,
                    "type": hook.__class__.__name__,
                    "enabled": hook.enabled,
                    "num_results": len(hook.get_results()),
                }
                for hook in self.hooks
            ],
            "total_results": len(results),
            "is_setup": self._is_setup,
        }

    def visualize(self, results: Optional[HookResultCollection] = None, **kwargs) -> None:
        """
        Visualize hook results.

        Args:
            results: Optional specific results to visualize. If None, uses all results.
            **kwargs: Additional arguments passed to visualization functions.
        """
        if results is None:
            results = self.get_results()

        # Import visualization module here to avoid circular imports
        from llm_hooks.visualization import visualize_results

        visualize_results(results, **kwargs)

    def __repr__(self) -> str:
        return f"HookManager(name={self.name}, hooks={len(self.hooks)}, setup={self._is_setup})"

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - teardown hooks."""
        self.teardown()
