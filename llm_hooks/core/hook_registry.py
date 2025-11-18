"""
Hook registry for discovering and managing hook types.
"""

from typing import Dict, Type, Optional, List
from llm_hooks.core.base_hook import BaseHook
import logging

logger = logging.getLogger(__name__)


class HookRegistry:
    """
    Global registry for hook types.

    Allows hooks to be registered and discovered by name.
    """

    _registry: Dict[str, Type[BaseHook]] = {}

    @classmethod
    def register(cls, name: str, hook_class: Type[BaseHook]) -> None:
        """
        Register a hook class.

        Args:
            name: Name to register the hook under.
            hook_class: The hook class to register.
        """
        if name in cls._registry:
            logger.warning(f"Overwriting existing hook registration: {name}")

        cls._registry[name] = hook_class
        logger.info(f"Registered hook: {name} -> {hook_class.__name__}")

    @classmethod
    def unregister(cls, name: str) -> None:
        """
        Unregister a hook class.

        Args:
            name: Name of the hook to unregister.
        """
        if name in cls._registry:
            del cls._registry[name]
            logger.info(f"Unregistered hook: {name}")

    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseHook]]:
        """
        Get a hook class by name.

        Args:
            name: Name of the hook to retrieve.

        Returns:
            The hook class if found, None otherwise.
        """
        return cls._registry.get(name)

    @classmethod
    def create(cls, name: str, **kwargs) -> Optional[BaseHook]:
        """
        Create a hook instance by name.

        Args:
            name: Name of the hook to create.
            **kwargs: Arguments to pass to hook constructor.

        Returns:
            A new hook instance if the name is registered, None otherwise.
        """
        hook_class = cls.get(name)
        if hook_class is None:
            logger.error(f"Hook not found in registry: {name}")
            return None

        return hook_class(**kwargs)

    @classmethod
    def list_hooks(cls) -> List[str]:
        """
        List all registered hook names.

        Returns:
            List of registered hook names.
        """
        return list(cls._registry.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear the registry."""
        cls._registry.clear()
        logger.info("Cleared hook registry")


def register_hook(name: str):
    """
    Decorator for registering hooks.

    Usage:
        @register_hook("my_hook")
        class MyHook(BaseHook):
            ...
    """
    def decorator(hook_class: Type[BaseHook]):
        HookRegistry.register(name, hook_class)
        return hook_class
    return decorator
