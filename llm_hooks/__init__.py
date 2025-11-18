"""
LLM Hook Analysis Framework

A pluggable framework for analyzing LLM inference and training through dynamic hooks.
"""

__version__ = "0.1.0"

from llm_hooks.core.hook_manager import HookManager
from llm_hooks.core.base_hook import BaseHook
from llm_hooks.core.hook_registry import HookRegistry

__all__ = [
    "HookManager",
    "BaseHook",
    "HookRegistry",
    "__version__",
]
