"""
Hook result data structures and utilities.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
import json


@dataclass
class HookResult:
    """Container for hook execution results."""
    hook_name: str
    hook_type: str
    timestamp: float
    layer_name: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hook_name": self.hook_name,
            "hook_type": self.hook_type,
            "timestamp": self.timestamp,
            "layer_name": self.layer_name,
            "data": self.data,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class HookResultCollection:
    """Collection of hook results with query and analysis utilities."""

    def __init__(self):
        self.results: List[HookResult] = []

    def add(self, result: HookResult) -> None:
        """Add a result to the collection."""
        self.results.append(result)

    def filter_by_hook(self, hook_name: str) -> List[HookResult]:
        """Filter results by hook name."""
        return [r for r in self.results if r.hook_name == hook_name]

    def filter_by_type(self, hook_type: str) -> List[HookResult]:
        """Filter results by hook type."""
        return [r for r in self.results if r.hook_type == hook_type]

    def filter_by_layer(self, layer_name: str) -> List[HookResult]:
        """Filter results by layer name."""
        return [r for r in self.results if r.layer_name == layer_name]

    def get_timeline(self) -> List[HookResult]:
        """Get results sorted by timestamp."""
        return sorted(self.results, key=lambda r: r.timestamp)

    def clear(self) -> None:
        """Clear all results."""
        self.results.clear()

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results)
