"""
Memory abstraction for AI Tutor – simple in-memory store for session data.
"""
from typing import Any, Dict

class MemoryStore:
    """Simple in-memory key-value store for session context."""
    def __init__(self):
        self._store: Dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value

    def delete(self, key: str) -> None:
        if key in self._store:
            del self._store[key]

    def clear(self) -> None:
        self._store.clear() 