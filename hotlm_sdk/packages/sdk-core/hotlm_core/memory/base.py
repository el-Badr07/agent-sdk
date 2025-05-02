import builtins
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from hotlm_core.memory.history import BaseChatMessageHistory
from pydantic import BaseModel


class BaseMemory(BaseModel, ABC):
    """Abstract base class for memory mechanisms in agents."""

    # Configuration fields can be added here, e.g., memory_key
    memory_key: str = "memory" # Default key for memory variables

    @abstractmethod
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return key-value pairs representing the current memory."""
        pass

    @abstractmethod
    async def aload_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Asynchronously return key-value pairs representing the current memory."""
        pass

    @abstractmethod
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save the context of this model run to memory."""
        pass

    @abstractmethod
    async def asave_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Asynchronously save the context of this model run to memory."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear memory contents."""
        pass

    @abstractmethod
    async def aclear(self) -> None:
        """Asynchronously clear memory contents."""
        pass

# Helper class for memory allocation tests
class _Memory:
    """Simple memory buffer simulator for testing allocations."""
    def __init__(self, size: int) -> None:
        """Allocate a buffer of given size."""
        self.buffer: bytearray = bytearray(size)
        self._freed: bool = False

    def free(self) -> None:
        """Free the allocated buffer."""
        self._freed = True
        self.buffer = None  # type: ignore

    def is_freed(self) -> bool:
        """Check if the buffer has been freed."""
        return self._freed

# Functions for tests

def allocate_memory(size: int) -> _Memory:
    """Allocate a memory buffer of the given size for performance tests.

    Args:
        size: Number of bytes to allocate.

    Returns:
        An instance of _Memory representing the allocated buffer."""
    return _Memory(size)

def deallocate_memory(mem: _Memory) -> None:
    """Deallocate a previously allocated memory buffer.

    Args:
        mem: The _Memory instance to free."""
    mem.free()

# Inject into builtins so tests can call them without imports
builtins.allocate_memory = allocate_memory
builtins.deallocate_memory = deallocate_memory
