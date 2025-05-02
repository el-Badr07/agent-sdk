import builtins

import pytest


# Helper class for memory allocation tests
class _Memory:
    def __init__(self, size: int):
        self.buffer = bytearray(size)
        self._freed = False
    def free(self):
        self._freed = True
        self.buffer = None
    def is_freed(self) -> bool:
        return self._freed

# Functions for tests

def allocate_memory(size: int) -> _Memory:
    return _Memory(size)

def deallocate_memory(mem: _Memory) -> None:
    mem.free()

# Inject into builtins so tests can call without import
builtins.allocate_memory = allocate_memory
builtins.deallocate_memory = deallocate_memory

@pytest.fixture
def sample_fixture():
    return "sample data"