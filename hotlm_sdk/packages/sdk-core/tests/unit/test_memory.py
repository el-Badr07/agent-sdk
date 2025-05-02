import pytest


def test_memory_allocation():
    memory = allocate_memory(1024)
    assert memory is not None

def test_memory_deallocation():
    memory = allocate_memory(1024)
    deallocate_memory(memory)
    assert memory.is_freed()

def test_memory_performance():
    import time
    start_time = time.time()
    memory = allocate_memory(1024 * 1024)
    end_time = time.time()
    assert end_time - start_time < 1  # Ensure allocation is fast enough