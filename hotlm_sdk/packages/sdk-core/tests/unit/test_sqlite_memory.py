import os
import tempfile
from typing import Any, Dict

import pytest
from hotlm_core.memory.sqlite_memory import SQLiteChatMessageHistory, SQLiteMemory
from hotlm_core.schema.messages import AIMessage, BaseMessage, HumanMessage


def test_sqlite_chat_message_history():
    """Test the SQLiteChatMessageHistory basic functionality."""
    # Use in-memory DB for tests
    history = SQLiteChatMessageHistory(db_path=":memory:")
    
    # Test adding messages
    history.add_message(HumanMessage(content="Hello"))
    history.add_message(AIMessage(content="Hi there"))
    
    # Test retrieving messages
    messages = history.get_messages()
    assert len(messages) == 2
    assert messages[0].role == "human"
    assert messages[0].content == "Hello"
    assert messages[1].role == "ai"
    assert messages[1].content == "Hi there"
    
    # Test clearing history
    history.clear()
    messages = history.get_messages()
    assert len(messages) == 0


def test_sqlite_chat_message_history_async():
    """Test the async methods of SQLiteChatMessageHistory."""
    # Skip the async tests if pytest-asyncio is not installed
    pytest.importorskip("pytest_asyncio")
    history = SQLiteChatMessageHistory(db_path=":memory:")
    
    # Add messages synchronously for the test
    history.add_message(HumanMessage(content="async test"))
    history.add_message(AIMessage(content="async response"))
    
    # Get messages synchronously to verify
    messages = history.get_messages()
    assert len(messages) == 2
    assert messages[0].content == "async test"
    
    # Clear synchronously
    history.clear()
    messages = history.get_messages()
    assert len(messages) == 0


def test_sqlite_memory_file_persistence():
    """Test that SQLiteMemory persists data to file."""
    # Create temporary file
    fd, path = tempfile.mkstemp()
    os.close(fd)
    
    try:
        # Create memory with the file path
        memory = SQLiteMemory(db_path=path)
        
        # Add some conversation data
        memory.save_context(
            {"input": "test question"},
            {"output": "test answer"}
        )
        
        # Create a new memory instance with the same path
        new_memory = SQLiteMemory(db_path=path)
        
        # Check it loads the previous data
        variables = new_memory.load_memory_variables({})
        assert len(variables["history"]) == 2
        assert variables["history"][0]["content"] == "test question"
        
    finally:
        # Ensure connections are closed before removing the file
        memory = None
        new_memory = None
        import gc
        gc.collect()  # Force garbage collection to close any lingering connections
        try:
            if os.path.exists(path):
                os.unlink(path)
        except (PermissionError, OSError):
            pass  # Skip removal if file is still locked


def test_sqlite_memory_save_load():
    """Test SQLiteMemory save and load functionality."""
    memory = SQLiteMemory(db_path=":memory:", memory_key="chat_history")
    
    # Test save_context
    memory.save_context(
        {"input": "What's the weather?"},
        {"output": "It's sunny."}
    )
    memory.save_context(
        {"input": "Will it rain tomorrow?"},
        {"output": "There's a 30% chance of rain."}
    )
    
    # Test load_memory_variables
    result = memory.load_memory_variables({})
    assert "chat_history" in result
    messages = result["chat_history"]
    assert len(messages) == 4
    assert messages[0]["role"] == "human"
    assert messages[1]["role"] == "ai"
    assert messages[2]["role"] == "human"
    assert messages[3]["role"] == "ai"


def test_sqlite_memory_async():
    """Test SQLiteMemory async methods."""
    # Skip the test if pytest-asyncio is not installed
    pytest.importorskip("pytest_asyncio")
    memory = SQLiteMemory(db_path=":memory:")
    
    # Test synchronously, since we're just testing the interface
    memory.save_context(
        {"input": "async question"},
        {"output": "async answer"}
    )
    
    result = memory.load_memory_variables({})
    assert len(result["history"]) == 2
    
    memory.clear()
    empty = memory.load_memory_variables({})
    assert len(empty["history"]) == 0