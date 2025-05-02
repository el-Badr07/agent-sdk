"""Tests for ConversationBufferMemory."""

import pytest
from hotlm_agents.memory_layers.buffer import ConversationBufferMemory

# Import InMemoryChatMessageHistory from sdk-core
from hotlm_core.memory.history import InMemoryChatMessageHistory
from hotlm_core.schema import AIMessage, HumanMessage


def test_buffer_memory_initialization():
    """Test basic initialization."""
    memory = ConversationBufferMemory()
    assert memory.memory_key == "history"
    assert not memory.return_messages
    # Check that it defaults to the core InMemoryChatMessageHistory
    assert isinstance(memory.chat_memory, InMemoryChatMessageHistory)


def test_buffer_memory_save_and_load_string():
    """Test saving context and loading as string."""
    memory = ConversationBufferMemory(return_messages=False)
    inputs = {"input": "Hello there"}
    outputs = {"output": "General Kenobi"}
    memory.save_context(inputs, outputs)

    loaded_vars = memory.load_memory_variables({})
    assert memory.memory_key in loaded_vars
    expected_string = "Human: Hello there\nAI: General Kenobi"
    assert loaded_vars[memory.memory_key] == expected_string


def test_buffer_memory_save_and_load_messages():
    """Test saving context and loading as BaseMessage objects."""
    memory = ConversationBufferMemory(return_messages=True)
    inputs = {"input": "Ping"}
    outputs = {"response": "Pong"}  # Use a different output key
    memory.save_context(inputs, outputs)

    loaded_vars = memory.load_memory_variables({})
    assert memory.memory_key in loaded_vars
    messages = loaded_vars[memory.memory_key]
    assert len(messages) == 2
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == "Ping"
    assert isinstance(messages[1], AIMessage)
    # Output key was 'response', but save_context saves the value
    assert messages[1].content == "Pong"


def test_buffer_memory_save_with_keys():
    """Test saving context using specific input/output keys."""
    memory = ConversationBufferMemory(
        input_key="user_query", output_key="agent_answer", return_messages=True
    )
    inputs = {"user_query": "Question?", "other_input": "ignore"}
    outputs = {"agent_answer": "Answer!", "debug_info": "ignore"}
    memory.save_context(inputs, outputs)

    messages = memory.load_memory_variables({})[memory.memory_key]
    assert len(messages) == 2
    assert messages[0].content == "Question?"
    assert messages[1].content == "Answer!"


def test_buffer_memory_clear():
    """Test clearing the memory."""
    memory = ConversationBufferMemory()
    memory.save_context({"input": "foo"}, {"output": "bar"})
    assert len(memory.chat_memory.messages) == 2
    loaded_vars_before = memory.load_memory_variables({})
    assert loaded_vars_before[memory.memory_key]  # Should not be empty

    memory.clear()
    assert len(memory.chat_memory.messages) == 0
    loaded_vars_after = memory.load_memory_variables({})
    assert not loaded_vars_after[
        memory.memory_key
    ]  # Should be empty string or empty list
