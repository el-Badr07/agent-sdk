"""Tests for OpenAI LLM Adapter."""

import pytest

# from hotlm_integrations.llms.openai import OpenAIAdapter

# TODO: Add actual tests using mocking or dedicated test keys/models

@pytest.mark.skip(reason="OpenAI adapter not fully implemented or requires API key")
def test_openai_adapter_initialization():
    """Test basic initialization."""
    # adapter = OpenAIAdapter(api_key="test_key", model_name="test_model")
    # assert adapter.model_name == "test_model"
    pass

@pytest.mark.skip(reason="OpenAI adapter not fully implemented or requires API key")
def test_openai_adapter_invoke():
    """Test invoke method (mocked)."""
    # adapter = OpenAIAdapter(api_key="test_key")
    # Mock the client call here
    # result = adapter.invoke([{"role": "user", "content": "Hello!"}])
    # assert isinstance(result, str) # Or ChatResult later
    pass

@pytest.mark.asyncio
@pytest.mark.skip(reason="OpenAI adapter not fully implemented or requires API key")
async def test_openai_adapter_ainvoke():
    """Test ainvoke method (mocked)."""
    # adapter = OpenAIAdapter(api_key="test_key")
    # Mock the async client call here
    # result = await adapter.ainvoke([{"role": "user", "content": "Hello!"}])
    # assert isinstance(result, str) # Or ChatResult later
    pass
