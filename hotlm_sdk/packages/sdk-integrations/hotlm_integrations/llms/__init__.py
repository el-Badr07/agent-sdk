"""LLM provider adapters and factory.

This module provides adapters for various LLM providers, including:
- OpenAI
- Azure OpenAI
- Anthropic
- Google Vertex AI
- Groq
- HuggingFace

It also includes a factory that can auto-detect the provider from configuration.
"""

from .anthropic import AnthropicAdapter
from .azure import AzureOpenAIAdapter
from .base import LLMProviderAdapter
from .factory import LLMProviderFactory
from .groq import GroqAdapter
from .huggingface import HuggingFaceAdapter

# Import all provider adapters
from .openai import OpenAIAdapter
from .vertexai import VertexAIAdapter


# Create convenience methods
def create_chat_model(
    model_name: str,
    provider: str = None,
    **kwargs
) -> LLMProviderAdapter:
    """Create a chat model from the specified provider.
    
    Args:
        model_name: The name of the model
        provider: The provider name (openai, azure, anthropic, vertexai, groq, huggingface).
                 If None, will auto-detect from model_name and configuration.
        **kwargs: Additional provider-specific parameters
        
    Returns:
        A chat model instance
    """
    return LLMProviderFactory.create(model_name=model_name, provider=provider, **kwargs)

__all__ = [
    # Base
    "LLMProviderAdapter",
    # Factory
    "LLMProviderFactory",
    "create_chat_model",
    # Adapters
    "OpenAIAdapter",
    "AzureOpenAIAdapter",
    "AnthropicAdapter", 
    "VertexAIAdapter",
    "GroqAdapter",
    "HuggingFaceAdapter",
]