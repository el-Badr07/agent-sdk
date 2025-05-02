"""HotLM SDK integrations package.

This package provides integrations with various external services and libraries:

- LLM providers (OpenAI, Azure OpenAI, Anthropic, Google Vertex AI, Groq, HuggingFace)
- Vector stores
- Document loaders
- And more
"""

from . import llms

# Re-export LLM provider functionality for easy access
from .llms import (  # Factory function for auto-detection; Base adapter; Provider adapters
    AnthropicAdapter,
    AzureOpenAIAdapter,
    GroqAdapter,
    HuggingFaceAdapter,
    LLMProviderAdapter,
    OpenAIAdapter,
    VertexAIAdapter,
    create_chat_model,
)

__all__ = [
    # Modules
    "llms",
    
    # LLM provider functionality
    "create_chat_model",
    "LLMProviderAdapter",
    "OpenAIAdapter",
    "AzureOpenAIAdapter",
    "AnthropicAdapter",
    "VertexAIAdapter",
    "GroqAdapter",
    "HuggingFaceAdapter",
]