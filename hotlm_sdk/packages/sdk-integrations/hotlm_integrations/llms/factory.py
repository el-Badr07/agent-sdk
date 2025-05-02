"""LLM provider factory with auto-detection capabilities."""

import os
import re
from typing import Any, Dict, List, Optional, Type, Union

from hotlm_core.errors import HotLMConfigurationError
from hotlm_core.models import BaseChatModel

from .base import LLMProviderAdapter


class LLMProviderFactory:
    """Factory for creating LLM provider adapters with auto-detection."""
    
    # Model name prefixes that indicate a specific provider
    _MODEL_PREFIX_MAPPING = {
        "gpt-": "openai",
        "text-embedding-ada": "openai",
        "claude-": "anthropic",
        "gemini-": "vertexai",
        "mistral/": "huggingface",
        "meta-llama/": "huggingface",
        "mixtral-": "groq",  # mixtral models on Groq
        "llama": "groq",     # llama models on Groq
    }
    
    @classmethod
    def create(
        cls,
        model_name: str,
        provider: Optional[str] = None,
        **kwargs
    ) -> BaseChatModel:
        """Create an LLM provider adapter instance.
        
        Args:
            model_name: The name of the model
            provider: The provider name, or None to auto-detect
            **kwargs: Additional parameters to pass to the adapter
            
        Returns:
            An instance of the appropriate LLM provider adapter
            
        Raises:
            HotLMConfigurationError: If the provider cannot be auto-detected or is not supported
        """
        if provider is None:
            provider = cls._detect_provider(model_name, **kwargs)
            
        adapter_class = LLMProviderAdapter.get_adapter_class(provider)
        if adapter_class is None:
            supported = list(LLMProviderAdapter._adapter_registry.keys())
            raise HotLMConfigurationError(
                f"Provider '{provider}' not supported. Supported providers: {', '.join(supported)}"
            )
            
        # Special handling for certain providers to adjust parameters as needed
        if provider == "azure":
            # For Azure, we need to convert model_name to deployment_name if not specified
            if "deployment_name" not in kwargs:
                kwargs["deployment_name"] = model_name
                
        elif provider == "huggingface":
            # For HuggingFace, try to determine if this is a Hub model or local path
            if "/" in model_name and not os.path.exists(model_name):
                # Likely a Hub model
                kwargs["repo_id"] = model_name
            elif os.path.exists(model_name):
                # Likely a local path
                kwargs["model_path"] = model_name
                
        # Create and return the adapter
        return adapter_class(model_name=model_name, **kwargs)
    
    @classmethod
    def _detect_provider(cls, model_name: str, **kwargs) -> str:
        """Auto-detect the provider from the model name and configuration.
        
        Args:
            model_name: The name of the model
            **kwargs: Additional configuration parameters
            
        Returns:
            The detected provider name
            
        Raises:
            HotLMConfigurationError: If the provider cannot be auto-detected
        """
        # If azure_endpoint or azure_deployment_name is specified, use Azure
        if kwargs.get("azure_endpoint") or kwargs.get("deployment_name"):
            return "azure"
            
        # Check for specific environment variables
        if "OPENAI_API_KEY" in os.environ and not any(k in os.environ for k in [
            "ANTHROPIC_API_KEY", "GROQ_API_KEY", "VERTEX_AI_CREDENTIALS"
        ]):
            return "openai"
            
        if "ANTHROPIC_API_KEY" in os.environ and not any(k in os.environ for k in [
            "OPENAI_API_KEY", "GROQ_API_KEY", "VERTEX_AI_CREDENTIALS"
        ]):
            return "anthropic"
            
        if "GROQ_API_KEY" in os.environ and not any(k in os.environ for k in [
            "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "VERTEX_AI_CREDENTIALS"
        ]):
            return "groq"
            
        # Check for model name prefixes
        for prefix, provider in cls._MODEL_PREFIX_MAPPING.items():
            if model_name.startswith(prefix) or model_name.startswith(prefix.lower()):
                return provider
                
        # If the model name contains a slash and doesn't exist as a local path,
        # assume it's a HuggingFace model
        if "/" in model_name and not os.path.exists(model_name):
            return "huggingface"
            
        # If the model name exists as a local path, assume it's a local HuggingFace model
        if os.path.exists(model_name):
            return "huggingface"
            
        # Default to OpenAI as the most common provider
        return "openai"