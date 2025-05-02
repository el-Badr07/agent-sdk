"""Utilities for LLM providers."""

import warnings
from typing import Any, Dict, List, Optional, Union

# Try importing tokenizers for different providers
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    warnings.warn("tiktoken not installed. Install with 'pip install tiktoken' for accurate OpenAI token counting.")

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    warnings.warn("transformers not installed. Install with 'pip install transformers' for HuggingFace tokenization.")

class TokenCounter:
    """Utility for counting tokens across different LLM providers."""
    
    # Cache tokenizers to avoid reloading them
    _tokenizers = {}
    
    @classmethod
    def count_tokens(
        cls, 
        text: str, 
        provider: str = "openai", 
        model_name: Optional[str] = None
    ) -> int:
        """Count the number of tokens in the given text.
        
        Args:
            text: The text to count tokens for
            provider: The provider name (openai, anthropic, huggingface, etc.)
            model_name: The specific model name, if needed for accurate counting
            
        Returns:
            The number of tokens in the text
        """
        provider = provider.lower()
        
        # Handle different providers
        if provider == "openai" or provider == "azure":
            return cls._count_openai_tokens(text, model_name)
        elif provider == "anthropic":
            return cls._count_anthropic_tokens(text)
        elif provider == "huggingface":
            return cls._count_huggingface_tokens(text, model_name)
        elif provider == "groq":
            # Groq uses similar tokenization to OpenAI for most models
            return cls._count_openai_tokens(text, model_name)
        elif provider == "vertexai":
            # For Vertex AI, use general purpose tokenization
            # This is a rough approximation
            return cls._count_generic_tokens(text)
        else:
            # Default to generic token counting
            return cls._count_generic_tokens(text)
    
    @classmethod
    def _count_openai_tokens(cls, text: str, model_name: Optional[str] = None) -> int:
        """Count tokens using OpenAI's tiktoken library."""
        if not HAS_TIKTOKEN:
            return cls._count_generic_tokens(text)
            
        # Default to gpt-3.5-turbo if no model specified
        if model_name is None:
            model_name = "gpt-3.5-turbo"
            
        # Get encoding based on model name
        try:
            # Use cached tokenizer if available
            encoding_name = f"openai_{model_name}"
            if encoding_name not in cls._tokenizers:
                # Try to get the encoding for the specific model
                try:
                    encoding = tiktoken.encoding_for_model(model_name)
                except KeyError:
                    # Fall back to cl100k_base for newer models
                    encoding = tiktoken.get_encoding("cl100k_base")
                cls._tokenizers[encoding_name] = encoding
            else:
                encoding = cls._tokenizers[encoding_name]
                
            # Count tokens
            return len(encoding.encode(text))
        except Exception as e:
            warnings.warn(f"Error counting OpenAI tokens: {e}. Using generic count instead.")
            return cls._count_generic_tokens(text)
    
    @classmethod
    def _count_anthropic_tokens(cls, text: str) -> int:
        """Count tokens for Anthropic models."""
        # Anthropic uses a similar tokenizer to GPT models
        if HAS_TIKTOKEN:
            return cls._count_openai_tokens(text, "cl100k_base")
        else:
            # Approximate count for Anthropic models
            return cls._count_generic_tokens(text)
    
    @classmethod
    def _count_huggingface_tokens(cls, text: str, model_name: Optional[str] = None) -> int:
        """Count tokens using a HuggingFace tokenizer."""
        if not HAS_TRANSFORMERS:
            return cls._count_generic_tokens(text)
            
        # Default model for tokenization if none specified
        if model_name is None:
            model_name = "gpt2"  # Good general purpose tokenizer
            
        # Use cached tokenizer if available
        tokenizer_key = f"hf_{model_name}"
        if tokenizer_key not in cls._tokenizers:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                cls._tokenizers[tokenizer_key] = tokenizer
            except Exception as e:
                warnings.warn(f"Error loading HF tokenizer: {e}. Using generic count.")
                return cls._count_generic_tokens(text)
        else:
            tokenizer = cls._tokenizers[tokenizer_key]
            
        # Count tokens
        return len(tokenizer.encode(text))
    
    @classmethod
    def _count_generic_tokens(cls, text: str) -> int:
        """Generic token counting as a fallback.
        
        This is a rough approximation that works reasonably well for many tokenizers.
        """
        # Approximate token count based on words/subwords
        # This is a rough approximation, not exact
        # Most tokenizers are roughly 4 chars per token on average
        return max(1, len(text) // 4)
        
    @classmethod
    def estimate_token_usage(
        cls,
        prompt: str,
        completion: str,
        provider: str = "openai",
        model_name: Optional[str] = None
    ) -> Dict[str, int]:
        """Estimate token usage for a prompt and completion.
        
        Args:
            prompt: The input prompt
            completion: The model's completion
            provider: The provider name
            model_name: The specific model name
            
        Returns:
            Dictionary with prompt_tokens, completion_tokens, and total_tokens
        """
        prompt_tokens = cls.count_tokens(prompt, provider, model_name)
        completion_tokens = cls.count_tokens(completion, provider, model_name)
        
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }