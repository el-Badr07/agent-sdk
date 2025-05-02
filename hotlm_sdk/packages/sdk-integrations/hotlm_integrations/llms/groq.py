"""Groq LLM Adapter."""

import os
import warnings
from typing import Any, ClassVar, Dict, List, Optional, Union

try:
    from groq import AsyncGroq, Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False
    warnings.warn("Groq package not found. Install it with 'pip install groq'.")

from hotlm_core.schema import AIMessage, BaseMessage, ChatResult, Generation

from .base import LLMProviderAdapter


class GroqAdapter(LLMProviderAdapter):
    """Adapter for Groq models."""

    provider_name: ClassVar[str] = "groq"
    
    def __init__(
        self,
        model_name: str = "llama3-70b-8192",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        **kwargs
    ):
        """Initialize Groq Adapter.
        
        Args:
            model_name: The name of the Groq model to use (e.g., "llama3-70b-8192", "mixtral-8x7b-32768")
            api_key: Groq API key. If None, will look for GROQ_API_KEY environment variable
            api_base: Base URL for API requests
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0-1)
            top_p: Nucleus sampling parameter (0-1)
            **kwargs: Additional parameters to pass to the Groq API
        """
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            api_base=api_base,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
        
        self._client = None
        self._async_client = None
        
        # Check if Groq is installed
        if not HAS_GROQ:
            raise ImportError(
                "To use the Groq adapter, you need to install the Groq package: `pip install groq`"
            )
    
    @property
    def client(self):
        """Get the Groq client, initializing if needed."""
        if self._client is None:
            kwargs = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.api_base:
                kwargs["base_url"] = self.api_base
            
            self._client = Groq(**kwargs)
        return self._client
    
    @property
    def async_client(self):
        """Get the async Groq client, initializing if needed."""
        if self._async_client is None:
            kwargs = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.api_base:
                kwargs["base_url"] = self.api_base
            
            self._async_client = AsyncGroq(**kwargs)
        return self._async_client
    
    def _format_messages(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """Format messages for the Groq API."""
        formatted_messages = []
        for msg in messages:
            role = msg.type.value
            if role == "ai":
                role = "assistant"
            elif role == "human":
                role = "user"
            formatted_messages.append({
                "role": role,
                "content": msg.content
            })
        return formatted_messages

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Invoke the model with a list of messages."""
        # Format messages for the Groq API
        formatted_messages = self._format_messages(messages)
        
        # Build request parameters
        params = {
            "model": self.model_name,
            "messages": formatted_messages,
        }
        
        # Add optional parameters
        if stop:
            params["stop"] = stop
        if self.max_tokens:
            params["max_tokens"] = self.max_tokens
        if self.temperature:
            params["temperature"] = self.temperature
        if self.top_p:
            params["top_p"] = self.top_p
            
        # Add any additional parameters
        params.update(self.additional_params)
        
        # Override with any parameters passed to this method
        params.update(kwargs)
        
        # Call the Groq API
        response = self.client.chat.completions.create(**params)
        
        # Extract content from the response
        response_message = response.choices[0].message
        response_content = response_message.content or ""
        
        # Create a Generation with the response
        generation = Generation(message=AIMessage(content=response_content))
        
        # Return the ChatResult
        return ChatResult(generations=[generation], llm_output=response.model_dump())

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Asynchronously invoke the model."""
        # Format messages for the Groq API
        formatted_messages = self._format_messages(messages)
        
        # Build request parameters
        params = {
            "model": self.model_name,
            "messages": formatted_messages,
        }
        
        # Add optional parameters
        if stop:
            params["stop"] = stop
        if self.max_tokens:
            params["max_tokens"] = self.max_tokens
        if self.temperature:
            params["temperature"] = self.temperature
        if self.top_p:
            params["top_p"] = self.top_p
            
        # Add any additional parameters
        params.update(self.additional_params)
        
        # Override with any parameters passed to this method
        params.update(kwargs)
        
        # Call the Groq API asynchronously
        response = await self.async_client.chat.completions.create(**params)
        
        # Extract content from the response
        response_message = response.choices[0].message
        response_content = response_message.content or ""
        
        # Create a Generation with the response
        generation = Generation(message=AIMessage(content=response_content))
        
        # Return the ChatResult
        return ChatResult(generations=[generation], llm_output=response.model_dump())

# Register the adapter
LLMProviderAdapter.register_adapter("groq", GroqAdapter)