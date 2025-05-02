"""Anthropic LLM Adapter."""

import json
import os
import warnings
from typing import Any, ClassVar, Dict, List, Optional, Union

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    warnings.warn("Anthropic package not found. Install it with 'pip install anthropic'.")

from hotlm_core.schema import AIMessage, BaseMessage, ChatResult, Generation

from .base import LLMProviderAdapter


class AnthropicAdapter(LLMProviderAdapter):
    """Adapter for Anthropic Claude models."""

    provider_name: ClassVar[str] = "anthropic"
    
    def __init__(
        self,
        model_name: str = "claude-3-opus-20240229",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 1.0,
        **kwargs
    ):
        """Initialize Anthropic Adapter.
        
        Args:
            model_name: The name of the Anthropic model to use
            api_key: Anthropic API key. If None, will look for ANTHROPIC_API_KEY environment variable
            api_base: Base URL for API requests. For non-default endpoints
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0-1)
            top_p: Nucleus sampling parameter (0-1)
            **kwargs: Additional parameters to pass to the Anthropic API
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
        
        # Check if Anthropic is installed
        if not HAS_ANTHROPIC:
            raise ImportError(
                "To use the Anthropic adapter, you need to install the Anthropic package: `pip install anthropic`"
            )
    
    @property
    def client(self):
        """Get the Anthropic client, initializing if needed."""
        if self._client is None:
            kwargs = {}
            
            # Use provided API key or fall back to environment variable
            api_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("No API key provided for Anthropic. Set the ANTHROPIC_API_KEY environment variable or pass api_key.")
            
            kwargs["api_key"] = api_key
            
            # Use custom base URL if provided
            if self.api_base:
                kwargs["base_url"] = self.api_base
            
            self._client = anthropic.Anthropic(**kwargs)
        return self._client
    
    def _format_messages(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """Format messages for the Anthropic API."""
        formatted_messages = []
        for msg in messages:
            role = msg.type.value
            if role == "ai":
                role = "assistant"
            elif role == "human":
                role = "user"
            elif role == "system":
                # Anthropic handles system messages differently
                continue  # We'll add the system message separately
                
            formatted_messages.append({
                "role": role,
                "content": msg.content
            })
            
        return formatted_messages

    def _extract_system_message(self, messages: List[BaseMessage]) -> Optional[str]:
        """Extract the system message from the list of messages."""
        for msg in messages:
            if msg.type.value == "system":
                return msg.content
        return None

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Invoke the model with a list of messages."""
        # Format messages for the Anthropic API
        formatted_messages = self._format_messages(messages)
        system_message = self._extract_system_message(messages)
        
        # Build request parameters
        params = {
            "model": self.model_name,
            "messages": formatted_messages,
            "max_tokens": self.max_tokens,
        }
        
        # Add system message if present
        if system_message:
            params["system"] = system_message
        
        # Add optional parameters
        if stop:
            params["stop_sequences"] = stop
        if self.temperature:
            params["temperature"] = self.temperature
        if self.top_p:
            params["top_p"] = self.top_p
            
        # Add any additional parameters
        params.update(self.additional_params)
        
        # Override with any parameters passed to this method
        params.update(kwargs)
        
        # Call the Anthropic API
        response = self.client.messages.create(**params)
        
        # Extract content from the response
        response_content = response.content[0].text
        
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
        # Format messages for the Anthropic API
        formatted_messages = self._format_messages(messages)
        system_message = self._extract_system_message(messages)
        
        # Build request parameters
        params = {
            "model": self.model_name,
            "messages": formatted_messages,
            "max_tokens": self.max_tokens,
        }
        
        # Add system message if present
        if system_message:
            params["system"] = system_message
        
        # Add optional parameters
        if stop:
            params["stop_sequences"] = stop
        if self.temperature:
            params["temperature"] = self.temperature
        if self.top_p:
            params["top_p"] = self.top_p
            
        # Add any additional parameters
        params.update(self.additional_params)
        
        # Override with any parameters passed to this method
        params.update(kwargs)
        
        # Call the Anthropic API asynchronously
        response = await self.client.messages.create(**params)
        
        # Extract content from the response
        response_content = response.content[0].text
        
        # Create a Generation with the response
        generation = Generation(message=AIMessage(content=response_content))
        
        # Return the ChatResult
        return ChatResult(generations=[generation], llm_output=response.model_dump())

    def function_call(
        self,
        messages: List[BaseMessage],
        functions: List[Dict[str, Any]],
        force_function: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        formatted = self._format_messages(messages)
        system = self._extract_system_message(messages)
        params: Dict[str, Any] = {"model": self.model_name, "messages": formatted, "functions": functions}
        if system: params["system"] = system
        if self.max_tokens: params["max_tokens"] = self.max_tokens
        if self.temperature: params["temperature"] = self.temperature
        if self.top_p: params["top_p"] = self.top_p
        if force_function:
            params["function_call"] = {"name": force_function}
        else:
            params["function_call"] = "auto"
        params.update(self.additional_params)
        params.update(kwargs)
        response = self.client.messages.create(**params)
        fc = getattr(response, 'function_call', None)
        if not fc:
            return {"function_name": None, "function_arguments": {}, "model_response": response.model_dump()}
        call = fc
        try:
            args = json.loads(call.arguments)
        except json.JSONDecodeError:
            args = {"error": "Invalid JSON", "raw": call.arguments}
        return {"function_name": call.name, "function_arguments": args, "model_response": response.model_dump()}

    async def afunction_call(
        self,
        messages: List[BaseMessage],
        functions: List[Dict[str, Any]],
        force_function: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        formatted = self._format_messages(messages)
        system = self._extract_system_message(messages)
        params: Dict[str, Any] = {"model": self.model_name, "messages": formatted, "functions": functions}
        if system: params["system"] = system
        if self.max_tokens: params["max_tokens"] = self.max_tokens
        if self.temperature: params["temperature"] = self.temperature
        if self.top_p: params["top_p"] = self.top_p
        if force_function:
            params["function_call"] = {"name": force_function}
        else:
            params["function_call"] = "auto"
        params.update(self.additional_params)
        params.update(kwargs)
        response = await self.client.messages.create(**params)
        fc = getattr(response, 'function_call', None)
        if not fc:
            return {"function_name": None, "function_arguments": {}, "model_response": response.model_dump()}
        call = fc
        try:
            args = json.loads(call.arguments)
        except json.JSONDecodeError:
            args = {"error": "Invalid JSON", "raw": call.arguments}
        return {"function_name": call.name, "function_arguments": args, "model_response": response.model_dump()}

# Register the adapter
LLMProviderAdapter.register_adapter("anthropic", AnthropicAdapter)