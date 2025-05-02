"""OpenAI LLM Adapter."""

import json
import os
import warnings
from typing import Any, AsyncIterator, ClassVar, Dict, Iterator, List, Optional, Union

try:
    from openai import AsyncOpenAI, OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    warnings.warn("OpenAI package not found. Install it with 'pip install openai'.")

from hotlm_core.schema import AIMessage, BaseMessage, ChatResult, Generation

from .base import LLMProviderAdapter
from .utils import TokenCounter


class OpenAIAdapter(LLMProviderAdapter):
    """Adapter for OpenAI chat models."""

    provider_name: ClassVar[str] = "openai"
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        api_base: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        **kwargs
    ):
        """Initialize OpenAI Adapter.
        
        Args:
            model_name: The name of the OpenAI model to use
            api_key: OpenAI API key. If None, will look for OPENAI_API_KEY environment variable
            organization: OpenAI organization ID
            api_base: Base URL for API requests. For non-OpenAI hosted endpoints
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0-1)
            top_p: Nucleus sampling parameter (0-1)
            **kwargs: Additional parameters to pass to the OpenAI API
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
        
        self.organization = organization
        self._client = None
        self._async_client = None
        
        # Check if OpenAI is installed
        if not HAS_OPENAI:
            raise ImportError(
                "To use the OpenAI adapter, you need to install the OpenAI package: `pip install openai`"
            )
    
    @property
    def client(self):
        """Get the OpenAI client, initializing if needed."""
        if self._client is None:
            kwargs = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.organization:
                kwargs["organization"] = self.organization
            if self.api_base:
                kwargs["base_url"] = self.api_base
            
            self._client = OpenAI(**kwargs)
        return self._client
    
    @property
    def async_client(self):
        """Get the async OpenAI client, initializing if needed."""
        if self._async_client is None:
            kwargs = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.organization:
                kwargs["organization"] = self.organization
            if self.api_base:
                kwargs["base_url"] = self.api_base
            
            self._async_client = AsyncOpenAI(**kwargs)
        return self._async_client
    
    def _format_messages(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """Format messages for the OpenAI API."""
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
    
    def _calculate_token_usage(self, prompt_text: str, completion_text: str) -> Dict[str, int]:
        """Calculate token usage for the request and response."""
        return TokenCounter.estimate_token_usage(
            prompt_text, 
            completion_text, 
            provider="openai", 
            model_name=self.model_name
        )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Invoke the model with a list of messages."""
        # Format messages for the OpenAI API
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
        
        # Call the OpenAI API
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
        # Format messages for the OpenAI API
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
        
        # Call the OpenAI API asynchronously
        response = await self.async_client.chat.completions.create(**params)
        
        # Extract content from the response
        response_message = response.choices[0].message
        response_content = response_message.content or ""
        
        # Create a Generation with the response
        generation = Generation(message=AIMessage(content=response_content))
        
        # Return the ChatResult
        return ChatResult(generations=[generation], llm_output=response.model_dump())

    def stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[AIMessage]:
        """Stream a response from the model token by token."""
        # Format messages for the OpenAI API
        formatted_messages = self._format_messages(messages)
        
        # Build request parameters
        params = {
            "model": self.model_name,
            "messages": formatted_messages,
            "stream": True,  # Enable streaming
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
        
        # Call the OpenAI API with streaming enabled
        response_stream = self.client.chat.completions.create(**params)
        
        # Process the streaming response
        current_content = ""
        for chunk in response_stream:
            if not chunk.choices:
                continue
                
            # Extract the delta content
            delta = chunk.choices[0].delta
            if not delta.content:
                continue
                
            # Accumulate the content
            current_content += delta.content
            
            # Yield a message with the accumulated content so far
            yield AIMessage(content=current_content)

    async def astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[AIMessage]:
        """Stream a response from the model token by token asynchronously."""
        # Format messages for the OpenAI API
        formatted_messages = self._format_messages(messages)
        
        # Build request parameters
        params = {
            "model": self.model_name,
            "messages": formatted_messages,
            "stream": True,  # Enable streaming
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
        
        # Call the OpenAI API asynchronously with streaming enabled
        response_stream = await self.async_client.chat.completions.create(**params)
        
        # Process the streaming response
        current_content = ""
        async for chunk in response_stream:
            if not chunk.choices:
                continue
                
            # Extract the delta content
            delta = chunk.choices[0].delta
            if not delta.content:
                continue
                
            # Accumulate the content
            current_content += delta.content
            
            # Yield a message with the accumulated content so far
            yield AIMessage(content=current_content)
            
    def function_call(
        self,
        messages: List[BaseMessage],
        functions: List[Dict[str, Any]],
        force_function: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Call a function based on the OpenAI model's decision.
        
        Args:
            messages: The messages to send to the model
            functions: List of function definitions in OpenAI format
            force_function: Force the model to call this specific function
            **kwargs: Additional parameters for the model
            
        Returns:
            A dictionary with function_name and function_arguments
        """
        # Format messages for the OpenAI API
        formatted_messages = self._format_messages(messages)
        
        # Build the request parameters
        params = {
            "model": self.model_name,
            "messages": formatted_messages,
            "tools": [{"type": "function", "function": func} for func in functions],
        }
        
        # Add optional parameters
        if self.max_tokens:
            params["max_tokens"] = self.max_tokens
        if self.temperature:
            params["temperature"] = self.temperature
        if self.top_p:
            params["top_p"] = self.top_p
            
        # Handle forced function calling
        if force_function:
            params["tool_choice"] = {
                "type": "function",
                "function": {"name": force_function}
            }
        else:
            # Auto mode - let the model decide
            params["tool_choice"] = "auto"
            
        # Add any additional parameters
        params.update(self.additional_params)
        
        # Override with any parameters passed to this method
        params.update(kwargs)
        
        # Call the OpenAI API
        response = self.client.chat.completions.create(**params)
        
        # Extract the function call information
        tool_calls = response.choices[0].message.tool_calls
        
        if not tool_calls:
            # Model chose not to call a function
            return {
                "function_name": None,
                "function_arguments": {},
                "function_response": response.choices[0].message.content or "",
            }
        
        # Get the first function call
        function_call = tool_calls[0].function
        
        # Parse the function arguments from JSON
        try:
            function_args = json.loads(function_call.arguments)
        except json.JSONDecodeError:
            function_args = {"error": "Invalid JSON", "raw": function_call.arguments}
        
        return {
            "function_name": function_call.name,
            "function_arguments": function_args,
            "model_response": response.model_dump(),
        }
    
    async def afunction_call(
        self,
        messages: List[BaseMessage],
        functions: List[Dict[str, Any]],
        force_function: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Asynchronously call a function based on the OpenAI model's decision.
        
        Args:
            messages: The messages to send to the model
            functions: List of function definitions in OpenAI format
            force_function: Force the model to call this specific function
            **kwargs: Additional parameters for the model
            
        Returns:
            A dictionary with function_name and function_arguments
        """
        # Format messages for the OpenAI API
        formatted_messages = self._format_messages(messages)
        
        # Build the request parameters
        params = {
            "model": self.model_name,
            "messages": formatted_messages,
            "tools": [{"type": "function", "function": func} for func in functions],
        }
        
        # Add optional parameters
        if self.max_tokens:
            params["max_tokens"] = self.max_tokens
        if self.temperature:
            params["temperature"] = self.temperature
        if self.top_p:
            params["top_p"] = self.top_p
            
        # Handle forced function calling
        if force_function:
            params["tool_choice"] = {
                "type": "function",
                "function": {"name": force_function}
            }
        else:
            # Auto mode - let the model decide
            params["tool_choice"] = "auto"
            
        # Add any additional parameters
        params.update(self.additional_params)
        
        # Override with any parameters passed to this method
        params.update(kwargs)
        
        # Call the OpenAI API asynchronously
        response = await self.async_client.chat.completions.create(**params)
        
        # Extract the function call information
        tool_calls = response.choices[0].message.tool_calls
        
        if not tool_calls:
            # Model chose not to call a function
            return {
                "function_name": None,
                "function_arguments": {},
                "function_response": response.choices[0].message.content or "",
            }
        
        # Get the first function call
        function_call = tool_calls[0].function
        
        # Parse the function arguments from JSON
        try:
            function_args = json.loads(function_call.arguments)
        except json.JSONDecodeError:
            function_args = {"error": "Invalid JSON", "raw": function_call.arguments}
        
        return {
            "function_name": function_call.name,
            "function_arguments": function_args,
            "model_response": response.model_dump(),
        }
        
    def supports_function_calling(self) -> bool:
        """Check if this adapter supports function calling.
        
        Returns:
            True if the adapter supports function calling, False otherwise
        """
        # Most modern OpenAI models support function calling
        # Specifically check for GPT-4 and GPT-3.5-Turbo models
        return (
            self.model_name.startswith("gpt-4") or
            self.model_name.startswith("gpt-3.5-turbo") or
            self.model_name.startswith("gpt-3.5-turbo")
        )
        
    def supports_vision(self) -> bool:
        """Check if this adapter supports vision/image input.
        
        Returns:
            True if the adapter supports vision input, False otherwise
        """
        # Only certain OpenAI models support vision
        return (
            "vision" in self.model_name or
            "gpt-4-vision" in self.model_name or
            self.model_name == "gpt-4-turbo" or
            self.model_name == "gpt-4-1106-preview" or
            self.model_name == "gpt-4o" or
            self.model_name.startswith("gpt-4o-")
        )

# Register the adapter
LLMProviderAdapter.register_adapter("openai", OpenAIAdapter)
