"""Azure OpenAI LLM Adapter."""

import json
import os
import warnings
from typing import Any, ClassVar, Dict, List, Optional, Union

try:
    from openai import AsyncAzureOpenAI, AzureOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    warnings.warn("OpenAI package not found. Install it with 'pip install openai'.")

from hotlm_core.schema import AIMessage, BaseMessage, ChatResult, Generation

from .base import LLMProviderAdapter


class AzureOpenAIAdapter(LLMProviderAdapter):
    """Adapter for Azure OpenAI chat models."""

    provider_name: ClassVar[str] = "azure"
    
    def __init__(
        self,
        deployment_name: str,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: str = "2023-05-15",
        azure_endpoint: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        **kwargs
    ):
        """Initialize Azure OpenAI Adapter.
        
        Args:
            deployment_name: The name of the Azure OpenAI deployment
            model_name: The name of the model (for logging/metadata purposes)
            api_key: Azure OpenAI API key. If None, will look for AZURE_OPENAI_API_KEY environment variable
            api_version: Azure OpenAI API version
            azure_endpoint: Azure OpenAI endpoint URL
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0-1)
            top_p: Nucleus sampling parameter (0-1)
            **kwargs: Additional parameters to pass to the Azure OpenAI API
        """
        # If model_name not specified, use deployment name
        model_name = model_name or deployment_name
        
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            api_base=azure_endpoint,
            api_version=api_version,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
        
        self.deployment_name = deployment_name
        self._client = None
        self._async_client = None
        
        # Check if OpenAI is installed
        if not HAS_OPENAI:
            raise ImportError(
                "To use the Azure OpenAI adapter, you need to install the OpenAI package: `pip install openai`"
            )
    
    @property
    def client(self):
        """Get the Azure OpenAI client, initializing if needed."""
        if self._client is None:
            kwargs = {
                "api_version": self.api_version
            }
            
            if self.api_key:
                kwargs["api_key"] = self.api_key
            
            if self.api_base:
                kwargs["azure_endpoint"] = self.api_base
            
            self._client = AzureOpenAI(**kwargs)
        return self._client
    
    @property
    def async_client(self):
        """Get the async Azure OpenAI client, initializing if needed."""
        if self._async_client is None:
            kwargs = {
                "api_version": self.api_version
            }
            
            if self.api_key:
                kwargs["api_key"] = self.api_key
            
            if self.api_base:
                kwargs["azure_endpoint"] = self.api_base
            
            self._async_client = AsyncAzureOpenAI(**kwargs)
        return self._async_client
    
    def _format_messages(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """Format messages for the Azure OpenAI API."""
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
        # Format messages for the Azure OpenAI API
        formatted_messages = self._format_messages(messages)
        
        # Build request parameters
        params = {
            "model": self.deployment_name,  # Azure uses deployment name instead of model name
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
        
        # Call the Azure OpenAI API
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
        # Format messages for the Azure OpenAI API
        formatted_messages = self._format_messages(messages)
        
        # Build request parameters
        params = {
            "model": self.deployment_name,  # Azure uses deployment name instead of model name
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
        
        # Call the Azure OpenAI API asynchronously
        response = await self.async_client.chat.completions.create(**params)
        
        # Extract content from the response
        response_message = response.choices[0].message
        response_content = response_message.content or ""
        
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
        params: Dict[str, Any] = {"model": self.deployment_name, "messages": formatted, "tools": [{"type":"function","function":f} for f in functions]}
        if self.max_tokens: params["max_tokens"] = self.max_tokens
        if self.temperature: params["temperature"] = self.temperature
        if self.top_p: params["top_p"] = self.top_p
        if force_function:
            params["tool_choice"] = {"type":"function","function":{"name":force_function}}
        else:
            params["tool_choice"] = "auto"
        params.update(self.additional_params)
        params.update(kwargs)
        response = self.client.chat.completions.create(**params)
        calls = response.choices[0].message.tool_calls
        if not calls:
            return {"function_name":None, "function_arguments":{}, "model_response": response.model_dump()}
        call = calls[0].function
        try: args = json.loads(call.arguments)
        except: args = {"error":"Invalid JSON","raw":call.arguments}
        return {"function_name": call.name, "function_arguments": args, "model_response": response.model_dump()}

    async def afunction_call(
        self,
        messages: List[BaseMessage],
        functions: List[Dict[str, Any]],
        force_function: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        formatted = self._format_messages(messages)
        params: Dict[str, Any] = {"model": self.deployment_name, "messages": formatted, "tools": [{"type":"function","function":f} for f in functions]}
        if self.max_tokens: params["max_tokens"] = self.max_tokens
        if self.temperature: params["temperature"] = self.temperature
        if self.top_p: params["top_p"] = self.top_p
        if force_function:
            params["tool_choice"] = {"type":"function","function":{"name":force_function}}
        else:
            params["tool_choice"] = "auto"
        params.update(self.additional_params)
        params.update(kwargs)
        response = await self.async_client.chat.completions.create(**params)
        calls = response.choices[0].message.tool_calls
        if not calls:
            return {"function_name":None, "function_arguments":{}, "model_response": response.model_dump()}
        call = calls[0].function
        try: args = json.loads(call.arguments)
        except: args = {"error":"Invalid JSON","raw":call.arguments}
        return {"function_name": call.name, "function_arguments": args, "model_response": response.model_dump()}

# Register the adapter
LLMProviderAdapter.register_adapter("azure", AzureOpenAIAdapter)