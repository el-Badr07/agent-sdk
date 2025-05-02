"""Base interface for LLM provider adapters."""

import asyncio
from abc import abstractmethod
from typing import (
    Any,
    AsyncIterator,
    Callable,
    ClassVar,
    Dict,
    Iterator,
    List,
    Optional,
    Type,
    Union,
)

from hotlm_core.models.chat import BaseChatModel
from hotlm_core.schema import AIMessage, BaseMessage, ChatResult


class LLMProviderAdapter(BaseChatModel):
    """Base class for LLM provider adapters.
    
    This class defines the common interface and behavior for all LLM provider adapters.
    """
    
    # Class variable that should be set by each provider adapter
    provider_name: ClassVar[str]
    
    # Registry of adapter classes by provider name
    _adapter_registry: ClassVar[Dict[str, Type['LLMProviderAdapter']]] = {}
    
    def __init__(
        self, 
        model_name: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        **kwargs
    ):
        """Initialize the LLM provider adapter.
        
        Args:
            model_name: The name of the model to use
            api_key: API key for the provider
            api_base: Base URL for API requests
            api_version: API version to use
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0-1)
            top_p: Nucleus sampling parameter (0-1)
            **kwargs: Additional provider-specific parameters
        """
        super().__init__()
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self.api_version = api_version
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.additional_params = kwargs
        
    @abstractmethod
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response from the LLM based on the messages."""
        pass
        
    @abstractmethod
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response from the LLM based on the messages asynchronously."""
        pass
        
    def stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[AIMessage]:
        """Stream a response from the LLM token by token.
        
        Args:
            messages: The messages to send to the LLM
            stop: Optional list of stop sequences
            **kwargs: Additional parameters for the LLM
            
        Yields:
            AIMessage: Messages with progressive content as tokens are generated
        """
        streaming_option = kwargs.pop("streaming", True)
        if not streaming_option:
            # If streaming is explicitly disabled, just use regular generation
            result = self._generate(messages, stop, **kwargs)
            if result.generations:
                yield result.generations[0].message
            return
            
        # Implementation should be overridden by providers with native streaming
        # This is a simple fallback that just returns the complete response
        result = self._generate(messages, stop, **kwargs)
        if result.generations:
            yield result.generations[0].message
            
    async def astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[AIMessage]:
        """Stream a response from the LLM token by token asynchronously.
        
        Args:
            messages: The messages to send to the LLM
            stop: Optional list of stop sequences
            **kwargs: Additional parameters for the LLM
            
        Yields:
            AIMessage: Messages with progressive content as tokens are generated
        """
        streaming_option = kwargs.pop("streaming", True)
        if not streaming_option:
            # If streaming is explicitly disabled, just use regular generation
            result = await self._agenerate(messages, stop, **kwargs)
            if result.generations:
                yield result.generations[0].message
            return
            
        # Implementation should be overridden by providers with native streaming
        # This is a simple fallback that just returns the complete response
        result = await self._agenerate(messages, stop, **kwargs)
        if result.generations:
            yield result.generations[0].message
    
    def function_call(
        self,
        messages: List[BaseMessage],
        functions: List[Dict[str, Any]],
        force_function: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Call a function based on the LLM's decision.
        
        Args:
            messages: The messages to send to the LLM
            functions: List of function definitions
            force_function: Force the LLM to call this specific function
            **kwargs: Additional parameters for the LLM
            
        Returns:
            A dictionary with function_name and function_arguments
            
        Note:
            This is a default implementation that should be overridden by providers
            that support function calling natively.
        """
        raise NotImplementedError(
            f"Function calling is not implemented for {self.provider_name} adapter."
        )
    
    async def afunction_call(
        self,
        messages: List[BaseMessage],
        functions: List[Dict[str, Any]],
        force_function: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Call a function based on the LLM's decision asynchronously.
        
        Args:
            messages: The messages to send to the LLM
            functions: List of function definitions
            force_function: Force the LLM to call this specific function
            **kwargs: Additional parameters for the LLM
            
        Returns:
            A dictionary with function_name and function_arguments
            
        Note:
            This is a default implementation that should be overridden by providers
            that support function calling natively.
        """
        # Default implementation runs the sync version in an executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self.function_call(messages, functions, force_function, **kwargs)
        )

    def supports_function_calling(self) -> bool:
        """Check if this adapter supports function calling.
        
        Returns:
            True if the adapter supports function calling, False otherwise
        """
        return hasattr(self, "function_call") and self.function_call.__func__ is not LLMProviderAdapter.function_call
    
    def supports_vision(self) -> bool:
        """Check if this adapter supports vision/image input.
        
        Returns:
            True if the adapter supports vision input, False otherwise
        """
        return False
        
    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return f"{self.provider_name}-{self.model_name}"
    
    @classmethod
    def register_adapter(cls, provider_name: str, adapter_class: Type['LLMProviderAdapter']) -> None:
        """Register a provider adapter class.
        
        Args:
            provider_name: The name of the provider
            adapter_class: The adapter class for the provider
        """
        cls._adapter_registry[provider_name.lower()] = adapter_class
    
    @classmethod
    def get_adapter_class(cls, provider_name: str) -> Optional[Type['LLMProviderAdapter']]:
        """Get the adapter class for a provider.
        
        Args:
            provider_name: The name of the provider
            
        Returns:
            The adapter class for the provider, or None if not found
        """
        return cls._adapter_registry.get(provider_name.lower())