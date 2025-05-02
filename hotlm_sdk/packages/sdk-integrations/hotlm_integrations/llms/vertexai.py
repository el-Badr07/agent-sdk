"""Google Vertex AI LLM Adapter."""

import os
import warnings
from typing import Any, ClassVar, Dict, List, Optional, Union

try:
    import vertexai
    from vertexai.generative_models import ChatSession, GenerativeModel
    HAS_VERTEXAI = True
except ImportError:
    HAS_VERTEXAI = False
    warnings.warn("Vertex AI package not found. Install it with 'pip install google-cloud-aiplatform'.")

from hotlm_core.schema import AIMessage, BaseMessage, ChatResult, Generation

from .base import LLMProviderAdapter


class VertexAIAdapter(LLMProviderAdapter):
    """Adapter for Google Vertex AI models."""

    provider_name: ClassVar[str] = "vertexai"
    
    def __init__(
        self,
        model_name: str = "gemini-1.5-pro",
        project_id: Optional[str] = None,
        region: str = "us-central1",
        credentials: Optional[Any] = None,
        max_tokens: Optional[int] = 1024,
        temperature: float = 0.7,
        top_p: float = 1.0,
        **kwargs
    ):
        """Initialize Vertex AI Adapter.
        
        Args:
            model_name: The name of the model (e.g., "gemini-1.5-pro")
            project_id: GCP project ID
            region: GCP region
            credentials: Google Cloud credentials object
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0-1)
            top_p: Nucleus sampling parameter (0-1)
            **kwargs: Additional parameters to pass to the Vertex AI API
        """
        super().__init__(
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
        
        self.project_id = project_id
        self.region = region
        self.credentials = credentials
        self._client = None
        self._initialized = False
        
        # Check if Vertex AI is installed
        if not HAS_VERTEXAI:
            raise ImportError(
                "To use the Vertex AI adapter, you need to install the Vertex AI package: "
                "`pip install google-cloud-aiplatform`"
            )
    
    def _initialize_vertexai(self):
        """Initialize the Vertex AI client."""
        if not self._initialized:
            kwargs = {}
            if self.project_id:
                kwargs["project"] = self.project_id
            if self.region:
                kwargs["location"] = self.region
            if self.credentials:
                kwargs["credentials"] = self.credentials
                
            # Initialize Vertex AI
            vertexai.init(**kwargs)
            self._initialized = True
    
    @property
    def client(self):
        """Get the Vertex AI model client."""
        self._initialize_vertexai()
        if self._client is None:
            generation_config = {
                "temperature": self.temperature,
                "top_p": self.top_p,
            }
            
            if self.max_tokens:
                generation_config["max_output_tokens"] = self.max_tokens
                
            # Add additional params if provided
            generation_config.update(self.additional_params)
            
            self._client = GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config
            )
        return self._client
    
    def _format_content_list(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """Format messages for the Vertex AI API content list."""
        content_list = []
        for msg in messages:
            role = msg.type.value
            if role == "ai":
                role = "model"
            elif role == "human":
                role = "user"
            elif role == "system":
                # For gemini models, system prompts are handled separately
                # We'll keep them in the content list for now
                role = "user"
                
            content_list.append({
                "role": role,
                "parts": [{"text": msg.content}]
            })
            
        return content_list

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Invoke the model with a list of messages."""
        # Format messages for Vertex AI
        content_list = self._format_content_list(messages)
        
        # Create a chat session
        chat = self.client.start_chat(history=content_list[:-1])
        
        # Get the last message (to send)
        last_message = content_list[-1]["parts"][0]["text"]
        
        # Call the Vertex AI API
        response = chat.send_message(last_message)
        
        # Extract content from the response
        response_content = response.text
        
        # Create a Generation with the response
        generation = Generation(message=AIMessage(content=response_content))
        
        # Create a dictionary representation of the response for llm_output
        response_dict = {
            "text": response.text,
            "model": self.model_name,
        }
        
        # Return the ChatResult
        return ChatResult(generations=[generation], llm_output=response_dict)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Asynchronously invoke the model."""
        # For Vertex AI, we'll just use the sync implementation
        # since the library doesn't provide native async support
        return self._generate(messages, stop, **kwargs)

# Register the adapter
LLMProviderAdapter.register_adapter("vertexai", VertexAIAdapter)