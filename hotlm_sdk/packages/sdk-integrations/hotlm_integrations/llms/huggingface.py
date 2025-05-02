"""HuggingFace LLM Adapter."""

import asyncio
import os
import warnings
from typing import Any, ClassVar, Dict, List, Optional

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    warnings.warn("Transformers package not found. Install it with 'pip install transformers torch'.")

from hotlm_core.schema import AIMessage, BaseMessage, ChatResult, Generation

from .base import LLMProviderAdapter


class HuggingFaceAdapter(LLMProviderAdapter):
    """Adapter for HuggingFace Hub models or local Transformers."""

    provider_name: ClassVar[str] = "huggingface"
    
    def __init__(
        self,
        model_name: str,
        repo_id: Optional[str] = None,
        model_path: Optional[str] = None,
        task: str = "text-generation",
        device: str = "auto",
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        **kwargs
    ):
        """Initialize HuggingFace Adapter.
        
        Args:
            model_name: The name/identifier of the model
            repo_id: HuggingFace Hub repository ID (e.g., "mistralai/Mistral-7B-v0.1")
            model_path: Path to local model files
            task: The task for the model (e.g., "text-generation")
            device: Device to run the model on ("cpu", "cuda", "auto")
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0-1)
            top_p: Nucleus sampling parameter (0-1)
            **kwargs: Additional parameters to pass to the model
        """
        # Use repo_id as model_name if not provided
        if not model_name and repo_id:
            model_name = repo_id
            
        super().__init__(
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
        
        if not repo_id and not model_path:
            raise ValueError("Either repo_id (for Hub) or model_path (for local) must be provided.")
        
        self.repo_id = repo_id
        self.model_path = model_path
        self.task = task
        self.device = device
        
        self._model = None
        self._tokenizer = None
        
        # Check if transformers is installed
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "To use the HuggingFace adapter, you need to install the transformers package: "
                "`pip install transformers torch`"
            )
    
    @property
    def model(self):
        """Get the HuggingFace model, loading it if needed."""
        if self._model is None:
            model_identifier = self.repo_id or self.model_path
            
            # Load the model
            if self.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device
                
            try:
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_identifier,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map=device,
                    **self.additional_params.get("model_kwargs", {})
                )
            except Exception as e:
                raise ValueError(f"Failed to load HuggingFace model {model_identifier}: {e}")
                
        return self._model
    
    @property
    def tokenizer(self):
        """Get the HuggingFace tokenizer, loading it if needed."""
        if self._tokenizer is None:
            model_identifier = self.repo_id or self.model_path
            
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    model_identifier,
                    **self.additional_params.get("tokenizer_kwargs", {})
                )
            except Exception as e:
                raise ValueError(f"Failed to load HuggingFace tokenizer {model_identifier}: {e}")
                
        return self._tokenizer
    
    def _format_prompt(self, messages: List[BaseMessage]) -> str:
        """Format messages into a prompt for the model."""
        # Try to use apply_chat_template if available
        try:
            if hasattr(self.tokenizer, "apply_chat_template"):
                # Convert our messages to the format expected by the tokenizer
                formatted_messages = []
                for msg in messages:
                    role = msg.type.value
                    if role == "ai":
                        role = "assistant"
                    elif role == "human":
                        role = "user"
                    formatted_messages.append({"role": role, "content": msg.content})
                
                # Apply the chat template
                prompt = self.tokenizer.apply_chat_template(
                    formatted_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                return prompt
        except Exception:
            pass
            
        # Fallback to basic prompt formatting
        formatted_prompt = ""
        for msg in messages:
            role = msg.type.value.capitalize()
            formatted_prompt += f"{role}: {msg.content}\n"
        formatted_prompt += "AI: "
        return formatted_prompt

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Invoke the model with a list of messages."""
        # Format messages into a prompt
        prompt = self._format_prompt(messages)
        
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Build generation parameters
        generation_kwargs = {
            "max_new_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": self.temperature > 0,
        }
        
        # Add stop sequences if provided
        # Note: This is not directly supported by all transformers, so we handle it in post-processing
        
        # Add any additional parameters
        generation_kwargs.update(self.additional_params)
        
        # Override with any parameters passed to this method
        generation_kwargs.update(kwargs)
        
        # Generate the response
        with torch.no_grad():
            output_ids = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.get('attention_mask', None),
                **generation_kwargs
            )
        
        # Decode the output
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Remove the input prompt from the output
        if output_text.startswith(prompt):
            response_content = output_text[len(prompt):].strip()
        else:
            response_content = output_text.strip()
        
        # Apply stop sequences if provided
        if stop:
            for stop_seq in stop:
                if stop_seq in response_content:
                    response_content = response_content[:response_content.index(stop_seq)]
        
        # Create a Generation with the response
        generation = Generation(message=AIMessage(content=response_content))
        
        # Return the ChatResult
        return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Asynchronously invoke the model."""
        # Since transformers doesn't have native async support,
        # we run the synchronous code in a separate thread
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self._generate(messages, stop, **kwargs)
        )

# Register the adapter
LLMProviderAdapter.register_adapter("huggingface", HuggingFaceAdapter)

