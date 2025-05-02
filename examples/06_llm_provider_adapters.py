"""Example demonstrating LLM provider adapters with auto-detection.

This example shows how to use the LLM provider adapters with different LLM providers.
It demonstrates:
1. Auto-detecting the provider based on the model name and configuration
2. Explicitly specifying the provider
3. Using provider-specific features

Prerequisites:
- Install the SDK: pip install -e .
- Set up appropriate API keys in environment variables (e.g., OPENAI_API_KEY, ANTHROPIC_API_KEY)
"""

import os

import dotenv
from hotlm_core.schema import HumanMessage, SystemMessage
from hotlm_integrations import create_chat_model
from hotlm_integrations.llms import (
    AnthropicAdapter,
    AzureOpenAIAdapter,
    GroqAdapter,
    HuggingFaceAdapter,
    OpenAIAdapter,
    VertexAIAdapter,
)

# Load environment variables from a .env file if present
dotenv.load_dotenv()

def run_example_with_model(model):
    """Run a simple example with the provided model."""
    print(f"\nUsing model: {model._llm_type}")
    
    # Define messages
    messages = [
        SystemMessage(content="You are a helpful assistant. Keep responses short and to the point."),
        HumanMessage(content="What are the benefits of using different LLM providers?")
    ]
    
    # Generate a response
    try:
        result = model.invoke(messages)
        print(f"Response: {result.content}")
    except Exception as e:
        print(f"Error: {e}")

def main():
    """Run examples with different LLM providers."""
    print("LLM Provider Adapters Example")
    print("============================")
    
    # Example 1: Auto-detect provider based on model name
    if os.environ.get("OPENAI_API_KEY"):
        print("\nExample 1: Auto-detect OpenAI model")
        model = create_chat_model(model_name="gpt-3.5-turbo")
        run_example_with_model(model)
    
    # Example 2: Explicitly specify provider
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("\nExample 2: Explicitly specify Anthropic")
        model = create_chat_model(
            model_name="claude-3-haiku-20240307",
            provider="anthropic"
        )
        run_example_with_model(model)
    
    # Example 3: Azure OpenAI
    azure_key = os.environ.get("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    
    if azure_key and azure_endpoint:
        print("\nExample 3: Azure OpenAI")
        model = create_chat_model(
            model_name="gpt-4",
            provider="azure",
            deployment_name="gpt4",
            api_key=azure_key,
            azure_endpoint=azure_endpoint
        )
        run_example_with_model(model)
    
    # Example 4: Groq (faster inference)
    if os.environ.get("GROQ_API_KEY"):
        print("\nExample 4: Groq")
        model = create_chat_model(
            model_name="llama3-70b-8192",
            provider="groq"
        )
        run_example_with_model(model)
    
    # Example 5: HuggingFace Hub model
    print("\nExample 5: HuggingFace Hub model")
    try:
        # This will attempt to use a small model from HuggingFace
        # Note: This might be slow on CPU and require additional dependencies
        model = create_chat_model(
            model_name="HuggingFaceH4/tiny-random-LlamaForCausalLM",
            provider="huggingface",
            device="cpu",  # Force CPU to avoid CUDA issues
            max_tokens=50  # Limit tokens for speed
        )
        run_example_with_model(model)
    except Exception as e:
        print(f"HuggingFace example failed: {e}")
        print("Make sure you have installed transformers and torch: pip install transformers torch")
    
    # Multiple providers with same interface
    print("\nAvailable adapters:")
    for provider, adapter_class in [
        ("openai", OpenAIAdapter),
        ("azure", AzureOpenAIAdapter),
        ("anthropic", AnthropicAdapter),
        ("groq", GroqAdapter),
        ("vertexai", VertexAIAdapter),
        ("huggingface", HuggingFaceAdapter),
    ]:
        print(f"- {provider}: {adapter_class.provider_name}")

if __name__ == "__main__":
    main()