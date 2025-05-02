"""Conversation Buffer Memory Layer."""

from typing import Any, Dict, List

from hotlm_core.memory.base import BaseMemory
from hotlm_core.memory.history import BaseChatMessageHistory, InMemoryChatMessageHistory
from hotlm_core.schema import AIMessage, BaseMessage, HumanMessage, get_buffer_string


class ConversationBufferMemory(BaseMemory):
    """Buffer for storing conversation history.

    Uses a BaseChatMessageHistory store underneath.
    """

    chat_memory: BaseChatMessageHistory
    memory_key: str = "history"  # Input key for memory variable
    input_key: str | None = None # Key for input to be saved
    output_key: str | None = None # Key for output to be saved
    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    return_messages: bool = False # Return history as BaseMessages or string

    def __init__(self,
                 chat_memory: BaseChatMessageHistory | None = None,
                 memory_key="history",
                 input_key: str | None = None,
                 output_key: str | None = None,
                 return_messages=False,
                 human_prefix: str = "Human",
                 ai_prefix: str = "AI",
                 **kwargs): # Allow passing other BaseMemory args if any
        super().__init__(**kwargs)
        # Use InMemoryChatMessageHistory from sdk-core by default
        self.chat_memory = chat_memory or InMemoryChatMessageHistory()
        self.memory_key = memory_key
        self.input_key = input_key
        self.output_key = output_key
        self.return_messages = return_messages
        self.human_prefix = human_prefix
        self.ai_prefix = ai_prefix

    @property
    def memory_variables(self) -> List[str]:
        """The list of keys returned by this memory."""
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return history buffer based on the initialization parameters."""
        messages = self.chat_memory.messages
        if self.return_messages:
            # Return the actual BaseMessage objects
            return {self.memory_key: messages}
        else:
            # Return a formatted string representation
            buffer_string = get_buffer_string(
                messages,
                human_prefix=self.human_prefix,
                ai_prefix=self.ai_prefix,
            )
            return {self.memory_key: buffer_string}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer.

        Determines which input and output values to save based on input_key and output_key.
        If keys are None, it attempts to save the 'input' and 'output' dictionary keys
        or converts the entire dicts to strings if those keys aren't found.
        Input values are saved as HumanMessage, output values as AIMessage.
        """
        # Determine the input message content
        if self.input_key is None:
            # Try to find a key that isn't the memory key
            prompt_input_key = next((k for k in inputs if k != self.memory_key), None)
            if prompt_input_key is None:
                 # Fallback: stringify the whole input dict if no other key found
                 input_content = str(inputs)
            else:
                 input_content = inputs[prompt_input_key]
        else:
            input_content = inputs.get(self.input_key)
            if input_content is None:
                # Log a warning or handle as needed if the specified key isn't present
                print(f"Warning: Input key '{self.input_key}' not found in inputs: {inputs}. Context not saved.")
                return # Don't save if expected input is missing

        # Determine the output message content
        if self.output_key is None:
            # If only one output key, assume that's the one to save
            if len(outputs) == 1:
                output_key = list(outputs.keys())[0]
                output_content = outputs[output_key]
            else:
                # Fallback: stringify the whole output dict if logic is ambiguous
                output_content = str(outputs)
        else:
            output_content = outputs.get(self.output_key)
            if output_content is None:
                # Log a warning or handle as needed
                print(f"Warning: Output key '{self.output_key}' not found in outputs: {outputs}. Context not saved.")
                return # Don't save if expected output is missing

        # Save to chat history using core schema types
        self.chat_memory.add_message(HumanMessage(content=str(input_content)))
        self.chat_memory.add_message(AIMessage(content=str(output_content)))

    def clear(self) -> None:
        """Clear memory contents."""
        self.chat_memory.clear()
        print("Memory cleared.")

