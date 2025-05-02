"""In-memory chat message history."""

from typing import Any, List

# from hotlm_core.schema import BaseMessage # Assuming this exists
# from hotlm_core.memory import BaseChatMessageHistory # Assuming this exists


class InMemoryChatMessageHistory:  # Replace with inheritance from BaseChatMessageHistory later
    """In memory implementation of chat message history."""

    def __init__(self):
        self._messages: List[Any] = []  # Replace Any with BaseMessage later

    @property
    def messages(self) -> List[Any]:  # Replace Any with BaseMessage later
        """Retrieve the messages."""
        return self._messages

    def add_message(self, message: Any) -> None:  # Replace Any with BaseMessage later
        """Add a message to the history."""
        self._messages.append(message)

    def add_user_message(self, message: str) -> None:
        """Add a user message to the history."""
        # self.add_message(HumanMessage(content=message)) # Use core schema later
        self.add_message({"role": "user", "content": message})

    def add_ai_message(self, message: str) -> None:
        """Add an AI message to the history."""
        # self.add_message(AIMessage(content=message)) # Use core schema later
        self.add_message({"role": "ai", "content": message})

    def clear(self) -> None:
        """Clear the history."""
        self._messages = []
