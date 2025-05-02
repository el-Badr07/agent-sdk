from abc import ABC, abstractmethod
from typing import Any, Dict, List

from hotlm_core.schema import BaseMessage


class BaseChatMessageHistory(ABC):
    """Abstract interface for storing chat message history."""

    @abstractmethod
    def get_messages(self) -> List[BaseMessage]:
        """Retrieve the chat message history."""
        pass

    @abstractmethod
    async def aget_messages(self) -> List[BaseMessage]:
        """Asynchronously retrieve the chat message history."""
        pass

    @abstractmethod
    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the history."""
        pass

    @abstractmethod
    async def aadd_message(self, message: BaseMessage) -> None:
        """Asynchronously add a message to the history."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all messages from the history."""
        pass

    @abstractmethod
    async def aclear(self) -> None:
        """Asynchronously clear all messages from the history."""
        pass

# Basic in-memory implementation
class InMemoryChatMessageHistory(BaseChatMessageHistory):
    """Basic in-memory implementation of chat message history."""

    def __init__(self):
        self._messages: List[BaseMessage] = []

    def get_messages(self) -> List[BaseMessage]:
        return list(self._messages) # Return a copy

    async def aget_messages(self) -> List[BaseMessage]:
        return self.get_messages()

    def add_message(self, message: BaseMessage) -> None:
        self._messages.append(message)

    async def aadd_message(self, message: BaseMessage) -> None:
        self.add_message(message)

    def clear(self) -> None:
        self._messages = []

    async def aclear(self) -> None:
        self.clear()
