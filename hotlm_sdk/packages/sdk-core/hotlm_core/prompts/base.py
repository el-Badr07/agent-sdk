from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from hotlm_core.runnables import BaseRunnable, RunnableConfig
from hotlm_core.schema import BaseMessage


class BasePromptTemplate(BaseRunnable[Dict[str, Any], str], ABC):
    """Base interface for string prompt templates."""

    @abstractmethod
    def format(self, **kwargs: Any) -> str:
        """Format the template with the given keyword arguments."""
        pass

    def invoke(self, input: Dict[str, Any], config: Optional[RunnableConfig] = None) -> str:
        return self.format(**input)

    async def ainvoke(self, input: Dict[str, Any], config: Optional[RunnableConfig] = None) -> str:
        # Default sync implementation for async
        return self.format(**input)

class BaseChatPromptTemplate(BaseRunnable[Dict[str, Any], List[BaseMessage]], ABC):
    """Base interface for chat prompt templates."""

    @abstractmethod
    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        """Format the template into a list of chat messages."""
        pass

    def invoke(self, input: Dict[str, Any], config: Optional[RunnableConfig] = None) -> List[BaseMessage]:
        return self.format_messages(**input)

    async def ainvoke(self, input: Dict[str, Any], config: Optional[RunnableConfig] = None) -> List[BaseMessage]:
        # Default sync implementation for async
        return self.format_messages(**input)

# TODO: Add MessagesPlaceholder concept if needed later
