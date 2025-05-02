from abc import ABC
from typing import Any, Dict, List, Optional

from hotlm_core.runnables import BaseRunnable, RunnableConfig
from hotlm_core.schema import AIMessage, BaseMessage


class BaseChatModel(BaseRunnable[List[BaseMessage], AIMessage], ABC):
    """Abstract base class for chat models.

    Takes a list of messages as input and returns an AI message.
    """

    # TODO: Add methods for token counting, specific model properties, etc.
    pass


class BaseMultiModalModel(BaseChatModel, ABC):
    """Abstract base class for multi-modal models.

    Extends BaseChatModel to explicitly handle multi-modal inputs
    (e.g., images within messages).
    """

    # TODO: Define how multi-modal content is represented in input messages
    # TODO: Define how multi-modal output might be represented
    pass
