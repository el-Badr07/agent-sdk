from abc import ABC
from typing import Any, Dict, List, Optional

from hotlm_core.runnables import BaseRunnable, RunnableConfig
from hotlm_core.schema import BaseMessage


class BaseLanguageModel(BaseRunnable[str, str], ABC):
    """Abstract base class for language models or LLMs.

    Exposes a simple string-in, string-out interface.
    """

    # TODO: Add methods for token counting, specific model properties, etc.
    pass
