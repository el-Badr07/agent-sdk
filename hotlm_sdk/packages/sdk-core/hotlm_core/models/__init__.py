"""Base interfaces for language models."""

from .base import BaseLanguageModel
from .chat import BaseChatModel, BaseMultiModalModel

__all__ = [
    "BaseLanguageModel",
    "BaseChatModel",
    "BaseMultiModalModel",
]
