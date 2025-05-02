"""Base interfaces for vector stores and retrievers."""

from .base import BaseRetriever, BaseVectorStore, BaseVectorStoreRetriever

__all__ = [
    "BaseVectorStore",
    "BaseRetriever",
    "BaseVectorStoreRetriever",
]
