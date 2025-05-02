from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddings(ABC):
    """Interface for embedding models."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        pass

    # Optional async versions
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously embed search docs."""
        # Default sync implementation (subclasses should override)
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronously embed query text."""
        # Default sync implementation (subclasses should override)
        return self.embed_query(text)
