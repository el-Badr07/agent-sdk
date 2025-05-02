from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional

from hotlm_core.schema import Document


# Default length function using len()
def _default_length_func(text: str) -> int:
    return len(text)

class BaseTextSplitter(ABC):
    """Interface for splitting text into chunks."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = _default_length_func,
        keep_separator: bool = False,
        strip_whitespace: bool = True,
    ):
        """Initialize the text splitter.

        Args:
            chunk_size: Maximum size of chunks to return (based on length_function).
            chunk_overlap: Overlap in characters between chunks.
            length_function: Function that measures text length. Defaults to len().
            keep_separator: Whether to keep the separators in the chunks.
            strip_whitespace: If True, strips whitespace from the start and end of chunks.
        """
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"Chunk overlap ({chunk_overlap}) cannot be larger than chunk size ({chunk_size})"
            )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function
        self._keep_separator = keep_separator
        self._strip_whitespace = strip_whitespace

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        pass

    def create_documents(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[Document]:
        """Create documents from a list of texts."""
        _metadatas = metadatas or [{}] * len(texts)
        if len(texts) != len(_metadatas):
            raise ValueError("Number of texts and metadatas must be the same.")

        documents = []
        for i, text in enumerate(texts):
            for chunk in self.split_text(text):
                new_metadata = _metadatas[i].copy() # Avoid modifying original metadata
                # Potentially add chunk-specific metadata here if needed
                if self._strip_whitespace:
                    chunk = chunk.strip()
                if not chunk: # Skip empty chunks
                    continue
                documents.append(Document(page_content=chunk, metadata=new_metadata))
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks, preserving metadata."""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.create_documents(texts, metadatas=metadatas)

    # TODO: Add async versions if needed (asplit_text, asplit_documents)
