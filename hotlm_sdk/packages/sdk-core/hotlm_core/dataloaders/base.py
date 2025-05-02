from abc import ABC, abstractmethod
from typing import Iterator, List

from hotlm_core.schema.documents import Document


class BaseDocumentLoader(ABC):
    """Abstract base class for document loaders.

    Implementers should subclass this class and override the `load` or `lazy_load`
    methods to define how to load documents from a specific source.
    """

    @abstractmethod
    def load(self) -> List[Document]:
        """Load data into Document objects.

        This method should be implemented by subclasses to load all documents
        from the source into a list.

        Returns:
            A list of Document objects.
        """
        pass

    @abstractmethod
    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """A lazy loader for Documents.

        This method should be implemented by subclasses to return an iterator
        that yields Documents one by one.

        Returns:
            An iterator of Document objects.
        """
        pass
