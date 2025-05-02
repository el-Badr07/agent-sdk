from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from hotlm_core.embeddings import BaseEmbeddings
from hotlm_core.runnables import BaseRunnable, RunnableConfig
from hotlm_core.schema import Document
from pydantic import Field


class BaseRetriever(BaseRunnable[str, List[Document]], ABC):
    """Abstract base class for a Document retrieval system.

    A retriever is defined as something that takes a string query as input
    and returns a list of relevant Documents.
    """

    @abstractmethod
    def _get_relevant_documents(
        self, query: str, *, config: Optional[RunnableConfig] = None
    ) -> List[Document]:
        """Retrieve relevant documents for a query.

        Args:
            query: string to find relevant documents for
            config: Optional configuration for the run.

        Returns:
            List of relevant documents
        """

    @abstractmethod
    async def _aget_relevant_documents(
        self, query: str, *, config: Optional[RunnableConfig] = None
    ) -> List[Document]:
        """Asynchronously retrieve relevant documents for a query.

        Args:
            query: string to find relevant documents for
            config: Optional configuration for the run.

        Returns:
            List of relevant documents
        """

    def invoke(self, input: str, config: Optional[RunnableConfig] = None) -> List[Document]:
        return self._get_relevant_documents(input, config=config)

    async def ainvoke(self, input: str, config: Optional[RunnableConfig] = None) -> List[Document]:
        return await self._aget_relevant_documents(input, config=config)


class BaseVectorStoreRetriever(BaseRetriever):
    """Base class for retrieving documents from a VectorStore."""

    vectorstore: BaseVectorStore
    """The underlying VectorStore."""
    k: int = 4
    """Number of documents to return."""
    search_type: str = "similarity"
    """Type of search to perform (e.g., 'similarity', 'mmr')."""
    search_kwargs: dict = Field(default_factory=dict)
    """Additional keyword arguments for the search method."""

    def _get_relevant_documents(
        self, query: str, *, config: Optional[RunnableConfig] = None
    ) -> List[Document]:
        if self.search_type == "similarity":
            return self.vectorstore.similarity_search(query, k=self.k, **self.search_kwargs)
        elif self.search_type == "mmr":
            return self.vectorstore.max_marginal_relevance_search(
                query, k=self.k, **self.search_kwargs
            )
        else:
            raise ValueError(f"Unsupported search type: {self.search_type}")

    async def _aget_relevant_documents(
        self, query: str, *, config: Optional[RunnableConfig] = None
    ) -> List[Document]:
        if self.search_type == "similarity":
            return await self.vectorstore.asimilarity_search(query, k=self.k, **self.search_kwargs)
        elif self.search_type == "mmr":
            return await self.vectorstore.amax_marginal_relevance_search(
                query, k=self.k, **self.search_kwargs
            )
        else:
            raise ValueError(f"Unsupported search type: {self.search_type}")


class BaseVectorStore(ABC):
    """Interface for vector stores."""

    embeddings: Optional[BaseEmbeddings] = None # Embeddings can be optional if adding pre-embedded vectors

    @abstractmethod
    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Run more documents through the embeddings and add to the vectorstore.

        Args:
            documents: Documents to add to the vectorstore.
            **kwargs: Other keyword arguments specific to the vectorstore implementation.

        Returns:
            List of IDs of the added documents.
        """
        pass

    async def aadd_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Asynchronously run more documents through the embeddings and add to the vectorstore.

        Args:
            documents: Documents to add to the vectorstore.
            **kwargs: Other keyword arguments specific to the vectorstore implementation.

        Returns:
            List of IDs of the added documents.
        """
        # Default sync implementation (subclasses should override)
        return self.add_documents(documents, **kwargs)

    # Optional: Add methods for adding texts directly
    def add_texts(
        self, texts: Iterable[str], metadatas: Optional[List[dict]] = None, **kwargs: Any
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore."""
        documents = [Document(page_content=t) for t in texts]
        if metadatas:
            if len(documents) != len(metadatas):
                raise ValueError("Number of texts and metadatas must match.")
            for i, m in enumerate(metadatas):
                documents[i].metadata = m
        return self.add_documents(documents, **kwargs)

    async def aadd_texts(
        self, texts: Iterable[str], metadatas: Optional[List[dict]] = None, **kwargs: Any
    ) -> List[str]:
        """Asynchronously run more texts through the embeddings and add to the vectorstore."""
        documents = [Document(page_content=t) for t in texts]
        if metadatas:
            if len(documents) != len(metadatas):
                raise ValueError("Number of texts and metadatas must match.")
            for i, m in enumerate(metadatas):
                documents[i].metadata = m
        return await self.aadd_documents(documents, **kwargs)

    @abstractmethod
    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            **kwargs: Other keyword arguments specific to the vectorstore implementation.

        Returns:
            List of Documents most similar to the query.
        """
        pass

    async def asimilarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Asynchronously return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            **kwargs: Other keyword arguments specific to the vectorstore implementation.

        Returns:
            List of Documents most similar to the query.
        """
        # Default sync implementation (subclasses should override)
        return self.similarity_search(query, k=k, **kwargs)

    # Optional: Add other search methods
    def max_marginal_relevance_search(
        self, query: str, k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5, **kwargs: Any
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity metric penalty, where 0 corresponds
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            **kwargs: Other keyword arguments specific to the vectorstore implementation.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        raise NotImplementedError("max_marginal_relevance_search not implemented for this vector store.")

    async def amax_marginal_relevance_search(
        self, query: str, k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5, **kwargs: Any
    ) -> List[Document]:
        """Asynchronously return docs selected using the maximal marginal relevance.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity metric penalty, where 0 corresponds
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            **kwargs: Other keyword arguments specific to the vectorstore implementation.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        raise NotImplementedError("amax_marginal_relevance_search not implemented for this vector store.")

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        """Return VectorStoreRetriever initialized from this VectorStore.

        Args:
            **kwargs: Keyword arguments to pass to the retriever.
                      Defaults to `search_type="similarity"`.

        Returns:
            VectorStoreRetriever
        """
        return BaseVectorStoreRetriever(vectorstore=self, **kwargs)
