from typing import Any, List, Optional

import faiss
import numpy as np
from hotlm_core.embeddings import BaseEmbeddings
from hotlm_core.schema.documents import Document
from hotlm_core.vectorstores.base import BaseVectorStore


class FaissVectorStore(BaseVectorStore):
    """Vector store backed by FAISS for efficient similarity search."""

    def __init__(
        self,
        embeddings: BaseEmbeddings,
        dimension: int,
        index: Optional[faiss.Index] = None,
    ) -> None:
        """Initialize FaissVectorStore.

        Args:
            embeddings: Embeddings model to encode documents.
            dimension: Dimensionality of embeddings.
            index: Optional existing FAISS index. If None, an IndexFlatL2 is created.
        """
        self.embeddings = embeddings
        self.dimension = dimension
        self.index = index or faiss.IndexFlatL2(dimension)
        self.doc_ids: List[str] = []

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Embed and add documents to the FAISS index."""
        texts = [doc.page_content for doc in documents]
        vectors = self.embeddings.embed_documents(texts)
        arr = np.vstack(vectors).astype("float32")
        self.index.add(arr)
        ids = []
        for doc in documents:
            doc_id = kwargs.get("ids") or str(len(self.doc_ids))
            self.doc_ids.append(doc_id)
            ids.append(doc_id)
        return ids

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Perform a similarity search for the query string."""
        q_vec = np.array(self.embeddings.embed_query(query), dtype="float32").reshape(
            1, -1
        )
        distances, indices = self.index.search(q_vec, k)
        results: List[Document] = []
        for idx in indices[0]:
            if idx < len(self.doc_ids):
                # In real use case, store metadata mapping ids to docs
                results.append(
                    Document(page_content="", metadata={"id": self.doc_ids[idx]})
                )
        return results

    async def asimilarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Asynchronous similarity search."""
        return self.similarity_search(query, k=k, **kwargs)
