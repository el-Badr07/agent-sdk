import logging
import uuid
from typing import Any, Dict, List, Optional, Sequence, Union, cast

from hotlm_core.embeddings import BaseEmbeddings
from hotlm_core.schema.documents import Document
from hotlm_core.vectorstores.base import BaseVectorStore

logger = logging.getLogger(__name__)

class PineconeVectorStore(BaseVectorStore):
    """Vector store using Pinecone (https://www.pinecone.io) as a backend."""
    
    def __init__(
        self,
        embeddings: BaseEmbeddings,
        index_name: str,
        namespace: str = "",
        pinecone_api_key: Optional[str] = None,
        pinecone_environment: Optional[str] = None,
        metadata_fields: Optional[List[str]] = None,
    ) -> None:
        """Initialize with Pinecone client.

        Args:
            embeddings: Embeddings model to create vectors.
            index_name: Name of the Pinecone index to use.
            namespace: Pinecone namespace, useful for multitenancy.
            pinecone_api_key: Pinecone API key. If not provided, will look for 
                              PINECONE_API_KEY env var.
            pinecone_environment: Pinecone environment. If not provided, will look for
                                 PINECONE_ENVIRONMENT env var.
            metadata_fields: Fields to store in Pinecone metadata.
        """
        self.embeddings = embeddings
        self.index_name = index_name
        self.namespace = namespace
        self.metadata_fields = metadata_fields or []
        
        try:
            import pinecone
        except ImportError:
            raise ImportError(
                "Could not import pinecone. Please install with 'pip install pinecone-client'."
            )
            
        self.pinecone_client = self._init_pinecone_client(
            api_key=pinecone_api_key, 
            environment=pinecone_environment
        )
        self.index = self._get_or_create_index()
        logger.debug(f"Initialized PineconeVectorStore with index: {self.index_name}")
    
    def _init_pinecone_client(self, api_key: Optional[str] = None, environment: Optional[str] = None):
        """Initialize the Pinecone client."""
        import os

        import pinecone
        
        api_key = api_key or os.environ.get("PINECONE_API_KEY")
        environment = environment or os.environ.get("PINECONE_ENVIRONMENT")
        
        if not api_key or not environment:
            raise ValueError(
                "Please provide a Pinecone API key and environment, either as arguments "
                "or by setting the PINECONE_API_KEY and PINECONE_ENVIRONMENT environment variables."
            )
        
        # Initialize pinecone client
        pinecone.init(api_key=api_key, environment=environment)
        return pinecone
    
    def _get_or_create_index(self):
        """Get the Pinecone index."""
        import pinecone

        # Check if index exists
        if self.index_name not in self.pinecone_client.list_indexes():
            logger.warning(
                f"Index '{self.index_name}' does not exist in Pinecone. "
                f"Please create it first. Available indexes: {self.pinecone_client.list_indexes()}"
            )
            raise ValueError(f"Index '{self.index_name}' does not exist in Pinecone.")
        
        # Get the index
        return self.pinecone_client.Index(self.index_name)
    
    def _create_metadata(self, document: Document) -> Dict[str, Any]:
        """Create metadata from document."""
        metadata = {}
        # Add page content under text key for potential filtering
        metadata["text"] = document.page_content
        
        # Add all metadata from document
        for key, value in document.metadata.items():
            if key in self.metadata_fields or not self.metadata_fields:
                metadata[key] = value
        
        return metadata
            
    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Add documents to the vector store.
        
        Args:
            documents: List of Documents to add.
            
        Returns:
            List of document IDs added.
        """
        logger.debug(f"Adding {len(documents)} documents to Pinecone")
        
        # Use provided IDs or generate new ones
        ids = kwargs.get("ids", [str(uuid.uuid4()) for _ in range(len(documents))])
        
        # Get document texts and create embeddings
        texts = [doc.page_content for doc in documents]
        embeddings = self.embeddings.embed_documents(texts)
        
        # Create metadata for each document
        metadatas = [self._create_metadata(doc) for doc in documents]
        
        # Prepare records for Pinecone upsert
        records = []
        for i, (doc_id, embedding, metadata) in enumerate(zip(ids, embeddings, metadatas)):
            records.append({
                "id": doc_id,
                "values": embedding,
                "metadata": metadata
            })
            
        # Upsert in batches to avoid API limits
        batch_size = 100
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            self.index.upsert(
                vectors=batch,
                namespace=self.namespace
            )
        
        return ids
    
    def _convert_metadata_to_document(self, result: Dict[str, Any]) -> Document:
        """Convert metadata from a query result to a Document."""
        metadata = result.get("metadata", {})
        text = metadata.pop("text", "")  # Extract the text and remove from metadata
        
        return Document(page_content=text, metadata=metadata)
    
    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        """Find similar documents to the query string.
        
        Args:
            query: String to search for.
            k: Number of documents to return.
            
        Returns:
            List of Documents most similar to the query.
        """
        logger.debug(f"Searching Pinecone for: '{query}' with k={k}")
        
        # Get embedding for the query
        embedding = self.embeddings.embed_query(query)
        
        # Search Pinecone
        results = self.index.query(
            vector=embedding,
            top_k=k,
            namespace=self.namespace,
            include_metadata=True
        )
        
        # Convert results to Documents
        documents = []
        for match in results.get("matches", []):
            doc = self._convert_metadata_to_document(match)
            documents.append(doc)
            
        return documents
    
    async def asimilarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        """Asynchronous similarity search (uses synchronous version for now).
        
        Args:
            query: String to search for.
            k: Number of documents to return.
            
        Returns:
            List of Documents most similar to the query.
        """
        # For now, just use the synchronous version
        return self.similarity_search(query, k, **kwargs)
    
    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> None:
        """Delete documents from the vector store.
        
        Args:
            ids: List of document IDs to delete.
        """
        if not ids:
            logger.warning("No document IDs provided for deletion")
            return
            
        logger.debug(f"Deleting {len(ids)} documents from Pinecone")
        self.index.delete(ids=ids, namespace=self.namespace)