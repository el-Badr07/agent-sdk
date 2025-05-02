import os
import uuid
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from hotlm_core.embeddings import BaseEmbeddings
from hotlm_core.schema.documents import Document
from hotlm_core.vectorstores.pinecone import PineconeVectorStore


class MockEmbeddings(BaseEmbeddings):
    """Mock embeddings for testing."""
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]
    
    def embed_query(self, text: str) -> List[float]:
        return [0.1, 0.2, 0.3]


@pytest.fixture
def mock_pinecone():
    """Create a mocked Pinecone client."""
    with patch("pinecone") as mock:
        # Set up the mock for the index
        mock_index = MagicMock()
        mock.Index.return_value = mock_index

        # Mock the list_indexes method
        mock.list_indexes.return_value = ["test-index"]

        # Set up default return values for query
        mock_index.query.return_value = {
            "matches": [
                {
                    "id": "doc1",
                    "score": 0.8,
                    "metadata": {
                        "text": "Test document 1",
                        "source": "test"
                    }
                },
                {
                    "id": "doc2",
                    "score": 0.7,
                    "metadata": {
                        "text": "Test document 2",
                        "source": "test"
                    }
                }
            ]
        }
        
        yield mock


@pytest.fixture
def mock_env_vars():
    """Set up environment variables for testing."""
    old_api_key = os.environ.get("PINECONE_API_KEY")
    old_env = os.environ.get("PINECONE_ENVIRONMENT")
    
    os.environ["PINECONE_API_KEY"] = "fake-api-key"
    os.environ["PINECONE_ENVIRONMENT"] = "fake-environment"
    
    yield
    
    # Restore environment
    if old_api_key is not None:
        os.environ["PINECONE_API_KEY"] = old_api_key
    else:
        os.environ.pop("PINECONE_API_KEY", None)
        
    if old_env is not None:
        os.environ["PINECONE_ENVIRONMENT"] = old_env
    else:
        os.environ.pop("PINECONE_ENVIRONMENT", None)


@patch("pinecone")
def test_pinecone_init(mock_pinecone, mock_env_vars):
    """Test initializing the Pinecone vector store."""
    mock_pinecone.list_indexes.return_value = ["test-index"]
    
    # Initialize with environment variables
    embeddings = MockEmbeddings()
    vector_store = PineconeVectorStore(
        embeddings=embeddings,
        index_name="test-index"
    )
    
    # Verify pinecone was initialized
    mock_pinecone.init.assert_called_once_with(
        api_key="fake-api-key", 
        environment="fake-environment"
    )
    
    # Verify index was retrieved
    mock_pinecone.Index.assert_called_once_with("test-index")


@patch("pinecone")
def test_pinecone_add_documents(mock_pinecone, mock_env_vars):
    """Test adding documents to Pinecone."""
    mock_pinecone.list_indexes.return_value = ["test-index"]
    mock_index = MagicMock()
    mock_pinecone.Index.return_value = mock_index
    
    embeddings = MockEmbeddings()
    vector_store = PineconeVectorStore(
        embeddings=embeddings,
        index_name="test-index"
    )
    
    # Create test documents
    docs = [
        Document(page_content="Test document 1", metadata={"source": "test"}),
        Document(page_content="Test document 2", metadata={"source": "test"})
    ]
    
    # Add documents with fixed IDs for testing
    fixed_ids = ["id1", "id2"]
    result_ids = vector_store.add_documents(docs, ids=fixed_ids)
    
    # Verify IDs were returned
    assert result_ids == fixed_ids
    
    # Verify upsert was called with correct data
    mock_index.upsert.assert_called_once()
    call_args = mock_index.upsert.call_args[1]
    
    # Check the vectors passed to upsert
    vectors = call_args["vectors"]
    assert len(vectors) == 2
    assert vectors[0]["id"] == "id1"
    assert vectors[0]["values"] == [0.1, 0.2, 0.3]
    assert "text" in vectors[0]["metadata"]
    assert vectors[0]["metadata"]["source"] == "test"


@patch("pinecone")
def test_pinecone_similarity_search(mock_pinecone, mock_env_vars):
    """Test similarity search in Pinecone."""
    mock_pinecone.list_indexes.return_value = ["test-index"]
    mock_index = MagicMock()
    mock_pinecone.Index.return_value = mock_index
    
    # Set up query response
    mock_index.query.return_value = {
        "matches": [
            {
                "id": "doc1",
                "score": 0.8,
                "metadata": {
                    "text": "Test document 1",
                    "source": "test"
                }
            },
            {
                "id": "doc2",
                "score": 0.7,
                "metadata": {
                    "text": "Test document 2",
                    "source": "test"
                }
            }
        ]
    }
    
    embeddings = MockEmbeddings()
    vector_store = PineconeVectorStore(
        embeddings=embeddings,
        index_name="test-index"
    )
    
    # Perform search
    results = vector_store.similarity_search("test query", k=2)
    
    # Verify search was performed
    mock_index.query.assert_called_once()
    assert "vector" in mock_index.query.call_args[1]
    assert mock_index.query.call_args[1]["top_k"] == 2
    
    # Verify results
    assert len(results) == 2
    assert results[0].page_content == "Test document 1"
    assert results[0].metadata["source"] == "test"
    assert results[1].page_content == "Test document 2"
    assert results[1].metadata["source"] == "test"


@patch("pinecone")
def test_pinecone_delete(mock_pinecone, mock_env_vars):
    """Test deleting documents from Pinecone."""
    mock_pinecone.list_indexes.return_value = ["test-index"]
    mock_index = MagicMock()
    mock_pinecone.Index.return_value = mock_index
    
    embeddings = MockEmbeddings()
    vector_store = PineconeVectorStore(
        embeddings=embeddings,
        index_name="test-index"
    )
    
    # Delete documents
    ids_to_delete = ["id1", "id2"]
    vector_store.delete(ids=ids_to_delete)
    
    # Verify delete was called
    mock_index.delete.assert_called_once_with(ids=ids_to_delete, namespace="")