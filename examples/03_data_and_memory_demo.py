#!/usr/bin/env python
"""
Example script demonstrating data loaders, vector stores, and memory components.

This example shows:
1. Loading documents from CSV and JSON
2. Transforming documents with RunnableSequence
3. Using FAISS for vector similarity search
4. Persisting chat history with SQLiteMemory
"""

import csv
import json

# Set up logging
import logging
import os
import tempfile
from typing import List

from hotlm_core.dataloaders.csv_loader import CSVLoader
from hotlm_core.dataloaders.json_loader import JSONLoader
from hotlm_core.embeddings import BaseEmbeddings
from hotlm_core.memory.sqlite_memory import SQLiteChatMessageHistory, SQLiteMemory
from hotlm_core.runnables import BaseRunnable, RunnableLambda, RunnableSequence
from hotlm_core.schema.documents import Document
from hotlm_core.schema.messages import AIMessage, HumanMessage
from hotlm_core.vectorstores.faiss import FaissVectorStore

logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mock embedding model for demo purposes
class MockEmbeddings(BaseEmbeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return simple mock embeddings for documents."""
        # Just create a simple embedding vector for each text
        return [[hash(text) % 1000 / 1000] * 4 for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Return simple mock embeddings for a query."""
        return [hash(text) % 1000 / 1000] * 4

# Create temporary sample data
def create_sample_data():
    """Create temporary CSV and JSON files for the demo."""
    # Create a temp directory
    temp_dir = tempfile.mkdtemp()
    
    # Create sample CSV
    csv_path = os.path.join(temp_dir, "articles.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["content", "source", "date"])
        writer.writeheader()
        writer.writerows([
            {"content": "Machine learning models are transforming businesses.", 
             "source": "Tech News", "date": "2025-03-01"},
            {"content": "New AI agents can plan and execute complex workflows.", 
             "source": "AI Weekly", "date": "2025-04-15"},
            {"content": "Researchers demonstrate advanced reasoning capabilities.", 
             "source": "Science Today", "date": "2025-02-28"}
        ])
    
    # Create JSON directory and files
    json_dir = os.path.join(temp_dir, "json_docs")
    os.makedirs(json_dir, exist_ok=True)
    
    json_files = [
        {"text": "Smart homes are becoming increasingly popular with IoT devices.", 
         "category": "technology", "keywords": ["IoT", "smart homes"]},
        {"text": "Electric vehicles saw record adoption rates this quarter.", 
         "category": "transportation", "keywords": ["EV", "sustainable"]},
        {"text": "New programming languages focus on safety and performance.", 
         "category": "software", "keywords": ["programming", "language design"]}
    ]
    
    for i, data in enumerate(json_files):
        with open(os.path.join(json_dir, f"doc{i}.json"), 'w', encoding='utf-8') as f:
            json.dump(data, f)
    
    return temp_dir, csv_path, json_dir

# Demo components
def demo_data_loaders(csv_path: str, json_dir: str):
    """Demonstrate loading documents from CSV and JSON."""
    print("\n===== DATA LOADERS =====")
    
    # Load from CSV
    print("\n--- CSV Loader ---")
    csv_loader = CSVLoader(csv_path, text_field="content")
    csv_docs = csv_loader.load()
    print(f"Loaded {len(csv_docs)} documents from CSV:")
    for i, doc in enumerate(csv_docs):
        print(f"  Doc {i}: {doc.page_content[:50]}... (metadata: {doc.metadata})")
    
    # Load from JSON
    print("\n--- JSON Loader ---")
    json_loader = JSONLoader(json_dir, text_field="text")
    json_docs = json_loader.load()
    print(f"Loaded {len(json_docs)} documents from JSON:")
    for i, doc in enumerate(json_docs):
        print(f"  Doc {i}: {doc.page_content[:50]}... (metadata: {doc.metadata})")
    
    # Return all loaded documents for next demos
    return csv_docs + json_docs

def demo_runnable_streaming(docs: List[Document]):
    """Demonstrate document transformation with RunnableSequence."""
    print("\n===== RUNNABLE STREAMING =====")
    
    # Define document transformation steps
    class Formatter(BaseRunnable):
        def _invoke(self, doc: Document, **kwargs):
            """Format document for display."""
            formatted = f"TITLE: {doc.metadata.get('category', 'Article')}\n"
            formatted += f"CONTENT: {doc.page_content}\n"
            formatted += f"SOURCE: {doc.metadata.get('source', 'Unknown')}"
            return formatted
        
        async def _ainvoke(self, doc: Document, **kwargs):
            return self._invoke(doc, **kwargs)
    
    # Create runnable sequence
    doc_processor = (
        # First: Add a timestamp to metadata
        RunnableLambda(lambda doc: Document(
            page_content=doc.page_content, 
            metadata={**doc.metadata, "processed_at": "2025-04-22"}
        )) | 
        # Then: Format for display
        Formatter()
    )
    
    # Process documents with streaming
    print("\nProcessing documents one by one:")
    for doc in docs[:2]:  # Just process a couple for the demo
        for chunk in doc_processor.stream(doc):
            print(f"\n{chunk}")
            print("-" * 40)

def demo_vector_store(docs: List[Document]):
    """Demonstrate FAISS vector store for similarity search."""
    print("\n===== FAISS VECTOR STORE =====")
    
    # Create embeddings and vector store
    embeddings = MockEmbeddings()
    vector_store = FaissVectorStore(embeddings=embeddings, dimension=4)
    
    # Add documents to the vector store
    doc_ids = vector_store.add_documents(docs)
    print(f"Added {len(doc_ids)} documents to FAISS index")
    
    # Perform similarity searches
    queries = [
        "machine learning and AI",
        "electric vehicles and sustainability",
        "programming languages"
    ]
    
    for query in queries:
        print(f"\nSearch for: '{query}'")
        results = vector_store.similarity_search(query, k=2)
        print(f"Found {len(results)} relevant documents")
        # In a real application, we'd display the actual documents
        # Here we're just showing the IDs since our mock embeddings are simplistic
        print(f"Result IDs: {[doc.metadata.get('id') for doc in results]}")

def demo_sqlite_memory():
    """Demonstrate SQLite memory for persisting chat history."""
    print("\n===== SQLITE MEMORY =====")
    
    # Create a SQLite memory instance
    memory = SQLiteMemory(db_path=":memory:")  # Use persistent path for real applications
    
    # Simulate a conversation
    print("\nAdding conversation turns to memory...")
    memory.save_context(
        {"input": "What is machine learning?"},
        {"output": "Machine learning is a subset of AI that allows systems to learn from data."}
    )
    memory.save_context(
        {"input": "Give me an example of ML application."},
        {"output": "Image recognition is a common application of machine learning."}
    )
    
    # Load the conversation history
    history = memory.load_memory_variables({})
    print("\nLoaded conversation history:")
    for msg in history["history"]:
        print(f"  - {msg['role'].upper()}: {msg['content']}")
    
    # Demonstrate clear functionality
    memory.clear()
    empty_history = memory.load_memory_variables({})
    print(f"\nAfter clearing memory, history has {len(empty_history['history'])} messages")
    
    # Advanced: Direct access to chat history
    print("\nDemonstrating SQLiteChatMessageHistory directly:")
    chat_history = SQLiteChatMessageHistory(db_path=":memory:")
    chat_history.add_message(HumanMessage(content="Hello, AI assistant!"))
    chat_history.add_message(AIMessage(content="Hello! How can I assist you today?"))
    
    messages = chat_history.get_messages()
    print(f"Retrieved {len(messages)} messages from chat history")
    for msg in messages:
        print(f"  - {msg.role.upper()}: {msg.content}")

def main():
    """Run the complete demo."""
    print("=== HotLM SDK Data and Memory Components Demo ===\n")
    
    print("Creating sample data...")
    temp_dir, csv_path, json_dir = create_sample_data()
    
    try:
        # Run the component demos
        docs = demo_data_loaders(csv_path, json_dir)
        demo_runnable_streaming(docs)
        demo_vector_store(docs)
        demo_sqlite_memory()
        
        print("\nDemo completed successfully!")
    
    finally:
        # Clean up temporary files
        import shutil
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up temporary data in {temp_dir}")

if __name__ == "__main__":
    main()