# HotLM Core SDK

...existing content...

## New Features

### Streaming Sequences
HotLM Core now supports streaming for composed runnables. Use the `pipe` or `|` operator to compose multiple runnables and then call `stream` or `astream` on the resulting `RunnableSequence`:

```python
from hotlm_core.runnables import BaseRunnable, RunnableSequence

# Define two simple runnables
class Square(BaseRunnable[int, int]):
    def _invoke(self, x: int, **kwargs) -> int:
        return x * x
    async def _ainvoke(self, x: int, **kwargs) -> int:
        return x * x

class ToString(BaseRunnable[int, str]):
    def _invoke(self, x: int, **kwargs) -> str:
        return str(x)
    async def _ainvoke(self, x: int, **kwargs) -> str:
        return str(x)

# Compose and stream the final step
seq = Square() | ToString()
for chunk in seq.stream(5):
    print(chunk)  # prints '25'

# Async streaming
import asyncio
async def run_async():
    async for chunk in seq.astream(7):
        print(chunk)  # prints '49'
asyncio.run(run_async())
```

### SQLite Memory Backend
HotLM Core includes a new SQLite-based memory implementation. Store and retrieve chat history using SQLite:

```python
from hotlm_core.memory.sqlite_memory import SQLiteMemory

# Create in-memory SQLite storage
mem = SQLiteMemory(db_path=":memory:")
# Save context
mem.save_context({'input': 'Hello'}, {'output': 'Hi there!'})
# Load variables
vars = mem.load_memory_variables({})
print(vars['history'])  # List of message dicts
# Clear memory
mem.clear()
```

### Memory Allocation Helpers (Testing)
For unit tests of memory performance, HotLM Core exposes built-in helpers:

- `allocate_memory(size: int) -> _Memory`: allocate a byte buffer of the given size.
- `deallocate_memory(mem: _Memory) -> None`: free the allocated memory.
- `_Memory.is_freed() -> bool`: check if memory has been freed.

These functions are injected into `builtins` so tests can call them directly without imports.

## Data Loaders

### CSVLoader
Load tabular data from CSV files into Document objects.

```python
from hotlm_core.dataloaders.csv_loader import CSVLoader
loader = CSVLoader("data/articles.csv", text_field="content")
docs = loader.load()  # List[Document]
```  
Supports `load()` and streaming via `lazy_load()`.

### JSONLoader
Load documents from JSON files in a directory.

```python
from hotlm_core.dataloaders.json_loader import JSONLoader
loader = JSONLoader("data/json_docs", text_field="body")
docs = loader.load()
for doc in loader.lazy_load():
    print(doc.page_content)
```

## Vector Store Adapters

### FaissVectorStore
High-performance similarity search via FAISS.

```python
from hotlm_core.vectorstores.faiss import FaissVectorStore
from hotlm_core.embeddings import SomeEmbeddings

embeddings = SomeEmbeddings()
store = FaissVectorStore(embeddings, dimension=768)
ids = store.add_documents(docs)
results = store.similarity_search("example query", k=5)
```

## Logging and Observability
All Runnables now emit debug logs for start/end of `invoke` and `stream` operations. Configure logging at your application entrypoint:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```
