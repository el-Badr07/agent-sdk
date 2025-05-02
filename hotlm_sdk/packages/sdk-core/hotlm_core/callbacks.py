import asyncio
import uuid
from abc import ABC, abstractmethod

# Forward declaration for type hinting
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Union,
)

from hotlm_core.schema.documents import Document
from hotlm_core.schema.messages import BaseMessage

if TYPE_CHECKING:
    from hotlm_core.config import RunnableConfig

class BaseCallbackHandler(ABC):
    """Base interface for callback handlers.

    Implement methods to react to different events during a Runnable's lifecycle.
    """

    # --- Run Lifecycle Events ---
    def on_run_start(
        self, run_id: uuid.UUID, name: str, inputs: Any, config: "RunnableConfig"
    ) -> None:
        """Called when a Runnable starts."""
        pass

    async def aon_run_start(
        self, run_id: uuid.UUID, name: str, inputs: Any, config: "RunnableConfig"
    ) -> None:
        """Asynchronously called when a Runnable starts."""
        pass

    def on_run_end(
        self, run_id: uuid.UUID, outputs: Any, config: "RunnableConfig"
    ) -> None:
        """Called when a Runnable ends successfully."""
        pass

    async def aon_run_end(
        self, run_id: uuid.UUID, outputs: Any, config: "RunnableConfig"
    ) -> None:
        """Asynchronously called when a Runnable ends successfully."""
        pass

    def on_run_error(
        self, run_id: uuid.UUID, error: BaseException, config: "RunnableConfig"
    ) -> None:
        """Called when a Runnable encounters an error."""
        pass

    async def aon_run_error(
        self, run_id: uuid.UUID, error: BaseException, config: "RunnableConfig"
    ) -> None:
        """Asynchronously called when a Runnable encounters an error."""
        pass 

    # --- Streaming Events ---
    def on_stream_chunk(
        self, run_id: uuid.UUID, chunk: Any, config: "RunnableConfig"
    ) -> None:
        """Called when a streaming Runnable yields a chunk."""
        pass

    async def aon_stream_chunk(
        self, run_id: uuid.UUID, chunk: Any, config: "RunnableConfig"
    ) -> None:
        """Asynchronously called when a streaming Runnable yields a chunk."""
        pass

    # --- LLM Events ---
    def on_llm_start(
        self, run_id: uuid.UUID, prompts: List[str], config: "RunnableConfig"
    ) -> None:
        """Called when an LLM call starts (with raw prompts)."""
        pass

    async def aon_llm_start(
        self, run_id: uuid.UUID, prompts: List[str], config: "RunnableConfig"
    ) -> None:
        """Asynchronously called when an LLM call starts."""
        pass

    def on_chat_model_start(
        self, run_id: uuid.UUID, messages: List[List[BaseMessage]], config: "RunnableConfig"
    ) -> None:
        """Called when a Chat Model call starts (with message lists)."""
        pass

    async def aon_chat_model_start(
        self, run_id: uuid.UUID, messages: List[List[BaseMessage]], config: "RunnableConfig"
    ) -> None:
        """Asynchronously called when a Chat Model call starts."""
        pass

    def on_llm_end(
        self, run_id: uuid.UUID, response: Any, config: "RunnableConfig"
    ) -> None:
        """Called when an LLM call ends successfully."""
        pass

    async def aon_llm_end(
        self, run_id: uuid.UUID, response: Any, config: "RunnableConfig"
    ) -> None:
        """Asynchronously called when an LLM call ends successfully."""
        pass

    def on_llm_error(
        self, run_id: uuid.UUID, error: BaseException, config: "RunnableConfig"
    ) -> None:
        """Called when an LLM call encounters an error."""
        pass

    async def aon_llm_error(
        self, run_id: uuid.UUID, error: BaseException, config: "RunnableConfig"
    ) -> None:
        """Asynchronously called when an LLM call encounters an error."""
        pass

    # --- Tool Events ---
    def on_tool_start(
        self, run_id: uuid.UUID, tool_input: Dict[str, Any], config: "RunnableConfig"
    ) -> None:
        """Called when a Tool execution starts."""
        pass

    async def aon_tool_start(
        self, run_id: uuid.UUID, tool_input: Dict[str, Any], config: "RunnableConfig"
    ) -> None:
        """Asynchronously called when a Tool execution starts."""
        pass

    def on_tool_end(
        self, run_id: uuid.UUID, tool_output: Any, config: "RunnableConfig"
    ) -> None:
        """Called when a Tool execution ends successfully."""
        pass

    async def aon_tool_end(
        self, run_id: uuid.UUID, tool_output: Any, config: "RunnableConfig"
    ) -> None:
        """Asynchronously called when a Tool execution ends successfully."""
        pass

    def on_tool_error(
        self, run_id: uuid.UUID, error: BaseException, config: "RunnableConfig"
    ) -> None:
        """Called when a Tool execution encounters an error."""
        pass

    async def aon_tool_error(
        self, run_id: uuid.UUID, error: BaseException, config: "RunnableConfig"
    ) -> None:
        """Asynchronously called when a Tool execution encounters an error."""
        pass

    # --- Retriever Events ---
    def on_retriever_start(
        self, run_id: uuid.UUID, query: str, config: "RunnableConfig"
    ) -> None:
        """Called when a Retriever starts."""
        pass

    async def aon_retriever_start(
        self, run_id: uuid.UUID, query: str, config: "RunnableConfig"
    ) -> None:
        """Asynchronously called when a Retriever starts."""
        pass

    def on_retriever_end(
        self, run_id: uuid.UUID, documents: List[Document], config: "RunnableConfig"
    ) -> None:
        """Called when a Retriever ends successfully."""
        pass

    async def aon_retriever_end(
        self, run_id: uuid.UUID, documents: List[Document], config: "RunnableConfig"
    ) -> None:
        """Asynchronously called when a Retriever ends successfully."""
        pass

    def on_retriever_error(
        self, run_id: uuid.UUID, error: BaseException, config: "RunnableConfig"
    ) -> None:
        """Called when a Retriever encounters an error."""
        pass

    async def aon_retriever_error(
        self, run_id: uuid.UUID, error: BaseException, config: "RunnableConfig"
    ) -> None:
        """Asynchronously called when a Retriever encounters an error."""
        pass


class CallbackManager:
    """Manages multiple callback handlers for a Runnable execution."""
    handlers: List[BaseCallbackHandler]

    def __init__(self, handlers: Sequence[BaseCallbackHandler]):
        self.handlers = list(handlers)

    def add_handler(self, handler: BaseCallbackHandler) -> None:
        self.handlers.append(handler)

    def remove_handler(self, handler: BaseCallbackHandler) -> None:
        self.handlers.remove(handler)

    # --- Run Lifecycle Events ---
    def on_run_start(
        self, run_id: uuid.UUID, name: str, inputs: Any, config: "RunnableConfig"
    ) -> None:
        for handler in self.handlers:
            try:
                handler.on_run_start(run_id, name, inputs, config)
            except Exception as e:
                print(f"Error in handler {handler.__class__.__name__}.on_run_start: {e}")

    async def aon_run_start(
        self, run_id: uuid.UUID, name: str, inputs: Any, config: "RunnableConfig"
    ) -> None:
        await asyncio.gather(
            *[
                self._safe_acall(handler.aon_run_start, run_id, name, inputs, config)
                for handler in self.handlers
            ]
        )

    def on_run_end(
        self, run_id: uuid.UUID, outputs: Any, config: "RunnableConfig"
    ) -> None:
        for handler in self.handlers:
            try:
                handler.on_run_end(run_id, outputs, config)
            except Exception as e:
                print(f"Error in handler {handler.__class__.__name__}.on_run_end: {e}")

    async def aon_run_end(
        self, run_id: uuid.UUID, outputs: Any, config: "RunnableConfig"
    ) -> None:
        await asyncio.gather(
            *[
                self._safe_acall(handler.aon_run_end, run_id, outputs, config)
                for handler in self.handlers
            ]
        )

    def on_run_error(
        self, run_id: uuid.UUID, error: BaseException, config: "RunnableConfig"
    ) -> None:
        for handler in self.handlers:
            try:
                handler.on_run_error(run_id, error, config)
            except Exception as e:
                print(f"Error in handler {handler.__class__.__name__}.on_run_error: {e}")

    async def aon_run_error(
        self, run_id: uuid.UUID, error: BaseException, config: "RunnableConfig"
    ) -> None:
        await asyncio.gather(
            *[
                self._safe_acall(handler.aon_run_error, run_id, error, config)
                for handler in self.handlers
            ]
        )

    # --- Streaming Events ---
    def on_stream_chunk(
        self, run_id: uuid.UUID, chunk: Any, config: "RunnableConfig"
    ) -> None:
        for handler in self.handlers:
            try:
                handler.on_stream_chunk(run_id, chunk, config)
            except Exception as e:
                print(f"Error in handler {handler.__class__.__name__}.on_stream_chunk: {e}")

    async def aon_stream_chunk(
        self, run_id: uuid.UUID, chunk: Any, config: "RunnableConfig"
    ) -> None:
        await asyncio.gather(
            *[
                self._safe_acall(handler.aon_stream_chunk, run_id, chunk, config)
                for handler in self.handlers
            ]
        )

    # --- LLM Events ---
    def on_llm_start(
        self, run_id: uuid.UUID, prompts: List[str], config: "RunnableConfig"
    ) -> None:
        for handler in self.handlers:
            try:
                handler.on_llm_start(run_id, prompts, config)
            except Exception as e:
                print(f"Error in handler {handler.__class__.__name__}.on_llm_start: {e}")

    async def aon_llm_start(
        self, run_id: uuid.UUID, prompts: List[str], config: "RunnableConfig"
    ) -> None:
        await asyncio.gather(
            *[
                self._safe_acall(handler.aon_llm_start, run_id, prompts, config)
                for handler in self.handlers
            ]
        )

    def on_chat_model_start(
        self, run_id: uuid.UUID, messages: List[List[BaseMessage]], config: "RunnableConfig"
    ) -> None:
        for handler in self.handlers:
            try:
                handler.on_chat_model_start(run_id, messages, config)
            except Exception as e:
                print(f"Error in handler {handler.__class__.__name__}.on_chat_model_start: {e}")

    async def aon_chat_model_start(
        self, run_id: uuid.UUID, messages: List[List[BaseMessage]], config: "RunnableConfig"
    ) -> None:
        await asyncio.gather(
            *[
                self._safe_acall(handler.aon_chat_model_start, run_id, messages, config)
                for handler in self.handlers
            ]
        )

    def on_llm_end(
        self, run_id: uuid.UUID, response: Any, config: "RunnableConfig"
    ) -> None:
        for handler in self.handlers:
            try:
                handler.on_llm_end(run_id, response, config)
            except Exception as e:
                print(f"Error in handler {handler.__class__.__name__}.on_llm_end: {e}")

    async def aon_llm_end(
        self, run_id: uuid.UUID, response: Any, config: "RunnableConfig"
    ) -> None:
        await asyncio.gather(
            *[
                self._safe_acall(handler.aon_llm_end, run_id, response, config)
                for handler in self.handlers
            ]
        )

    def on_llm_error(
        self, run_id: uuid.UUID, error: BaseException, config: "RunnableConfig"
    ) -> None:
        for handler in self.handlers:
            try:
                handler.on_llm_error(run_id, error, config)
            except Exception as e:
                print(f"Error in handler {handler.__class__.__name__}.on_llm_error: {e}")

    async def aon_llm_error(
        self, run_id: uuid.UUID, error: BaseException, config: "RunnableConfig"
    ) -> None:
        await asyncio.gather(
            *[
                self._safe_acall(handler.aon_llm_error, run_id, error, config)
                for handler in self.handlers
            ]
        )

    # --- Tool Events ---
    def on_tool_start(
        self, run_id: uuid.UUID, tool_input: Dict[str, Any], config: "RunnableConfig"
    ) -> None:
        for handler in self.handlers:
            try:
                handler.on_tool_start(run_id, tool_input, config)
            except Exception as e:
                print(f"Error in handler {handler.__class__.__name__}.on_tool_start: {e}")

    async def aon_tool_start(
        self, run_id: uuid.UUID, tool_input: Dict[str, Any], config: "RunnableConfig"
    ) -> None:
        await asyncio.gather(
            *[
                self._safe_acall(handler.aon_tool_start, run_id, tool_input, config)
                for handler in self.handlers
            ]
        )

    def on_tool_end(
        self, run_id: uuid.UUID, tool_output: Any, config: "RunnableConfig"
    ) -> None:
        for handler in self.handlers:
            try:
                handler.on_tool_end(run_id, tool_output, config)
            except Exception as e:
                print(f"Error in handler {handler.__class__.__name__}.on_tool_end: {e}")

    async def aon_tool_end(
        self, run_id: uuid.UUID, tool_output: Any, config: "RunnableConfig"
    ) -> None:
        await asyncio.gather(
            *[
                self._safe_acall(handler.aon_tool_end, run_id, tool_output, config)
                for handler in self.handlers
            ]
        )

    def on_tool_error(
        self, run_id: uuid.UUID, error: BaseException, config: "RunnableConfig"
    ) -> None:
        for handler in self.handlers:
            try:
                handler.on_tool_error(run_id, error, config)
            except Exception as e:
                print(f"Error in handler {handler.__class__.__name__}.on_tool_error: {e}")

    async def aon_tool_error(
        self, run_id: uuid.UUID, error: BaseException, config: "RunnableConfig"
    ) -> None:
        await asyncio.gather(
            *[
                self._safe_acall(handler.aon_tool_error, run_id, error, config)
                for handler in self.handlers
            ]
        )

    # --- Retriever Events ---
    def on_retriever_start(
        self, run_id: uuid.UUID, query: str, config: "RunnableConfig"
    ) -> None:
        for handler in self.handlers:
            try:
                handler.on_retriever_start(run_id, query, config)
            except Exception as e:
                print(f"Error in handler {handler.__class__.__name__}.on_retriever_start: {e}")

    async def aon_retriever_start(
        self, run_id: uuid.UUID, query: str, config: "RunnableConfig"
    ) -> None:
        await asyncio.gather(
            *[
                self._safe_acall(handler.aon_retriever_start, run_id, query, config)
                for handler in self.handlers
            ]
        )

    def on_retriever_end(
        self, run_id: uuid.UUID, documents: List[Document], config: "RunnableConfig"
    ) -> None:
        for handler in self.handlers:
            try:
                handler.on_retriever_end(run_id, documents, config)
            except Exception as e:
                print(f"Error in handler {handler.__class__.__name__}.on_retriever_end: {e}")

    async def aon_retriever_end(
        self, run_id: uuid.UUID, documents: List[Document], config: "RunnableConfig"
    ) -> None:
        await asyncio.gather(
            *[
                self._safe_acall(handler.aon_retriever_end, run_id, documents, config)
                for handler in self.handlers
            ]
        )

    def on_retriever_error(
        self, run_id: uuid.UUID, error: BaseException, config: "RunnableConfig"
    ) -> None:
        for handler in self.handlers:
            try:
                handler.on_retriever_error(run_id, error, config)
            except Exception as e:
                print(f"Error in handler {handler.__class__.__name__}.on_retriever_error: {e}")

    async def aon_retriever_error(
        self, run_id: uuid.UUID, error: BaseException, config: "RunnableConfig"
    ) -> None:
        await asyncio.gather(
            *[
                self._safe_acall(handler.aon_retriever_error, run_id, error, config)
                for handler in self.handlers
            ]
        )

    # --- Helper for safe async calls ---
    async def _safe_acall(self, func, *args, **kwargs):
        try:
            await func(*args, **kwargs)
        except Exception as e:
            print(f"Error in async handler {func.__name__}: {e}")


# --- Concrete Handler Example ---

class ConsoleCallbackHandler(BaseCallbackHandler):
    """A simple callback handler that prints events to the console."""

    def on_run_start(
        self, run_id: uuid.UUID, name: str, inputs: Any, config: "RunnableConfig"
    ) -> None:
        print(f"[RUN START] ID: {run_id} Name: {name} Inputs: {inputs} Config: {config.dict(exclude={'callback_manager'})}")

    def on_run_end(
        self, run_id: uuid.UUID, outputs: Any, config: "RunnableConfig"
    ) -> None:
        print(f"[RUN END] ID: {run_id} Outputs: {outputs}")

    def on_run_error(
        self, run_id: uuid.UUID, error: BaseException, config: "RunnableConfig"
    ) -> None:
        print(f"[RUN ERROR] ID: {run_id} Error: {error}")

    def on_stream_chunk(
        self, run_id: uuid.UUID, chunk: Any, config: "RunnableConfig"
    ) -> None:
        print(f"[STREAM CHUNK] ID: {run_id} Chunk: {chunk}")

    def on_llm_start(
        self, run_id: uuid.UUID, prompts: List[str], config: "RunnableConfig"
    ) -> None:
        print(f"[LLM START] ID: {run_id} Prompts: {prompts}")

    def on_chat_model_start(
        self, run_id: uuid.UUID, messages: List[List[BaseMessage]], config: "RunnableConfig"
    ) -> None:
        print(f"[CHAT START] ID: {run_id} Messages: {[[m.dict() for m in msg_list] for msg_list in messages]}")

    def on_llm_end(
        self, run_id: uuid.UUID, response: Any, config: "RunnableConfig"
    ) -> None:
        print(f"[LLM END] ID: {run_id} Response: {response}") # Might need response.dict()

    def on_llm_error(
        self, run_id: uuid.UUID, error: BaseException, config: "RunnableConfig"
    ) -> None:
        print(f"[LLM ERROR] ID: {run_id} Error: {error}")

    def on_tool_start(
        self, run_id: uuid.UUID, tool_input: Dict[str, Any], config: "RunnableConfig"
    ) -> None:
        print(f"[TOOL START] ID: {run_id} Input: {tool_input}")

    def on_tool_end(
        self, run_id: uuid.UUID, tool_output: Any, config: "RunnableConfig"
    ) -> None:
        print(f"[TOOL END] ID: {run_id} Output: {tool_output}")

    def on_tool_error(
        self, run_id: uuid.UUID, error: BaseException, config: "RunnableConfig"
    ) -> None:
        print(f"[TOOL ERROR] ID: {run_id} Error: {error}")

    def on_retriever_start(
        self, run_id: uuid.UUID, query: str, config: "RunnableConfig"
    ) -> None:
        print(f"[RETRIEVER START] ID: {run_id} Query: {query}")

    def on_retriever_end(
        self, run_id: uuid.UUID, documents: List[Document], config: "RunnableConfig"
    ) -> None:
        print(f"[RETRIEVER END] ID: {run_id} Documents: {[doc.dict() for doc in documents]}")

    def on_retriever_error(
        self, run_id: uuid.UUID, error: BaseException, config: "RunnableConfig"
    ) -> None:
        print(f"[RETRIEVER ERROR] ID: {run_id} Error: {error}")

    # Async versions simply call the sync versions for this basic handler
    async def aon_run_start(self, *args, **kwargs): self.on_run_start(*args, **kwargs)
    async def aon_run_end(self, *args, **kwargs): self.on_run_end(*args, **kwargs)
    async def aon_run_error(self, *args, **kwargs): self.on_run_error(*args, **kwargs)
    async def aon_stream_chunk(self, *args, **kwargs): self.on_stream_chunk(*args, **kwargs)
    async def aon_llm_start(self, *args, **kwargs): self.on_llm_start(*args, **kwargs)
    async def aon_chat_model_start(self, *args, **kwargs): self.on_chat_model_start(*args, **kwargs)
    async def aon_llm_end(self, *args, **kwargs): self.on_llm_end(*args, **kwargs)
    async def aon_llm_error(self, *args, **kwargs): self.on_llm_error(*args, **kwargs)
    async def aon_tool_start(self, *args, **kwargs): self.on_tool_start(*args, **kwargs)
    async def aon_tool_end(self, *args, **kwargs): self.on_tool_end(*args, **kwargs)
    async def aon_tool_error(self, *args, **kwargs): self.on_tool_error(*args, **kwargs)
    async def aon_retriever_start(self, *args, **kwargs): self.on_retriever_start(*args, **kwargs)
    async def aon_retriever_end(self, *args, **kwargs): self.on_retriever_end(*args, **kwargs)
    async def aon_retriever_error(self, *args, **kwargs): self.on_retriever_error(*args, **kwargs)

