import sqlite3
import threading
from typing import Any, Dict, List, Optional

from hotlm_core.memory.base import BaseMemory
from hotlm_core.memory.history import BaseChatMessageHistory
from hotlm_core.schema.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import ConfigDict, Field


class SQLiteChatMessageHistory(BaseChatMessageHistory):
    """SQLite-backed implementation of chat message history."""

    def __init__(self, db_path: str = ":memory:") -> None:
        """Initialize the SQLite history store.

        Args:
            db_path: Path to the SQLite database file (use ":memory:" for in-memory DB).
        """
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database with required tables."""
        # Create the table immediately
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL
                )"""
            )
            conn.commit()

    def get_messages(self) -> List[BaseMessage]:
        """Retrieve all stored messages in chronological order.

        Returns:
            List of BaseMessage objects (HumanMessage, AIMessage, or generic).
        """
        with self._lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT role, content FROM messages ORDER BY id ASC")
            rows = cursor.fetchall()
        messages: List[BaseMessage] = []
        for role, content in rows:
            if role == "human":
                messages.append(HumanMessage(content=content))
            elif role == "ai":
                messages.append(AIMessage(content=content))
            else:
                # fallback generic
                msg = BaseMessage(content=content, role=role)
                messages.append(msg)
        return messages

    async def aget_messages(self) -> List[BaseMessage]:
        """Asynchronously retrieve all stored messages.

        Returns:
            List of BaseMessage objects.
        """
        return self.get_messages()

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the SQLite history.

        Args:
            message: BaseMessage instance to store (HumanMessage or AIMessage).
        """
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO messages (role, content) VALUES (?, ?)",
                (message.role, message.content),
            )
            conn.commit()

    async def aadd_message(self, message: BaseMessage) -> None:
        """Asynchronously add a message to the SQLite history.

        Args:
            message: BaseMessage instance to store.
        """
        self.add_message(message)

    def clear(self) -> None:
        """Clear all messages from the SQLite history."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM messages")
            conn.commit()

    async def aclear(self) -> None:
        """Asynchronously clear all messages from the SQLite history."""
        self.clear()


class SQLiteMemory(BaseMemory):
    """Memory that persists conversational history to SQLite."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    memory_key: str = Field(default="history")
    db_path: str = Field(default=":memory:")
    history: SQLiteChatMessageHistory = None

    def __init__(self, db_path: str = ":memory:", memory_key: str = "history") -> None:
        """Initialize SQLiteMemory with a history backend.

        Args:
            db_path: Path to SQLite DB or ":memory:".
            memory_key: Key under which memory variables are stored.
        """
        super().__init__()
        self.db_path = db_path
        self.memory_key = memory_key
        self.history = SQLiteChatMessageHistory(db_path=db_path)

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load memory variables for the agent run.

        Args:
            inputs: Input variables for the run (unused here).

        Returns:
            A dict mapping memory_key to a list of message dicts.
        """
        messages = self.history.get_messages()
        return {self.memory_key: [self._message_to_dict(msg) for msg in messages]}

    def _message_to_dict(self, message: BaseMessage) -> Dict[str, Any]:
        """Convert a message to a dictionary representation."""
        return {"role": message.role, "content": message.content}

    async def aload_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Asynchronously load memory variables.

        Args:
            inputs: Input variables for the run.

        Returns:
            A dict mapping memory_key to a list of message dicts.
        """
        msgs = await self.history.aget_messages()
        return {self.memory_key: [self._message_to_dict(msg) for msg in msgs]}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save a turn of context to SQLite memory.

        Args:
            inputs: Dict containing input values (key 'input').
            outputs: Dict containing output values.
        """
        # Save last human and AI messages
        human_input = inputs.get("input", str(inputs))
        ai_output = next(iter(outputs.values()), "")

        self.history.add_message(HumanMessage(content=str(human_input)))
        self.history.add_message(AIMessage(content=str(ai_output)))

    async def asave_context(
        self, inputs: Dict[str, Any], outputs: Dict[str, str]
    ) -> None:
        """Asynchronously save a turn of context."""
        self.save_context(inputs, outputs)

    def clear(self) -> None:
        """Clear all stored memory in SQLite."""
        self.history.clear()

    async def aclear(self) -> None:
        """Asynchronously clear all stored memory."""
        await self.history.aclear()
