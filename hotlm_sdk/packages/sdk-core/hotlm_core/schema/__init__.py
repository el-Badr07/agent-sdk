"""Core schemas for HotLM."""

from typing import List

from .documents import Document
from .messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    MessageType,
    SystemMessage,
    ToolMessage,
)
from .tool_defs import ToolCall, ToolObservation


def get_buffer_string(
    messages: List[BaseMessage], human_prefix: str = "Human", ai_prefix: str = "AI"
) -> str:
    """Format a list of BaseMessage into a conversational buffer string.

    Args:
        messages: List of BaseMessage objects to format.
        human_prefix: Prefix to use for human messages.
        ai_prefix: Prefix to use for AI messages.

    Returns:
        A newline-separated string representing the message history."""
    from hotlm_core.schema.messages import AIMessage, HumanMessage

    lines = []
    for message in messages:
        if isinstance(message, HumanMessage):
            prefix = human_prefix
        elif isinstance(message, AIMessage):
            prefix = ai_prefix
        else:
            prefix = getattr(message, "role", "").capitalize() or "Message"
        lines.append(f"{prefix}: {message.content}")
    return "\n".join(lines)


__all__ = [
    "BaseMessage",
    "HumanMessage",
    "AIMessage",
    "SystemMessage",
    "ToolMessage",
    "MessageType",
    "Document",
    "ToolCall",
    "ToolObservation",
    "get_buffer_string",
]
