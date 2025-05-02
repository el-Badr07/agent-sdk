from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


# Base Message Type
class BaseMessage(BaseModel):
    """Base abstract message class."""
    content: Union[str, List[Dict[str, Any]]] # Content can be string or list for multimodal
    role: str
    name: Optional[str] = None
    additional_kwargs: Dict[str, Any] = Field(default_factory=dict)

    # TODO: Add methods for serialization, representation, etc. if needed

# Concrete Message Types
class HumanMessage(BaseMessage):
    """Message from a human user."""
    role: Literal["human"] = "human"

class AIMessage(BaseMessage):
    """Message from an AI model."""
    role: Literal["ai"] = "ai"
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list) # For potential tool usage

class SystemMessage(BaseMessage):
    """System message to set context or instructions."""
    role: Literal["system"] = "system"

class ToolMessage(BaseMessage):
    """Message containing the result of a tool call."""
    role: Literal["tool"] = "tool"
    tool_call_id: str # ID of the tool call this message is a response to

# Generic Message Type for flexibility
MessageType = Union[HumanMessage, AIMessage, SystemMessage, ToolMessage]
