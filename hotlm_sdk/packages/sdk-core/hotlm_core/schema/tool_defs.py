from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """Represents a request from the AI to call a tool."""

    id: str  # Unique ID for this specific tool call instance
    name: str  # The name of the tool to be called
    args: Dict[str, Any]  # Arguments for the tool, structured as a dictionary


class ToolObservation(BaseModel):
    """Represents the result/output of a tool execution."""

    content: str  # The string content of the tool's output
    tool_call_id: str  # The ID of the ToolCall this observation corresponds to
    name: Optional[str] = (
        None  # Optional: Name of the tool that produced this observation
    )
