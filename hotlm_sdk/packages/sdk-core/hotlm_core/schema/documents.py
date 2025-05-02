from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Class for storing a piece of text and associated metadata."""

    page_content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    id: Optional[str] = None # Optional unique identifier

    # TODO: Add methods for formatting, splitting, etc. if needed
