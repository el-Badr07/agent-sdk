import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

from hotlm_core.callbacks import (
    BaseCallbackHandler,  # Needed for CallbackConfig.handlers
)
from pydantic import BaseModel, Field, field_validator

# Avoid circular import
if TYPE_CHECKING:
    from hotlm_core.callbacks import BaseCallbackHandler, CallbackManager

class HotLMConfig(BaseModel):
    """Base class for configuration settings within HotLM.

    Allows for extra fields and provides basic configuration management.
    """
    class Config:
        extra = "allow" # Allow arbitrary fields
        arbitrary_types_allowed = True

# Example: Configuration for retry mechanisms (could be used by Runnables)
class RetryConfig(HotLMConfig):
    max_retries: int = Field(default=3, description="Maximum number of retries.")
    delay_seconds: float = Field(default=1.0, description="Initial delay between retries in seconds.")
    backoff_factor: float = Field(default=2.0, description="Multiplier for increasing delay between retries.")
    retryable_exceptions: List[type[Exception]] = Field(default_factory=list, description="List of exception types to retry on.")

# Example: Configuration for callbacks (could be part of RunnableConfig)
class CallbackConfig(BaseModel):
    """Configuration for callbacks (deprecated in favor of direct manager)."""
    # This might be kept for simpler cases or backward compatibility
    # but RunnableConfig will primarily use CallbackManager directly.
    handlers: List[BaseCallbackHandler] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

class RunnableConfig(BaseModel):
    """Configuration for a Runnable invocation.

    Allows specifying tags, metadata, callbacks, and other execution parameters.
    """
    tags: List[str] = Field(default_factory=list, description="Tags for this run.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata for this run.")
    callback_manager: Optional["CallbackManager"] = Field(default=None, description="Callback manager for this run.")
    run_id: Optional[uuid.UUID] = Field(default=None, description="Unique ID for this run.")
    # Add other potential config fields like max_concurrency, timeout, etc.

    class Config:
        arbitrary_types_allowed = True # Allow CallbackManager

    @field_validator('run_id', mode='before')
    def ensure_run_id(cls, v):
        return v or uuid.uuid4()
