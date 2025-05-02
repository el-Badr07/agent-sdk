from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

from hotlm_core.errors import HotLMToolError, HotLMValidationError
from hotlm_core.runnables import BaseRunnable, RunnableConfig
from pydantic import BaseModel, Field, ValidationError


class BaseTool(BaseRunnable[Dict[str, Any], str], ABC):
    """Interface for tools.

    A tool is a runnable component that takes a dictionary of arguments
    and returns a string output.
    """

    name: str = Field(..., description="The unique name of the tool.")
    description: str = Field(..., description="Description of the tool's purpose, used by LLM.")
    args_schema: Optional[Type[BaseModel]] = Field(None, description="Pydantic model defining the expected arguments.")

    def _validate_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and parse arguments against the args_schema."""
        if self.args_schema is None:
            return args # No schema to validate against
        try:
            validated_args = self.args_schema(**args).model_dump()
            return validated_args
        except ValidationError as e:
            raise HotLMValidationError(f"Invalid arguments for tool '{self.name}': {e}") from e
        except Exception as e:
            # Catch other potential instantiation errors
            raise HotLMToolError(f"Error processing arguments for tool '{self.name}': {e}") from e

    # Override invoke/ainvoke to use _run/_arun with argument validation
    def invoke(self, input: Dict[str, Any], config: Optional[RunnableConfig] = None) -> str:
        validated_args = self._validate_args(input)
        try:
            return self._run(validated_args, config=config)
        except Exception as e:
            # Catch errors during the actual tool run
            if isinstance(e, HotLMError):
                raise # Re-raise framework errors directly
            raise HotLMToolError(f"Error running tool '{self.name}': {e}") from e

    async def ainvoke(self, input: Dict[str, Any], config: Optional[RunnableConfig] = None) -> str:
        validated_args = self._validate_args(input) # Validation is sync for now
        try:
            return await self._arun(validated_args, config=config)
        except Exception as e:
            # Catch errors during the actual tool run
            if isinstance(e, HotLMError):
                raise # Re-raise framework errors directly
            raise HotLMToolError(f"Error running tool '{self.name}' asynchronously: {e}") from e

    @abstractmethod
    def _run(self, **kwargs: Any) -> str:
        """Synchronous execution logic of the tool. Accepts validated args as kwargs."""
        pass

    @abstractmethod
    async def _arun(self, **kwargs: Any) -> str:
        """Asynchronous execution logic of the tool. Accepts validated args as kwargs."""
        pass
