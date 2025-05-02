import asyncio
import functools
import inspect
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    get_type_hints,
)

from hotlm_core.errors import HotLMToolError
from hotlm_core.tools.base import BaseTool
from pydantic import BaseModel, Field, create_model

T = TypeVar("T", bound=Callable)


class ParamStyle(str, Enum):
    """Supported parameter styles for tool functions."""

    POSITIONAL = "positional"  # Pass args as positional
    KEYWORD = "keyword"  # Pass args as keywords
    DICT = "dict"  # Pass args as a single dictionary


def _infer_schema(func: Callable) -> Type[BaseModel]:
    """Infer a Pydantic schema from function signature."""
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)
    fields = {}

    for param in sig.parameters.values():
        # Skip self for methods
        if param.name == "self":
            continue

        # Get annotation with fallback to Any
        ann = type_hints.get(param.name, Any)

        # Get default or make required
        if param.default is not param.empty:
            fields[param.name] = (ann, Field(default=param.default))
        else:
            fields[param.name] = (ann, ...)

    # Create a Pydantic model for the arguments
    return create_model(f"{func.__name__}Schema", **fields, __base__=BaseModel)


def _get_doc_sections(doc_str: Optional[str]) -> Dict[str, str]:
    """Parse docstring to extract sections like Args, Returns, etc."""
    if not doc_str:
        return {}

    sections = {}
    current_section = "description"
    current_content = []

    for line in doc_str.split("\n"):
        line = line.strip()

        # Check for section headers
        if (
            line
            and line.endswith(":")
            and not line.startswith(" ")
            and not line.startswith("\t")
        ):
            # Save previous section
            if current_section:
                sections[current_section] = "\n".join(current_content).strip()

            # Start new section
            current_section = line[:-1].lower()  # Remove the trailing colon
            current_content = []
        else:
            current_content.append(line)

    # Don't forget the last section
    if current_section and current_content:
        sections[current_section] = "\n".join(current_content).strip()

    return sections


def tool(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    args_schema: Optional[Type[BaseModel]] = None,
    return_direct: bool = False,
    param_style: ParamStyle = ParamStyle.KEYWORD,
    handle_tool_error: Optional[Callable] = None,
) -> Union[Callable[[T], BaseTool], BaseTool]:
    """
    Decorator to convert a function into a BaseTool.

    Args:
        func: The function to convert into a tool
        name: Optional name override for the tool (defaults to function name)
        description: Optional description (defaults to function docstring)
        args_schema: Optional Pydantic model for argument validation (inferred if not provided)
        return_direct: Whether to return the tool result directly or embed in a response
        param_style: How to pass parameters to the function (keyword, positional, or as dict)
        handle_tool_error: Optional custom error handler function

    Returns:
        A BaseTool instance or a decorator function that returns a BaseTool
    """

    def decorator(func: Callable) -> BaseTool:
        # Get function metadata
        is_coroutine = asyncio.iscoroutinefunction(func)
        func_name = func.__name__

        # Set name and description
        tool_name = name or func_name

        # Process docstring for description and param help
        if func.__doc__:
            doc_sections = _get_doc_sections(func.__doc__)
            func_description = doc_sections.get("description", func.__doc__.strip())
        else:
            func_description = ""

        tool_description = description or func_description

        # Set up the args schema
        tool_args_schema = args_schema or _infer_schema(func)

        _return_direct = return_direct  # Capture closure variable

        class DynamicFunctionTool(BaseTool):
            name = tool_name
            description = tool_description
            args_schema = tool_args_schema
            return_direct = _return_direct

            def _run(self, args: Dict[str, Any], config=None) -> str:
                """Execute the tool synchronously."""
                try:
                    # Handle different parameter styles
                    if param_style == ParamStyle.DICT:
                        result = func(args)
                    elif param_style == ParamStyle.POSITIONAL:
                        # Convert to positional args in the order of schema fields
                        pos_args = [
                            args.get(field)
                            for field in self.args_schema.schema()["properties"].keys()
                        ]
                        result = func(*pos_args)
                    else:  # Default: KEYWORD
                        result = func(**args)

                    return str(result)
                except Exception as e:
                    if handle_tool_error:
                        return handle_tool_error(e, args)
                    raise HotLMToolError(f"Error in tool {self.name}: {str(e)}") from e

            async def _arun(self, args: Dict[str, Any], config=None) -> str:
                """Execute the tool asynchronously."""
                try:
                    # For coroutine functions, use await
                    if is_coroutine:
                        if param_style == ParamStyle.DICT:
                            result = await func(args)
                        elif param_style == ParamStyle.POSITIONAL:
                            pos_args = [
                                args.get(field)
                                for field in self.args_schema.schema()[
                                    "properties"
                                ].keys()
                            ]
                            result = await func(*pos_args)
                        else:  # Default: KEYWORD
                            result = await func(**args)
                    else:
                        # For regular functions, run in executor to avoid blocking
                        if param_style == ParamStyle.DICT:
                            result = await asyncio.to_thread(func, args)
                        elif param_style == ParamStyle.POSITIONAL:
                            pos_args = [
                                args.get(field)
                                for field in self.args_schema.schema()[
                                    "properties"
                                ].keys()
                            ]
                            result = await asyncio.to_thread(func, *pos_args)
                        else:  # Default: KEYWORD
                            result = await asyncio.to_thread(
                                functools.partial(func, **args)
                            )

                    return str(result)
                except Exception as e:
                    if handle_tool_error:
                        return handle_tool_error(e, args)
                    raise HotLMToolError(f"Error in tool {self.name}: {str(e)}") from e

        return DynamicFunctionTool()

    # Handle the case where the decorator is used with or without arguments
    if func is None:
        return decorator
    return decorator(func)
