"""
Tool prompt generator.

This module provides functionality for generating formatted prompt snippets
from tool metadata and signatures.
"""

import json
import textwrap
from typing import Any, Dict, List, Optional

from hotlm_agents.tooling.registry import ToolRegistry
from hotlm_core.tools import BaseTool


class ToolPromptGenerator:
    """
    Generator for tool-related prompt snippets.

    This class provides methods to generate formatted descriptions of tools
    for inclusion in prompts to language models.
    """

    @staticmethod
    def generate_tool_description(tool: BaseTool) -> str:
        """
        Generate a formatted description of a tool.

        Args:
            tool: The tool to describe

        Returns:
            A formatted description string
        """
        return f"Tool: {tool.name}\nDescription: {tool.description}"

    @staticmethod
    def generate_schema_snippet(tool: BaseTool) -> str:
        """
        Generate a formatted snippet of the tool's schema.

        Args:
            tool: The tool to describe

        Returns:
            A formatted schema string
        """
        schema = tool.get_args_schema()
        schema_lines = []

        for arg_name, arg_spec in schema.items():
            arg_type = arg_spec.get("type", "any")
            arg_desc = arg_spec.get("description", "No description")
            required = (
                not arg_spec.get("required") is False
            )  # Default to True if not specified

            required_str = "" if required else " (optional)"
            schema_lines.append(
                f"- {arg_name}: {arg_desc} (Type: {arg_type}){required_str}"
            )

        return "\n".join(schema_lines)

    @staticmethod
    def generate_example_usage(tool: BaseTool) -> str:
        """
        Generate example usage of a tool.

        Args:
            tool: The tool to generate an example for

        Returns:
            A formatted example usage string
        """
        schema = tool.get_args_schema()
        example_args = {}

        for arg_name, arg_spec in schema.items():
            arg_type = arg_spec.get("type", "any")
            required = not arg_spec.get("required") is False

            # Skip optional args in simple examples
            if not required:
                continue

            # Generate example values based on the type
            if arg_type == "string":
                example_args[arg_name] = f"example_{arg_name}"
            elif arg_type == "number":
                example_args[arg_name] = 42
            elif arg_type == "integer":
                example_args[arg_name] = 42
            elif arg_type == "boolean":
                example_args[arg_name] = True
            elif arg_type == "array":
                example_args[arg_name] = []
            elif arg_type == "object":
                example_args[arg_name] = {}
            else:
                example_args[arg_name] = f"<{arg_type}_value>"

        example_json = json.dumps(example_args, indent=2)
        return (
            f'```json\n{{\n  "tool": "{tool.name}",\n  "args": {example_json}\n}}\n```'
        )

    @staticmethod
    def generate_full_tool_snippet(tool: BaseTool) -> str:
        """
        Generate a complete formatted description of a tool.

        Args:
            tool: The tool to describe

        Returns:
            A complete tool description
        """
        description = ToolPromptGenerator.generate_tool_description(tool)
        schema = ToolPromptGenerator.generate_schema_snippet(tool)
        example = ToolPromptGenerator.generate_example_usage(tool)

        return f"""
{description}

Arguments:
{schema}

Example:
{example}
"""

    @staticmethod
    def generate_tools_list(
        tools: List[BaseTool], format_type: str = "markdown"
    ) -> str:
        """
        Generate a formatted list of tools.

        Args:
            tools: List of tools to include
            format_type: Format type ("markdown", "json", or "text")

        Returns:
            A formatted list of tools
        """
        if format_type == "markdown":
            tool_entries = []
            for tool in tools:
                tool_entries.append(
                    f"### {tool.name}\n\n{tool.description}\n\n**Arguments:**\n\n{ToolPromptGenerator.generate_schema_snippet(tool)}"
                )
            return "\n\n".join(tool_entries)

        elif format_type == "json":
            tool_list = []
            for tool in tools:
                tool_obj = {
                    "name": tool.name,
                    "description": tool.description,
                    "args": {},
                }

                for arg_name, arg_spec in tool.get_args_schema().items():
                    tool_obj["args"][arg_name] = {
                        "type": arg_spec.get("type", "any"),
                        "description": arg_spec.get("description", ""),
                        "required": not arg_spec.get("required") is False,
                    }

                tool_list.append(tool_obj)

            return json.dumps(tool_list, indent=2)

        else:  # text format
            tool_entries = []
            for tool in tools:
                schema = ToolPromptGenerator.generate_schema_snippet(tool)
                tool_entries.append(
                    f"Tool: {tool.name}\nDescription: {tool.description}\nArguments:\n{schema}"
                )
            return "\n\n".join(tool_entries)

    @staticmethod
    def generate_system_prompt_with_tools(
        tools: List[BaseTool], system_prompt_template: str, format_type: str = "text"
    ) -> str:
        """
        Generate a system prompt that includes tool descriptions.

        Args:
            tools: List of tools to include
            system_prompt_template: Template string with {tools} placeholder
            format_type: Format for the tools list

        Returns:
            A system prompt with tool descriptions
        """
        tools_list = ToolPromptGenerator.generate_tools_list(tools, format_type)
        return system_prompt_template.format(tools=tools_list)
