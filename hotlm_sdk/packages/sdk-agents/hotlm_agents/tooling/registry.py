"""Tool registry for managing available tools.

This module provides a registry for managing the available tools,
their metadata, and generating prompt snippets.
"""

import inspect
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Set, Type, Union

from hotlm_core.schema.tool_defs import ToolCall
from hotlm_core.tools.base import BaseTool
from pydantic import BaseModel

from .prompt_generator import ToolPromptGenerator


class ToolRegistry:
    """
    Registry for available tools.
    
    This registry keeps track of all available tools and provides
    methods for retrieving them and generating prompt snippets.
    """
    
    def __init__(self, name: str = "tool_registry"):
        """Initialize a new tool registry.
        
        Args:
            name (str): Name of the registry for identification purposes
        """
        self.name = name
        self._tools: Dict[str, BaseTool] = OrderedDict()
    
    def register(self, tool: Union[BaseTool, Type[BaseTool]]) -> None:
        """Register a tool with the registry.
        
        Args:
            tool: Either a BaseTool instance or a BaseTool class that will be instantiated
        
        Raises:
            ValueError: If a tool with the same name is already registered
        """
        # Handle both instance and class
        if inspect.isclass(tool) and issubclass(tool, BaseTool):
            tool_instance = tool()
        elif isinstance(tool, BaseTool):
            tool_instance = tool
        else:
            raise ValueError(f"Tool must be a BaseTool instance or class, got {type(tool)}")
        
        if tool_instance.name in self._tools:
            raise ValueError(f"A tool with name '{tool_instance.name}' is already registered")
        
        self._tools[tool_instance.name] = tool_instance
    
    def unregister(self, tool_name: str) -> None:
        """Unregister a tool from the registry.
        
        Args:
            tool_name: Name of the tool to unregister
            
        Raises:
            KeyError: If no tool with the given name is registered
        """
        if tool_name not in self._tools:
            raise KeyError(f"No tool named '{tool_name}' is registered")
        
        del self._tools[tool_name]
    
    def get_tool(self, tool_name: str) -> BaseTool:
        """Get a tool by name.
        
        Args:
            tool_name: Name of the tool to retrieve
            
        Returns:
            The requested tool
            
        Raises:
            KeyError: If no tool with the given name is registered
        """
        if tool_name not in self._tools:
            raise KeyError(f"No tool named '{tool_name}' is registered")
        
        return self._tools[tool_name]
    
    def get_all_tools(self) -> List[BaseTool]:
        """Get all registered tools.
        
        Returns:
            List of all registered tools
        """
        return list(self._tools.values())
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get schema definitions for all tools in a format suitable for LLM function calling.
        
        Returns:
            List of tool schema definitions
        """
        schemas = []
        for tool in self._tools.values():
            schema = {
                "name": tool.name,
                "description": tool.description,
            }
            
            # Add parameters schema if available
            if tool.args_schema:
                schema["parameters"] = tool.args_schema.model_json_schema()
            
            schemas.append(schema)
        
        return schemas
    
    def execute_tool(self, tool_call: ToolCall) -> str:
        """Execute a tool based on a tool call.
        
        Args:
            tool_call: The tool call to execute
            
        Returns:
            The result of the tool execution
            
        Raises:
            KeyError: If the requested tool is not registered
        """
        tool = self.get_tool(tool_call.name)
        return tool.invoke(tool_call.args)
    
    async def execute_tool_async(self, tool_call: ToolCall) -> str:
        """Execute a tool asynchronously based on a tool call.
        
        Args:
            tool_call: The tool call to execute
            
        Returns:
            The result of the tool execution
            
        Raises:
            KeyError: If the requested tool is not registered
        """
        tool = self.get_tool(tool_call.name)
        return await tool.ainvoke(tool_call.args)
    
    def __len__(self) -> int:
        """Get the number of registered tools."""
        return len(self._tools)
    
    def __contains__(self, tool_name: str) -> bool:
        """Check if a tool is registered."""
        return tool_name in self._tools
    
    def __repr__(self) -> str:
        """String representation of the registry."""
        return f"ToolRegistry(name='{self.name}', tools={list(self._tools.keys())})"
    
    # New methods for prompt generation
    
    def generate_tools_prompt(self, format_type: str = "text") -> str:
        """
        Generate a formatted list of all tools for inclusion in prompts.
        
        Args:
            format_type: Format type ("markdown", "json", or "text")
            
        Returns:
            A formatted list of tools
        """
        return ToolPromptGenerator.generate_tools_list(
            self.get_all_tools(), format_type=format_type
        )
    
    def generate_system_prompt(self, template: str, format_type: str = "text") -> str:
        """
        Generate a system prompt that includes tool descriptions.
        
        Args:
            template: Template string with {tools} placeholder
            format_type: Format for the tools list
            
        Returns:
            A system prompt with tool descriptions
        """
        return ToolPromptGenerator.generate_system_prompt_with_tools(
            self.get_all_tools(), template, format_type
        )
    
    def get_tool_snippet(self, tool_name: str) -> str:
        """
        Get a formatted snippet for a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            A formatted tool snippet
            
        Raises:
            KeyError: If the tool is not registered
        """
        tool = self.get_tool(tool_name)
        return ToolPromptGenerator.generate_full_tool_snippet(tool)
    
    def filter_tools(self, query: Optional[str] = None, 
                   categories: Optional[List[str]] = None) -> List[BaseTool]:
        """
        Filter tools by query or categories.
        
        Args:
            query: Optional text query to filter tools by name or description
            categories: Optional list of categories to filter by
            
        Returns:
            List of tools that match the filter criteria
        """
        tools = self.get_all_tools()
        
        # Filter by query
        if query:
            query = query.lower()
            tools = [
                tool for tool in tools
                if query in tool.name.lower() or query in tool.description.lower()
            ]
        
        # Filter by categories if implemented in the future
        if categories:
            # Future enhancement: Filter by tool categories when supported
            pass
        
        return tools