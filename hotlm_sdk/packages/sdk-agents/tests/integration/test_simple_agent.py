"""Integration test demonstrating a simple agent with tools."""

import unittest
from typing import Dict

from hotlm_agents import tool
from hotlm_agents.agents.base import SimpleAgent
from hotlm_core.models import BaseChatModel
from hotlm_core.prompts import ChatPromptTemplate
from hotlm_core.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage


# Mock LLM for testing that returns pre-defined responses
class MockChatModel(BaseChatModel):
    """Mock chat model that returns pre-defined responses for testing."""
    
    def __init__(self, responses=None):
        self.responses = responses or {}
        self.invoke_count = 0
        self.last_messages = None
    
    def invoke(self, messages: list[BaseMessage], **kwargs) -> AIMessage:
        """Return a pre-defined response based on the input."""
        self.invoke_count += 1
        self.last_messages = messages
        
        # Convert messages to a string key for lookup
        key = self._get_message_key(messages)
        
        if key in self.responses:
            return AIMessage(content=self.responses[key])
        
        # Default message if no match found
        return AIMessage(content="I don't know how to respond to that.")
    
    async def ainvoke(self, messages: list[BaseMessage], **kwargs) -> AIMessage:
        """Async version of invoke for testing."""
        return self.invoke(messages, **kwargs)
    
    def _get_message_key(self, messages: list[BaseMessage]) -> str:
        """Create a simple string key from the messages for lookup."""
        # For simple tests, just look at the last user message
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                return msg.content.strip().lower()
        return ""


# Define some mock tools for testing
@tool
def calculator(expression: str) -> float:
    """Calculate the result of a mathematical expression."""
    try:
        return eval(expression, {"__builtins__": {}}, {"abs": abs})
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def weather(location: str) -> str:
    """Get the current weather for a location."""
    # In a real implementation, this would call a weather API
    return f"The weather in {location} is sunny and 75 degrees."


class TestSimpleAgent(unittest.TestCase):
    """Test cases for a simple agent with tools."""
    
    def setUp(self):
        """Set up mock LLM with pre-defined responses for testing."""
        # Define responses for specific inputs
        self.responses = {
            "what is 2+2?": """I'll use the calculator tool to solve this.
```json
{"tool": "calculator", "args": {"expression": "2+2"}}
```""",
            
            "what's the weather like in new york?": """I'll check the weather for you.
```json
{"tool": "weather", "args": {"location": "New York"}}
```""",
            
            "hello": "Hello there! How can I help you today?",
        }
        
        # Create a MockChatModel with our predefined responses
        self.mock_llm = MockChatModel(responses=self.responses)
        
        # Create system prompt
        system_message = """You are a helpful assistant that can use tools.
        When you need to use a tool, format your response as JSON with "tool" and "args" keys.
        """
        
        # Create prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_message),
            {"role": "placeholder", "content": "{messages}"},
        ])
        
        # Create the agent
        self.agent = SimpleAgent(
            llm=self.mock_llm,
            prompt_template=self.prompt_template,
            tools=[calculator, weather],
        )
    
    def test_simple_reply(self):
        """Test agent providing a direct answer without using tools."""
        response = self.agent.run("hello")
        self.assertEqual(response, "Hello there! How can I help you today?")
        self.assertEqual(self.mock_llm.invoke_count, 1)
    
    def test_calculator_tool_usage(self):
        """Test agent using the calculator tool."""
        response = self.agent.run("What is 2+2?")
        self.assertEqual(response, "4.0")  # Result after tool execution
        
        # Should have made 2 LLM calls: 1 for the tool call, potentially 1 for final answer
        # but depends on implementation
        self.assertGreaterEqual(self.mock_llm.invoke_count, 1)
    
    def test_weather_tool_usage(self):
        """Test agent using the weather tool."""
        response = self.agent.run("What's the weather like in New York?")
        self.assertIn("sunny and 75 degrees", response)  # Result after tool execution
        self.assertGreaterEqual(self.mock_llm.invoke_count, 1)


if __name__ == "__main__":
    unittest.main()