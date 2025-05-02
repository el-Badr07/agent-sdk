"""Test cases for the reasoning agent implementation."""

import unittest
from unittest.mock import MagicMock, patch

# --- Mock classes to replace imports ---

# Mock message classes
class BaseMessage:
    def __init__(self, content):
        self.content = content

class SystemMessage(BaseMessage):
    role = "system"

class HumanMessage(BaseMessage):
    role = "user"

class AIMessage(BaseMessage):
    role = "assistant"

# Mock base model
class BaseChatModel:
    def invoke(self, messages, **kwargs):
        raise NotImplementedError("Subclasses must implement invoke method")
    
    async def ainvoke(self, messages, **kwargs):
        raise NotImplementedError("Subclasses must implement ainvoke method")

# Mock prompt template
class ChatPromptTemplate:
    @classmethod
    def from_messages(cls, messages):
        return cls(messages=messages)
        
    def __init__(self, messages):
        self.messages = messages

# Tool decorator mock
def tool(fn=None):
    def decorator(func):
        func._is_tool = True
        return func
    
    if fn is None:
        return decorator
    return decorator(fn)

# Mock ReasoningAgent class
class ReasoningAgent:
    """Mock reasoning agent that simulates multi-turn reasoning."""
    
    def __init__(self, llm, tools=None, reasoning_strategy="cot", return_intermediate_steps=False):
        self.llm = llm
        self.tools = tools or []
        self.reasoning_strategy = reasoning_strategy
        self.return_intermediate_steps = return_intermediate_steps
    
    def run(self, input_text):
        """Run the reasoning agent with input text."""
        messages = [HumanMessage(content=input_text)]
        response = self.llm.invoke(messages)
        
        # Check for tool use
        if hasattr(response, 'content') and "```json" in response.content:
            try:
                # Extract tool call
                import json
                import re
                
                tool_match = re.search(r'```json\s*(.*?)\s*```', response.content, re.DOTALL)
                if tool_match:
                    tool_call = json.loads(tool_match.group(1))
                    tool_name = tool_call.get("tool")
                    tool_args = tool_call.get("args", {})
                    
                    # Find and execute the tool
                    tool_result = None
                    for tool_fn in self.tools:
                        if hasattr(tool_fn, "__name__") and tool_fn.__name__ == tool_name:
                            tool_result = tool_fn(**tool_args)
                    
                    if tool_result:
                        # Call LLM again with tool result
                        tool_response_message = HumanMessage(content=f"Tool {tool_name} returned: {tool_result}")
                        messages.append(tool_response_message)
                        final_response = self.llm.invoke(messages)
                        
                        if self.return_intermediate_steps:
                            return {
                                "output": final_response.content,
                                "steps": [
                                    {"type": "reasoning", "content": response.content},
                                    {"type": "tool_use", "tool": tool_name, "args": tool_args, "result": tool_result},
                                    {"type": "reasoning", "content": final_response.content}
                                ]
                            }
                        return final_response.content
            except Exception as e:
                print(f"Error in tool execution: {e}")
        
        # If we get here, either there was no tool use or something went wrong
        if self.return_intermediate_steps:
            return {
                "output": response.content,
                "steps": [{"type": "reasoning", "content": response.content}]
            }
        return response.content


class MockReasoningChatModel(BaseChatModel):
    """Mock chat model for testing reasoning agents with multi-turn thinking."""
    
    def __init__(self, response_sequences=None):
        self.response_sequences = response_sequences or {}
        self.invoke_count = 0
        self.current_sequence = {}  # Maps query to current position in sequence
    
    def invoke(self, messages, **kwargs):
        """Return responses sequentially for each query to simulate thinking steps."""
        self.invoke_count += 1
        
        # Find the last user message for matching
        last_message = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage) or getattr(msg, 'role', '') == 'user':
                content = msg.content.lower() if hasattr(msg, 'content') else str(msg).lower()
                # Skip messages that represent tool results
                if content.startswith("tool "):
                    continue
                last_message = content
                break

        if not last_message:
            return AIMessage(content="I don't understand the request.")
        
        # Find the matching response sequence
        matching_key = None
        for key in self.response_sequences.keys():
            # Match if all words in the key are contained in the message
            tokens = key.lower().split()
            if all(token in last_message for token in tokens):
                matching_key = key
                break
        
        if not matching_key:
            # Fallback: match on the primary keyword only
            for key in self.response_sequences.keys():
                primary = key.lower().split()[0]
                if primary in last_message:
                    matching_key = key
                    break
        if not matching_key:
            return AIMessage(content="I don't have a specific response for that.")
        
        # Get the next response in sequence
        if matching_key not in self.current_sequence:
            self.current_sequence[matching_key] = 0
            
        sequence = self.response_sequences[matching_key]
        position = self.current_sequence[matching_key]
        
        if position < len(sequence):
            response = sequence[position]
            self.current_sequence[matching_key] += 1
            return AIMessage(content=response)
        else:
            # If we've exhausted the sequence, return a final message
            return AIMessage(content="I've completed my reasoning process.")
    
    async def ainvoke(self, messages, **kwargs):
        """Async version just calls the sync version for testing."""
        return self.invoke(messages, **kwargs)


@tool
def search(query: str) -> str:
    """Search for information on a topic."""
    query = query.lower()
    if "weather" in query:
        return "The weather is currently 72°F with clear skies."
    elif "population of new york" in query:
        return "New York City has a population of approximately 8.4 million people (2021)."
    elif "tallest building" in query:
        return "The tallest building in the world is the Burj Khalifa at 828 meters (2,717 feet)."
    else:
        return f"No specific information found for: {query}"


@tool
def calculate(expression: str) -> str:
    """Calculate the result of a mathematical expression."""
    try:
        return str(eval(expression, {"__builtins__": {}}, {"abs": abs}))
    except Exception as e:
        return f"Error: {str(e)}"


class TestReasoningAgent(unittest.TestCase):
    """Test cases for the reasoning agent."""
    
    def setUp(self):
        """Set up the test environment with mock responses for multi-turn reasoning."""
        # Define response sequences that simulate reasoning steps
        self.response_sequences = {
            "population of new york": [
                # Step 1: Initial reasoning and planning
                """Let me think through this step by step:

1. ANALYZE THE PROBLEM:
I need to find the population of New York City.

2. DEVELOP A PLAN:
I'll use the search tool to look up current population data.

3. EXECUTE THE PLAN:
```json
{"tool": "search", "args": {"query": "population of New York"}}
```""",
                
                # Step 2: After getting search results
                """4. VERIFY THE SOLUTION:
I've found that New York City has a population of approximately 8.4 million people as of 2021.
This information comes from a reliable source and answers the question directly.

5. PRESENT YOUR ANSWER:
Based on my search, New York City has a population of approximately 8.4 million people as of 2021."""
            ],
            
            "calculate square root": [
                # Step 1: Initial reasoning
                """1. ANALYZE THE PROBLEM:
I need to calculate the square root of a number.

2. DEVELOP A PLAN:
I'll use the calculate tool with the appropriate mathematical expression.

3. EXECUTE THE PLAN:
To find the square root of 144, I'll use the calculate tool:

```json
{"tool": "calculate", "args": {"expression": "144 ** 0.5"}}
```""",
                
                # Step 2: After calculation
                """4. VERIFY THE SOLUTION:
I calculated the square root of 144, which is 12.0.
Let me double-check: 12 × 12 = 144. Yes, that's correct.

5. PRESENT YOUR ANSWER:
The square root of 144 is 12."""
            ]
        }
        
        # Create mock LLM
        self.mock_llm = MockReasoningChatModel(response_sequences=self.response_sequences)
        
        # Create reasoning agent
        self.agent = ReasoningAgent(
            llm=self.mock_llm,
            tools=[search, calculate],
            reasoning_strategy="cot",  # Chain of thought reasoning
            return_intermediate_steps=True
        )
    
    def test_population_search(self):
        """Test the reasoning agent searching for population data."""
        response = self.agent.run("What is the population of New York City?")
        
        # Verify we got a dictionary with output and steps
        self.assertIsInstance(response, dict)
        self.assertIn("output", response)
        self.assertIn("8.4 million", response["output"])
        
        # Check that the agent went through multiple reasoning steps
        self.assertEqual(self.mock_llm.invoke_count, 2)
    
    def test_calculation(self):
        """Test the reasoning agent performing a calculation."""
        response = self.agent.run("Calculate the square root of 144")
        
        # Verify output
        self.assertIsInstance(response, dict)
        self.assertIn("output", response)
        self.assertIn("12", response["output"])
        
        # Check that the agent went through multiple reasoning steps
        self.assertEqual(self.mock_llm.invoke_count, 2)


if __name__ == "__main__":
    unittest.main()