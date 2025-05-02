"""Integration test demonstrating the ReasoningAgent with chain-of-thought capabilities."""

import unittest
from typing import Dict, List

from hotlm_agents import tool
from hotlm_agents.agents.reasoning import PlanAndExecuteAgent, ReasoningAgent
from hotlm_core.models import BaseChatModel
from hotlm_core.prompts import ChatPromptTemplate
from hotlm_core.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage


# Mock LLM for testing that simulates chain-of-thought reasoning
class MockReasoningChatModel(BaseChatModel):
    """Mock chat model that returns chain-of-thought responses for testing."""

    def __init__(self, responses=None):
        self.responses = responses or {}
        self.invoke_count = 0
        self.history = []  # Track conversation history for multi-turn reasoning

    def invoke(self, messages: List[BaseMessage], **kwargs) -> AIMessage:
        """Return a pre-defined response based on the input and conversation history."""
        self.invoke_count += 1

        # Add messages to history for context tracking
        user_message = None
        for msg in messages:
            if isinstance(msg, HumanMessage):
                user_message = msg.content.strip().lower()

        # Track which turn we're on in the conversation to simulate multi-step reasoning
        turn_count = len([m for m in self.history if isinstance(m, HumanMessage)])

        # Add the current message batch to history
        self.history.extend(messages)

        # Determine response based on user message and turn
        if user_message in self.responses:
            responses = self.responses[user_message]
            # If we have a list of responses, use turn count to determine which one
            if isinstance(responses, list):
                if turn_count < len(responses):
                    return AIMessage(content=responses[turn_count])
                else:
                    return AIMessage(content="I've completed my reasoning.")
            else:
                return AIMessage(content=responses)

        # Default response
        return AIMessage(content="I'm thinking about this problem step by step...")

    async def ainvoke(self, messages: List[BaseMessage], **kwargs) -> AIMessage:
        """Async version of invoke for testing."""
        return self.invoke(messages, **kwargs)


# Define mock tools for the reasoning agent
@tool
def search(query: str) -> str:
    """Search for information on a topic."""
    if "population of france" in query.lower():
        return (
            "The population of France is approximately 67.75 million (2023 estimate)."
        )
    elif "capital of france" in query.lower():
        return "The capital of France is Paris."
    elif "france" in query.lower():
        return "France is a country in Western Europe with a population of about 67.75 million. Its capital is Paris."
    else:
        return f"No information found for: {query}"


@tool
def calculate(expression: str) -> float:
    """Calculate the result of a mathematical expression."""
    try:
        return eval(expression, {"__builtins__": {}}, {"abs": abs})
    except Exception as e:
        return f"Error: {str(e)}"


class TestReasoningAgent(unittest.TestCase):
    """Test cases for the ReasoningAgent class."""

    def setUp(self):
        """Set up mock LLM with multi-turn reasoning responses."""
        # Define responses for chain-of-thought reasoning
        self.cot_responses = {
            "what is the population of france?": [
                # First response with reasoning and tool call
                """Let me think through this step by step:

1. I need to find the population of France.
2. I can search for this information using the search tool.

```json
{"tool": "search", "args": {"query": "population of France"}}
```""",
                # Second response after seeing tool output
                """Now I have the information:

The search tells me that the population of France is approximately 67.75 million (2023 estimate).

Therefore, the population of France is approximately 67.75 million people as of 2023.
""",
            ],
            "calculate the square root of 144 and add 10": [
                # First response shows reasoning steps
                """I'll break this down into steps:

1. First, I need to find the square root of 144.
2. Then, I need to add 10 to that result.

Step 1: Calculate the square root of 144 using the calculate tool.

```json
{"tool": "calculate", "args": {"expression": "144 ** 0.5"}}
```""",
                # Second response with final calculation
                """Now I'll continue with the calculation:

The square root of 144 is 12.0.

Next, I need to add 10 to this result:
12.0 + 10 = 22.0

```json
{"tool": "calculate", "args": {"expression": "12 + 10"}}
```""",
                # Final response
                """Great, I now have the complete answer:

The square root of 144 is 12.
Adding 10 to 12 gives us 22.

Therefore, the square root of 144 plus 10 equals 22.
""",
            ],
        }

        # Create the mock LLM
        self.mock_llm = MockReasoningChatModel(responses=self.cot_responses)

        # Create the reasoning agent
        self.agent = ReasoningAgent(
            llm=self.mock_llm,
            tools=[search, calculate],
            reasoning_strategy="cot",  # Chain of thought reasoning
            max_iterations=5,
            return_intermediate_steps=True,  # Return the intermediate steps for testing
        )

    def test_search_reasoning(self):
        """Test the agent using search with chain-of-thought reasoning."""
        response = self.agent.run("What is the population of France?")

        # Check that we got the final result
        self.assertIsInstance(
            response, dict
        )  # Should return dict with intermediate steps
        self.assertIn("output", response)
        self.assertIn("67.75 million", response["output"])

        # Check that we made the correct number of LLM calls
        self.assertEqual(
            self.mock_llm.invoke_count, 2
        )  # Initial reasoning + final answer

    def test_multi_step_calculation(self):
        """Test a multi-step calculation with chain-of-thought reasoning."""
        response = self.agent.run("Calculate the square root of 144 and add 10")

        # Verify final answer
        self.assertIsInstance(response, dict)
        self.assertIn("output", response)
        self.assertIn("22", response["output"])

        # Check intermediate steps
        self.assertIn("intermediate_steps", response)
        # Should include calculator calls
        tool_names = [step[0] for step in response["intermediate_steps"]]
        self.assertIn("calculate", tool_names)


class TestPlanAndExecuteAgent(unittest.TestCase):
    """Test cases for the PlanAndExecuteAgent."""

    def setUp(self):
        """Set up mock LLM with planning and execution responses."""
        # Define responses for plan and execute
        self.plan_responses = {
            "tell me about france and its population": [
                # First response with planning
                """I'll solve this step by step:

PLAN:
1. Search for general information about France
2. Search specifically for population data
3. Compile the information into a comprehensive answer

Let me start with step 1:

```json
{"tool": "search", "args": {"query": "France general information"}}
```""",
                # Second response after first search
                """Great, I have some general information about France.

Now for step 2, I'll search for more specific population data:

```json
{"tool": "search", "args": {"query": "population of France"}}
```""",
                # Final response with compiled answer
                """Now I have all the information I need to answer the question:

France is a country in Western Europe with a population of approximately 67.75 million people as of 2023. The capital of France is Paris.

This completes all the steps in my plan.
""",
            ]
        }

        # Create the mock LLM
        self.mock_llm = MockReasoningChatModel(responses=self.plan_responses)

        # Create the plan and execute agent
        self.agent = PlanAndExecuteAgent(
            llm=self.mock_llm,
            tools=[search],
            max_iterations=5,
            return_intermediate_steps=True,
        )

    def test_plan_and_execute(self):
        """Test the agent's planning and execution abilities."""
        response = self.agent.run("Tell me about France and its population")

        # Check final output
        self.assertIsInstance(response, dict)
        self.assertIn("output", response)
        self.assertIn("67.75 million", response["output"])
        self.assertIn("Western Europe", response["output"])

        # Should have made multiple LLM calls for the different reasoning steps
        self.assertGreaterEqual(self.mock_llm.invoke_count, 2)


if __name__ == "__main__":
    unittest.main()
