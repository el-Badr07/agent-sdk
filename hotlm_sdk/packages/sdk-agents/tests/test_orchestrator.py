"""
Tests for the planner/executor orchestrator.

This module tests the complete planning and execution workflow
implemented by the PlannerExecutorOrchestrator.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from hotlm_agents.planning.executor import SimpleExecutor
from hotlm_agents.planning.orchestrator import PlannerExecutorOrchestrator
from hotlm_agents.planning.plan import Plan, PlanStep, StepStatus
from hotlm_agents.planning.planner import LLMPlanner
from hotlm_agents.tooling.registry import ToolRegistry
from hotlm_core.models import BaseChatModel, ModelGeneration, ModelResponse
from hotlm_core.tools import BaseTool


# Create mock tools for testing
class MockCalculatorTool(BaseTool):
    name = "calculator"
    description = "Performs basic arithmetic operations"
    
    def __init__(self):
        self.calls = []
        
    def get_args_schema(self):
        return {
            "operation": {"type": "string", "description": "The operation to perform (add, subtract, multiply, divide)"},
            "x": {"type": "number", "description": "First operand"},
            "y": {"type": "number", "description": "Second operand"}
        }
    
    async def ainvoke(self, args):
        self.calls.append(args)
        
        op = args["operation"]
        x = args["x"]
        y = args["y"]
        
        if op == "add":
            return str(x + y)
        elif op == "subtract":
            return str(x - y)
        elif op == "multiply":
            return str(x * y)
        elif op == "divide":
            return str(x / y)
        else:
            raise ValueError(f"Unknown operation: {op}")

class MockSearchTool(BaseTool):
    name = "search"
    description = "Searches for information on a topic"
    
    def __init__(self):
        self.calls = []
        
    def get_args_schema(self):
        return {
            "query": {"type": "string", "description": "The search query"}
        }
    
    async def ainvoke(self, args):
        self.calls.append(args)
        return f"Search results for: {args['query']}"

# Mock LLM for testing
class MockLLM(BaseChatModel):
    def __init__(self, plan_json):
        self.plan_json = plan_json
        self.calls = []
    
    async def agenerate(self, messages, **kwargs):
        self.calls.append(messages)
        # Return the preset plan JSON for planner calls
        if any("Create a plan" in msg.get("content", "") for msg in messages):
            return ModelResponse(generations=[ModelGeneration(text=self.plan_json)])
        # Return a summary for answer synthesis
        elif any("Based on the following execution results" in msg.get("content", "") for msg in messages):
            return ModelResponse(generations=[ModelGeneration(text="This is a summary of the execution results.")])
        else:
            return ModelResponse(generations=[ModelGeneration(text="Default response")])

@pytest.fixture
def mock_tools():
    calculator = MockCalculatorTool()
    search = MockSearchTool()
    return {"calculator": calculator, "search": search}

@pytest.fixture
def mock_llm():
    # Plan JSON that will be "generated" by the mock LLM
    plan_json = """
    {
      "description": "Calculate the sum of two numbers and find information about the result",
      "steps": [
        {
          "tool_name": "calculator",
          "tool_args": {
            "operation": "add",
            "x": 5,
            "y": 7
          },
          "description": "Calculate 5 + 7",
          "depends_on": []
        },
        {
          "tool_name": "search",
          "tool_args": {
            "query": "facts about the number 12"
          },
          "description": "Search for information about the number 12",
          "depends_on": ["1"]
        }
      ]
    }
    """
    return MockLLM(plan_json)

@pytest.mark.asyncio
async def test_orchestrator_basic_flow(mock_tools, mock_llm):
    # Set up the test objects
    tool_registry = ToolRegistry()
    for tool in mock_tools.values():
        tool_registry.register(tool)
    
    planner = LLMPlanner(mock_llm)
    executor = SimpleExecutor()
    
    # Create the orchestrator
    orchestrator = PlannerExecutorOrchestrator(
        planner=planner,
        executor=executor,
        model=mock_llm,
        tool_registry=tool_registry,
        synthesize_answer=True
    )
    
    # Run the orchestrator with a test query
    result = await orchestrator.run("Calculate 5 + 7 and find information about the result")
    
    # Check the result structure
    assert result.query == "Calculate 5 + 7 and find information about the result"
    assert result.plan_result is not None
    assert result.final_answer is not None
    
    # Check that the plan was created correctly
    plan = result.plan_result.plan
    assert len(plan.steps) == 2
    assert plan.steps[0].tool_name == "calculator"
    assert plan.steps[1].tool_name == "search"
    
    # Check that the tools were called correctly
    calculator = mock_tools["calculator"]
    search = mock_tools["search"]
    
    assert len(calculator.calls) == 1
    assert calculator.calls[0]["operation"] == "add"
    assert calculator.calls[0]["x"] == 5
    assert calculator.calls[0]["y"] == 7
    
    assert len(search.calls) == 1
    assert search.calls[0]["query"] == "facts about the number 12"
    
    # Check the status of steps
    assert plan.steps[0].status == StepStatus.SUCCEEDED
    assert plan.steps[1].status == StepStatus.SUCCEEDED
    
    # Check that the plan was executed successfully
    assert result.plan_result.success is True

@pytest.mark.asyncio
async def test_orchestrator_with_hooks(mock_tools, mock_llm):
    # Set up the test objects
    tool_registry = ToolRegistry()
    for tool in mock_tools.values():
        tool_registry.register(tool)
    
    planner = LLMPlanner(mock_llm)
    executor = SimpleExecutor()
    
    # Create the orchestrator
    orchestrator = PlannerExecutorOrchestrator(
        planner=planner,
        executor=executor,
        model=mock_llm,
        tool_registry=tool_registry,
        synthesize_answer=True
    )
    
    # Set up hooks to verify they're called
    pre_planning_called = False
    post_planning_called = False
    pre_execution_called = False
    post_execution_called = False
    answer_synthesis_called = False
    
    def pre_planning_hook(query, registry):
        nonlocal pre_planning_called
        pre_planning_called = True
        return f"{query} (pre-planning processed)"
    
    def post_planning_hook(plan, query, registry):
        nonlocal post_planning_called
        post_planning_called = True
        # Modify the plan description as a marker
        plan.description = f"{plan.description} (post-planning processed)"
        return plan
    
    def pre_execution_hook(plan, registry):
        nonlocal pre_execution_called
        pre_execution_called = True
        return plan
    
    def post_execution_hook(result, query, registry):
        nonlocal post_execution_called
        post_execution_called = True
        return result
    
    def answer_synthesis_hook(result, query, registry):
        nonlocal answer_synthesis_called
        answer_synthesis_called = True
        return "Custom synthesized answer"
    
    # Set the hooks
    orchestrator.set_pre_planning_hook(pre_planning_hook)
    orchestrator.set_post_planning_hook(post_planning_hook)
    orchestrator.set_pre_execution_hook(pre_execution_hook)
    orchestrator.set_post_execution_hook(post_execution_hook)
    orchestrator.set_answer_synthesis_hook(answer_synthesis_hook)
    
    # Run the orchestrator
    result = await orchestrator.run("Test query with hooks")
    
    # Check that all hooks were called
    assert pre_planning_called is True
    assert post_planning_called is True
    assert pre_execution_called is True
    assert post_execution_called is True
    assert answer_synthesis_called is True
    
    # Check that the hooks had an effect
    assert "post-planning processed" in result.plan_result.plan.description
    assert result.final_answer == "Custom synthesized answer"

@pytest.mark.asyncio
async def test_orchestrator_tool_failure():
    # Set up a tool that will fail
    failing_tool = MagicMock(spec=BaseTool)
    failing_tool.name = "failing_tool"
    failing_tool.description = "A tool that will fail"
    failing_tool.__contains__ = lambda self, item: item == "failing_tool"
    failing_tool.ainvoke.side_effect = Exception("Tool execution failed")
    failing_tool.get_args_schema.return_value = {"arg": {"type": "string"}}
    
    # Mock LLM that will return a plan using the failing tool
    mock_llm = MockLLM("""
    {
      "description": "Plan with a failing step",
      "steps": [
        {
          "tool_name": "failing_tool",
          "tool_args": {"arg": "value"},
          "description": "This step will fail",
          "depends_on": []
        }
      ]
    }
    """)
    
    # Set up registry and orchestrator
    tool_registry = ToolRegistry()
    tool_registry.register = MagicMock(return_value=None)
    tool_registry.get_tool = MagicMock(return_value=failing_tool)
    tool_registry.get_all_tools = MagicMock(return_value=[failing_tool])
    tool_registry.__contains__ = lambda self, item: item == "failing_tool"
    
    orchestrator = PlannerExecutorOrchestrator(
        model=mock_llm,
        tool_registry=tool_registry
    )
    
    # Run the orchestrator
    result = await orchestrator.run("Test with failing tool")
    
    # Check that the overall execution failed
    assert result.plan_result.success is False
    
    # Check that the step was marked as failed
    assert result.plan_result.plan.steps[0].status == StepStatus.FAILED
    assert "Tool execution failed" in result.plan_result.plan.steps[0].error