"""
Example demonstrating the beam search planner with the orchestrator.

This example shows how to use the BeamSearchPlanner with the PlannerExecutorOrchestrator
to generate more effective plans by exploring multiple planning paths.
"""

import asyncio
import logging
from pprint import pprint

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

from hotlm_agents.planning.executor import ParallelExecutor
from hotlm_agents.planning.orchestrator import PlannerExecutorOrchestrator
from hotlm_agents.planning.planner import BeamSearchPlanner
from hotlm_agents.tooling.registry import ToolRegistry

# Import necessary components
from hotlm_core.models import OpenAIModel
from hotlm_core.tools import BaseTool


# Define some example tools
class CalculatorTool(BaseTool):
    """Calculator tool for arithmetic operations."""
    
    name = "calculator"
    description = "Performs arithmetic calculations"
    
    def get_args_schema(self):
        return {
            "operation": {"type": "string", "description": "Operation to perform (add, subtract, multiply, divide)"},
            "x": {"type": "number", "description": "First operand"},
            "y": {"type": "number", "description": "Second operand"}
        }
    
    async def ainvoke(self, args):
        op = args["operation"]
        x = args["x"]
        y = args["y"]
        
        if op == "add":
            result = x + y
        elif op == "subtract":
            result = x - y
        elif op == "multiply":
            result = x * y
        elif op == "divide":
            result = x / y
        else:
            raise ValueError(f"Unknown operation: {op}")
            
        print(f"Calculator: {x} {op} {y} = {result}")
        return str(result)

class SearchTool(BaseTool):
    """Simulated search tool."""
    
    name = "search"
    description = "Search for information on a topic"
    
    def get_args_schema(self):
        return {
            "query": {"type": "string", "description": "Search query"}
        }
    
    async def ainvoke(self, args):
        query = args["query"]
        print(f"Searching for: {query}")
        
        # Simulate search results
        if "weather" in query.lower():
            return "The weather is sunny with a high of 75°F."
        elif "population" in query.lower():
            return "The population is approximately 8 billion people worldwide."
        else:
            return f"Search results for: {query}"

class WeatherTool(BaseTool):
    """Weather information tool."""
    
    name = "weather"
    description = "Get current weather for a location"
    
    def get_args_schema(self):
        return {
            "location": {"type": "string", "description": "City or location name"}
        }
    
    async def ainvoke(self, args):
        location = args["location"]
        print(f"Getting weather for: {location}")
        
        # Simulate weather data
        weather_data = {
            "New York": "72°F, Partly Cloudy",
            "London": "65°F, Rainy",
            "Tokyo": "78°F, Sunny",
            "Sydney": "68°F, Clear"
        }
        
        return weather_data.get(location, f"Weather data not available for {location}")

class StorageTool(BaseTool):
    """Simple key-value storage tool."""
    
    name = "storage"
    description = "Store and retrieve data by key"
    
    def __init__(self):
        self.data = {}
    
    def get_args_schema(self):
        return {
            "action": {"type": "string", "description": "Action to perform (get, set, list)"},
            "key": {"type": "string", "description": "Key for the data (for get/set)"},
            "value": {"type": "string", "description": "Value to store (for set)", "required": False}
        }
    
    async def ainvoke(self, args):
        action = args["action"]
        
        if action == "get":
            key = args["key"]
            return self.data.get(key, f"No data found for key: {key}")
        
        elif action == "set":
            key = args["key"]
            value = args["value"]
            self.data[key] = value
            return f"Stored value for key: {key}"
        
        elif action == "list":
            return str(list(self.data.keys()))
        
        else:
            raise ValueError(f"Unknown action: {action}")

# Custom evaluation function for the beam search planner
def plan_evaluator(plan):
    """
    Evaluate the quality of a plan (0.0 to 1.0).
    
    This function scores plans based on:
    1. Number of steps (rewards more comprehensive plans)
    2. Tool variety (rewards using diverse tools)
    3. Dependency structure (rewards plans with good step dependencies)
    """
    # No steps means zero score
    if len(plan.steps) == 0:
        return 0.0
    
    # Count unique tools used
    unique_tools = set(step.tool_name for step in plan.steps)
    tool_variety_score = min(len(unique_tools) / 3, 1.0)  # Cap at 1.0
    
    # Evaluate step count (prefer 3-5 steps)
    step_count = len(plan.steps)
    if step_count <= 2:
        step_count_score = step_count / 3.0
    elif step_count <= 5:
        step_count_score = 1.0
    else:
        step_count_score = 5.0 / step_count  # Penalize for too many steps
    
    # Evaluate dependencies (reward non-trivial dependency structures)
    total_deps = sum(len(step.depends_on) for step in plan.steps)
    if step_count <= 1:
        dependency_score = 0.0
    else:
        ideal_deps = max(1, step_count - 1)  # At least some dependencies, but not all-to-all
        dependency_score = min(total_deps / ideal_deps, 1.5)  # Allow slight bonus for good structure
    
    # Weight the different factors
    final_score = (
        step_count_score * 0.35 +  # 35% weight on step count
        tool_variety_score * 0.35 +  # 35% weight on tool variety
        dependency_score * 0.30  # 30% weight on dependencies
    )
    
    return min(final_score, 1.0)  # Ensure score is between 0 and 1

# Custom hook to enhance the final answer
def answer_synthesis_hook(plan_result, query, tool_registry):
    """Custom hook to create a more detailed final answer."""
    success_count = sum(1 for step in plan_result.plan.steps if step.status.value == "succeeded")
    fail_count = sum(1 for step in plan_result.plan.steps if step.status.value == "failed")
    
    if fail_count > 0:
        failed_steps = [step for step in plan_result.plan.steps if step.status.value == "failed"]
        failed_info = "\n".join([f"- {step.description}: {step.error}" for step in failed_steps])
        return f"""
I attempted to answer your query, but encountered {fail_count} issue(s):

{failed_info}

I was able to complete {success_count} step(s) successfully. Please refine your query or 
check if the required tools are available.
"""
    
    # For successful execution, build a summary from the results
    results = []
    for step in plan_result.plan.steps:
        if step.result:
            results.append(f"- {step.description}: {step.result}")
    
    return f"""
I've successfully addressed your query:

{plan_result.plan.description}

Here are the results:
{chr(10).join(results)}
"""

async def main():
    # Initialize the LLM (replace with your actual API key)
    model = OpenAIModel(model_name="gpt-4")  # Use appropriate model
    
    # Initialize tools and registry
    calculator = CalculatorTool()
    search = SearchTool()
    weather = WeatherTool()
    storage = StorageTool()
    
    tool_registry = ToolRegistry()
    tool_registry.register(calculator)
    tool_registry.register(search)
    tool_registry.register(weather)
    tool_registry.register(storage)
    
    # Initialize components with beam search planner
    planner = BeamSearchPlanner(
        model=model,
        beam_width=3,  # Consider 3 alternative plans at each step
        max_depth=3,   # Search up to depth 3
        evaluator=plan_evaluator  # Use custom evaluator
    )
    
    # Use parallel executor for efficiency
    executor = ParallelExecutor(max_concurrency=2)
    
    # Create the orchestrator
    orchestrator = PlannerExecutorOrchestrator(
        planner=planner,
        executor=executor,
        model=model,
        tool_registry=tool_registry,
        synthesize_answer=True
    )
    
    # Set custom answer synthesis
    orchestrator.set_answer_synthesis_hook(answer_synthesis_hook)
    
    # Run the orchestrator with a complex query
    query = "Calculate the sum of 42 and 17, then get the weather in Tokyo, and store both results"
    print(f"\nExecuting query: {query}\n")
    
    result = await orchestrator.run(query)
    
    # Print the results
    print("\n" + "="*50)
    print("FINAL ANSWER:")
    print(result.final_answer)
    print("="*50)
    
    # Print the executed plan for reference
    print("\nExecuted Plan:")
    print(f"Description: {result.plan_result.plan.description}")
    print("\nSteps:")
    for step in result.plan_result.plan.steps:
        print(f"- {step.id}: {step.status.value.upper()}: {step.description}")
        if step.result:
            print(f"  Result: {step.result}")

if __name__ == "__main__":
    asyncio.run(main())