"""
Example demonstrating planning strategy swapping and tool prompt generation.

This example showcases:
1. Using the PlanningStrategyFactory to switch between planning algorithms
2. Using the ToolPromptGenerator to create prompt snippets from tool metadata
3. Comparing different planning strategies on the same task
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from hotlm_agents.planning.executor import ParallelExecutor
from hotlm_agents.planning.factory import PlanningStrategyFactory
from hotlm_agents.planning.orchestrator import PlannerExecutorOrchestrator
from hotlm_agents.tooling.prompt_generator import ToolPromptGenerator
from hotlm_agents.tooling.registry import ToolRegistry

# Import necessary components
from hotlm_core.models import OpenAIModel
from hotlm_core.tools import BaseTool


# Define example tools
class WeatherTool(BaseTool):
    """Tool for checking weather information."""

    name = "weather"
    description = "Get current weather information for a specified location"

    def get_args_schema(self):
        return {
            "location": {
                "type": "string",
                "description": "The city or location to check weather for",
            }
        }

    async def ainvoke(self, args):
        location = args["location"]
        # Simulate API call with dummy data
        weather_data = {
            "New York": "72°F, Partly Cloudy",
            "San Francisco": "64°F, Foggy",
            "London": "58°F, Rainy",
            "Tokyo": "77°F, Sunny",
            "Sydney": "70°F, Clear",
        }
        return weather_data.get(location, f"Weather data not available for {location}")


class RestaurantTool(BaseTool):
    """Tool for finding restaurant recommendations."""

    name = "restaurants"
    description = "Find restaurant recommendations based on location and cuisine type"

    def get_args_schema(self):
        return {
            "location": {
                "type": "string",
                "description": "The city or neighborhood to search in",
            },
            "cuisine": {
                "type": "string",
                "description": "Type of cuisine (e.g., Italian, Japanese, Vegetarian)",
            },
            "price_range": {
                "type": "string",
                "description": "Price range from $ to $$$$ (optional)",
                "required": False,
            },
        }

    async def ainvoke(self, args):
        location = args["location"]
        cuisine = args["cuisine"]
        price_range = args.get("price_range", "$$")

        # Simulate API call with dummy data
        return f"Found 5 {cuisine} restaurants in {location} with price range {price_range}"


class CalendarTool(BaseTool):
    """Tool for managing calendar events."""

    name = "calendar"
    description = "Check availability and schedule events on a calendar"

    def __init__(self):
        self.events = {}

    def get_args_schema(self):
        return {
            "action": {
                "type": "string",
                "description": "The action to perform: 'check' or 'schedule'",
            },
            "date": {"type": "string", "description": "The date in YYYY-MM-DD format"},
            "time": {
                "type": "string",
                "description": "The time in HH:MM format (optional for 'check')",
                "required": False,
            },
            "duration": {
                "type": "integer",
                "description": "Duration in minutes (only for 'schedule')",
                "required": False,
            },
            "title": {
                "type": "string",
                "description": "Title of the event (only for 'schedule')",
                "required": False,
            },
        }

    async def ainvoke(self, args):
        action = args["action"]
        date = args["date"]

        if action == "check":
            events = self.events.get(date, [])
            if events:
                return f"Events on {date}: {json.dumps(events)}"
            else:
                return f"No events scheduled for {date}"

        elif action == "schedule":
            time = args.get("time", "12:00")
            duration = args.get("duration", 60)
            title = args.get("title", "Untitled Event")

            if date not in self.events:
                self.events[date] = []

            self.events[date].append(
                {"time": time, "duration": duration, "title": title}
            )

            return f"Scheduled '{title}' on {date} at {time} for {duration} minutes"

        return "Invalid action. Use 'check' or 'schedule'."


class NoteTool(BaseTool):
    """Tool for taking and retrieving notes."""

    name = "notes"
    description = "Create and retrieve text notes"

    def __init__(self):
        self.notes = {}

    def get_args_schema(self):
        return {
            "action": {
                "type": "string",
                "description": "The action to perform: 'create', 'get', or 'list'",
            },
            "title": {
                "type": "string",
                "description": "Title/identifier for the note (required for 'create' and 'get')",
                "required": False,
            },
            "content": {
                "type": "string",
                "description": "Content of the note (only for 'create')",
                "required": False,
            },
        }

    async def ainvoke(self, args):
        action = args["action"]

        if action == "create":
            title = args.get("title", f"Note-{len(self.notes)+1}")
            content = args.get("content", "")
            self.notes[title] = content
            return f"Created note '{title}'"

        elif action == "get":
            title = args.get("title")
            if not title:
                return "Error: title required for 'get' action"

            content = self.notes.get(title)
            if content is None:
                return f"No note found with title '{title}'"
            return f"Note '{title}': {content}"

        elif action == "list":
            titles = list(self.notes.keys())
            if not titles:
                return "No notes available"
            return f"Available notes: {', '.join(titles)}"

        return "Invalid action. Use 'create', 'get', or 'list'."


async def compare_planning_strategies(
    query: str, tool_registry: ToolRegistry, model: OpenAIModel
):
    """
    Compare different planning strategies on the same task.

    Args:
        query: The query to plan for
        tool_registry: Registry of available tools
        model: The language model to use
    """
    results = {}

    # Available planning strategies
    strategies = PlanningStrategyFactory.list_available_strategies()
    logger.info(f"Available planning strategies: {strategies}")

    for strategy_name, description in strategies.items():
        logger.info(f"\n\n--- Testing {strategy_name} planning strategy ---")
        logger.info(f"Description: {description}")

        # Create planner using the factory
        planner = PlanningStrategyFactory.create(
            strategy_type=strategy_name, model=model
        )

        # Create executor and orchestrator
        executor = ParallelExecutor(max_concurrency=3)
        orchestrator = PlannerExecutorOrchestrator(
            planner=planner,
            executor=executor,
            model=model,
            tool_registry=tool_registry,
            synthesize_answer=True,
        )

        # Track execution time
        start_time = time.time()

        # Run the orchestrator
        result = await orchestrator.run(query)

        # Calculate execution time
        execution_time = time.time() - start_time

        # Store results
        results[strategy_name] = {
            "execution_time": execution_time,
            "plan_steps": len(result.plan_result.plan.steps),
            "success": result.plan_result.success,
            "plan": result.plan_result.plan,
        }

        # Print results
        logger.info(f"Execution time: {execution_time:.2f} seconds")
        logger.info(f"Plan steps: {len(result.plan_result.plan.steps)}")
        logger.info(f"Success: {result.plan_result.success}")
        logger.info("\nPlan:")
        for step in result.plan_result.plan.steps:
            logger.info(f"- {step.id}: {step.tool_name} - {step.description}")
            logger.info(f"  Status: {step.status.value}")
            if step.result:
                logger.info(f"  Result: {step.result[:100]}...")

        logger.info("\nFinal Answer:")
        logger.info(result.final_answer)

    return results


def demonstrate_tool_prompts(tools: List[BaseTool]):
    """
    Demonstrate the tool prompt generation capabilities.

    Args:
        tools: List of tools to generate prompts for
    """
    logger.info("\n\n=== Tool Prompt Generation Examples ===\n")

    # Example 1: Generate a complete snippet for a single tool
    weather_tool = next(tool for tool in tools if tool.name == "weather")
    tool_snippet = ToolPromptGenerator.generate_full_tool_snippet(weather_tool)
    logger.info("Example 1: Complete Tool Snippet")
    logger.info(tool_snippet)

    # Example 2: Generate a markdown-formatted list of all tools
    logger.info("\nExample 2: Markdown-formatted Tool List")
    md_list = ToolPromptGenerator.generate_tools_list(tools, format_type="markdown")
    logger.info(md_list)

    # Example 3: Generate a JSON-formatted list of all tools
    logger.info("\nExample 3: JSON-formatted Tool List")
    json_list = ToolPromptGenerator.generate_tools_list(tools, format_type="json")
    logger.info(json_list)

    # Example 4: Generate a system prompt with tools
    logger.info("\nExample 4: System Prompt with Tools")
    system_prompt_template = """
You are an AI assistant that can help users with various tasks.
You have access to the following tools:

{tools}

When a user asks you to perform a task, analyze their request and use the appropriate tool.
    """
    system_prompt = ToolPromptGenerator.generate_system_prompt_with_tools(
        tools, system_prompt_template, format_type="text"
    )
    logger.info(system_prompt)


async def main():
    """Main function to run the examples."""
    # Initialize the language model
    model = OpenAIModel(model_name="gpt-4")  # Replace with your preferred model

    # Create tools
    weather_tool = WeatherTool()
    restaurant_tool = RestaurantTool()
    calendar_tool = CalendarTool()
    note_tool = NoteTool()

    # Initialize tool registry
    tool_registry = ToolRegistry()
    tool_registry.register(weather_tool)
    tool_registry.register(restaurant_tool)
    tool_registry.register(calendar_tool)
    tool_registry.register(note_tool)

    # Demonstrate tool prompt generation
    demonstrate_tool_prompts([weather_tool, restaurant_tool, calendar_tool, note_tool])

    # Define a query that requires planning
    query = (
        "I'll be visiting Tokyo next week. Find me the weather forecast, "
        "recommend some Japanese restaurants, and create a note with this information"
    )

    # Compare different planning strategies
    await compare_planning_strategies(query, tool_registry, model)


if __name__ == "__main__":
    asyncio.run(main())
