"""
Orchestrator for the two-step "plan then act" pattern.

This module provides a high-level orchestrator that combines a planner and an executor
to implement a complete "plan then act" workflow.
"""

import logging
import uuid
from typing import Any, Callable, Dict, List, Optional, Type, Union

from hotlm_agents.planning.executor import (
    BaseExecutor,
    ParallelExecutor,
    SimpleExecutor,
)
from hotlm_agents.planning.plan import Plan, PlanResult
from hotlm_agents.planning.planner import BasePlanner, LLMPlanner
from hotlm_agents.tooling.registry import ToolRegistry
from hotlm_core.models import BaseChatModel
from hotlm_core.tools import BaseTool
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class OrchestrationResult(BaseModel):
    """Result of a complete planning and execution cycle."""
    query: str
    plan_result: PlanResult
    final_answer: Optional[str] = None

class PlannerExecutorOrchestrator:
    """
    Orchestrator for the two-step "plan then act" pattern.
    
    This class coordinates the planning and execution phases, provides hooks for
    customization, and manages the overall workflow.
    """
    
    def __init__(
        self,
        planner: Optional[BasePlanner] = None,
        executor: Optional[BaseExecutor] = None,
        model: Optional[BaseChatModel] = None,
        tool_registry: Optional[ToolRegistry] = None,
        synthesize_answer: bool = True,
    ):
        """
        Initialize the orchestrator.
        
        Args:
            planner: The planner to use for generating plans
            executor: The executor to use for executing plans
            model: LLM model to use (required if planner is not provided or for answer synthesis)
            tool_registry: Registry of available tools
            synthesize_answer: Whether to synthesize a final answer from execution results
        """
        self.tool_registry = tool_registry or ToolRegistry()
        
        # Set up planner (create a default if none is provided)
        if planner is None:
            if model is None:
                raise ValueError("Either a planner or a model must be provided")
            self.planner = LLMPlanner(model)
        else:
            self.planner = planner
        
        # Set up executor (create a default if none is provided)
        self.executor = executor or SimpleExecutor()
        
        self.model = model
        self.synthesize_answer = synthesize_answer
        
        # Hooks for customization
        self._pre_planning_hook = None
        self._post_planning_hook = None
        self._pre_execution_hook = None
        self._post_execution_hook = None
        self._answer_synthesis_hook = None
    
    def register_tool(self, tool: Union[BaseTool, Type[BaseTool]]) -> None:
        """
        Register a tool with the orchestrator.
        
        Args:
            tool: The tool to register
        """
        self.tool_registry.register(tool)
    
    def register_tools(self, tools: List[Union[BaseTool, Type[BaseTool]]]) -> None:
        """
        Register multiple tools with the orchestrator.
        
        Args:
            tools: The tools to register
        """
        for tool in tools:
            self.tool_registry.register(tool)
    
    # Hook setters for customization
    def set_pre_planning_hook(self, hook: Callable[[str, ToolRegistry], str]):
        """Set hook to run before planning, can modify the query."""
        self._pre_planning_hook = hook
    
    def set_post_planning_hook(self, hook: Callable[[Plan, str, ToolRegistry], Plan]):
        """Set hook to run after planning, can modify the plan."""
        self._post_planning_hook = hook
    
    def set_pre_execution_hook(self, hook: Callable[[Plan, ToolRegistry], Plan]):
        """Set hook to run before execution, can modify the plan."""
        self._pre_execution_hook = hook
    
    def set_post_execution_hook(self, hook: Callable[[PlanResult, str, ToolRegistry], PlanResult]):
        """Set hook to run after execution, can modify the result."""
        self._post_execution_hook = hook
    
    def set_answer_synthesis_hook(self, hook: Callable[[PlanResult, str, ToolRegistry], str]):
        """Set hook to synthesize the final answer, replacing the default."""
        self._answer_synthesis_hook = hook
    
    async def run(self, query: str) -> OrchestrationResult:
        """
        Run the full planning and execution workflow.
        
        Args:
            query: The user query or objective to fulfill
            
        Returns:
            The result of the orchestration process
        """
        # Apply pre-planning hook if set
        if self._pre_planning_hook:
            query = self._pre_planning_hook(query, self.tool_registry)
        
        # Generate a plan
        logger.info(f"Generating plan for query: {query}")
        plan = await self.planner.create_plan(query, self.tool_registry)
        
        # Apply post-planning hook if set
        if self._post_planning_hook:
            plan = self._post_planning_hook(plan, query, self.tool_registry)
        
        # Apply pre-execution hook if set
        if self._pre_execution_hook:
            plan = self._pre_execution_hook(plan, self.tool_registry)
        
        # Execute the plan
        logger.info(f"Executing plan with {len(plan.steps)} steps")
        plan_result = await self.executor.execute_plan(plan, self.tool_registry)
        
        # Apply post-execution hook if set
        if self._post_execution_hook:
            plan_result = self._post_execution_hook(plan_result, query, self.tool_registry)
        
        # Generate a final answer if required
        final_answer = None
        if self.synthesize_answer:
            if self._answer_synthesis_hook:
                final_answer = self._answer_synthesis_hook(plan_result, query, self.tool_registry)
            else:
                final_answer = await self._synthesize_answer(plan_result, query)
        
        return OrchestrationResult(
            query=query,
            plan_result=plan_result,
            final_answer=final_answer
        )
    
    async def _synthesize_answer(self, plan_result: PlanResult, query: str) -> str:
        """
        Synthesize a final answer from the execution results.
        
        Args:
            plan_result: The result of the plan execution
            query: The original user query
            
        Returns:
            A synthesized answer to the user's query
        """
        if not self.model:
            return plan_result.summary or "No summary available"
        
        # Create a prompt for answer synthesis
        step_results = []
        for step in plan_result.plan.steps:
            status = step.status.value.upper()
            step_info = f"Step {step.id}: {status} - {step.description}"
            if step.result:
                step_info += f"\nResult: {step.result}"
            step_results.append(step_info)
        
        prompt = f"""
        Based on the following execution results, provide a concise answer to the user's query.
        
        User query: {query}
        
        Plan description: {plan_result.plan.description}
        
        Execution results:
        {"".join(f"\n{result}" for result in step_results)}
        
        Please provide a clear and direct answer to the user's query based on these results.
        """
        
        # Get the LLM to synthesize an answer
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        response = await self.model.agenerate(messages=messages)
        return response.generations[0].text