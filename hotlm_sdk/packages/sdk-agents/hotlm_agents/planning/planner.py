"""
Planners for generating execution plans.

This module defines the planner interfaces and implementations for generating
execution plans in the two-step "plan then act" pattern.
"""

import abc
from typing import Any, Callable, Dict, List, Optional

from hotlm_agents.planning.plan import Plan, PlanStep
from hotlm_agents.tooling.registry import ToolRegistry
from hotlm_core.models import BaseChatModel


class BasePlanner(abc.ABC):
    """
    Base class for plan generators.
    
    A planner is responsible for generating an execution plan to fulfill a given
    objective or query.
    """
    
    @abc.abstractmethod
    async def create_plan(self, query: str, tool_registry: ToolRegistry) -> Plan:
        """
        Create a plan to fulfill the given query or objective.
        
        Args:
            query: The query or objective to fulfill
            tool_registry: Registry of available tools
            
        Returns:
            An execution plan
        """
        pass


class LLMPlanner(BasePlanner):
    """
    LLM-based planner that generates plans using a language model.
    
    This planner uses an LLM to analyze the query, available tools, and
    generate a suitable execution plan.
    """
    
    def __init__(
        self, 
        model: BaseChatModel, 
        max_steps: int = 10,
        system_prompt_template: Optional[str] = None
    ):
        """
        Initialize the LLM planner.
        
        Args:
            model: The language model to use for planning
            max_steps: Maximum number of steps to include in a plan
            system_prompt_template: Optional custom system prompt template
        """
        self.model = model
        self.max_steps = max_steps
        self._system_prompt_template = system_prompt_template
    
    def _create_system_prompt(self, tool_registry: ToolRegistry) -> str:
        """
        Create a system prompt for the planning LLM.
        
        Args:
            tool_registry: Registry of available tools
            
        Returns:
            A system prompt
        """
        if self._system_prompt_template:
            tool_descriptions = ""
            for tool in tool_registry.get_all_tools():
                args_desc = []
                for name, schema in tool.get_args_schema().items():
                    args_desc.append(f"  - {name}: {schema.get('description', 'No description')} (type: {schema.get('type', 'any')})")
                
                tool_descriptions += f"""
Name: {tool.name}
Description: {tool.description}
Arguments:
{chr(10).join(args_desc)}
"""
            return self._system_prompt_template.format(tools=tool_descriptions)
        
        # Default system prompt
        tool_descriptions = []
        for tool in tool_registry.get_all_tools():
            args_desc = []
            for name, schema in tool.get_args_schema().items():
                args_desc.append(f"  - {name}: {schema.get('description', 'No description')} (type: {schema.get('type', 'any')})")
            
            tool_descriptions.append(f"""
- Tool: {tool.name}
  Description: {tool.description}
  Arguments:
{chr(10).join(args_desc)}
""")
        
        return f"""You are a planner AI that creates detailed execution plans to fulfill user objectives.
Your job is to analyze the user's request and create a step-by-step plan using available tools.

Available tools:
{"".join(tool_descriptions)}

Your plan should:
1. Break down the task into logical steps
2. Specify which tool to use for each step
3. Provide the exact arguments to pass to each tool
4. Specify dependencies between steps (which steps must complete before others can start)
5. Include a clear description of what each step accomplishes

Format your response as a valid JSON object with the following structure:
{{
  "description": "A description of the overall plan",
  "steps": [
    {{
      "tool_name": "name_of_tool",
      "tool_args": {{
        "arg1": "value1",
        "arg2": "value2"
      }},
      "description": "What this step accomplishes",
      "depends_on": ["step_id_1", "step_id_2"]  // IDs of steps that must complete first
    }}
  ]
}}

You can have up to {self.max_steps} steps in your plan. Ensure each step has the correct arguments for the specified tool.
"""

    async def create_plan(self, query: str, tool_registry: ToolRegistry) -> Plan:
        """
        Create a plan to fulfill the given query or objective.
        
        Args:
            query: The query or objective to fulfill
            tool_registry: Registry of available tools
            
        Returns:
            An execution plan
        """
        # Create the system prompt with tool descriptions
        system_prompt = self._create_system_prompt(tool_registry)
        
        # Create messages for the LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Create a plan to fulfill this objective: {query}"}
        ]
        
        # Generate plan using the LLM
        response = await self.model.agenerate(messages=messages)
        plan_text = response.generations[0].text
        
        # Parse the plan
        try:
            import json
            import re

            # Try to extract JSON if it's embedded in markdown or text
            json_match = re.search(r'```json\s*(.+?)\s*```', plan_text, re.DOTALL)
            if json_match:
                plan_text = json_match.group(1)
            else:
                # Try to find JSON object without markdown
                json_match = re.search(r'({.+})', plan_text, re.DOTALL)
                if json_match:
                    plan_text = json_match.group(1)
            
            plan_data = json.loads(plan_text)
            
            # Create the Plan object
            plan = Plan(description=plan_data.get("description", query))
            
            # Add steps
            for i, step_data in enumerate(plan_data.get("steps", [])):
                step_id = str(i + 1)  # Use sequential numbers as IDs
                step = PlanStep(
                    id=step_id,
                    tool_name=step_data["tool_name"],
                    tool_args=step_data["tool_args"],
                    description=step_data["description"],
                    depends_on=step_data.get("depends_on", [])
                )
                plan.add_step(step)
            
            return plan
            
        except Exception as e:
            # If parsing fails, create a minimal plan
            plan = Plan(description=f"Failed to parse plan: {str(e)}")
            return plan


class BeamSearchPlanner(BasePlanner):
    """
    Planner that uses beam search to explore multiple planning paths.
    
    This planner generates and evaluates multiple candidate plans using beam
    search, a breadth-first search algorithm that maintains a set of the most
    promising candidates at each step.
    """
    
    def __init__(
        self, 
        model: BaseChatModel, 
        beam_width: int = 3,
        max_depth: int = 5,
        evaluator: Optional[Callable[[Plan], float]] = None
    ):
        """
        Initialize the beam search planner.
        
        Args:
            model: The language model to use for planning
            beam_width: Number of candidates to maintain at each step
            max_depth: Maximum depth of the search tree
            evaluator: Function to evaluate the quality of a plan
        """
        self.model = model
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.evaluator = evaluator or self._default_evaluator
        
        # Create a base LLM planner for generating initial plans
        self.base_planner = LLMPlanner(model)
    
    def _default_evaluator(self, plan: Plan) -> float:
        """
        Default function to evaluate the quality of a plan.
        
        Args:
            plan: The plan to evaluate
            
        Returns:
            A score from 0.0 to 1.0
        """
        # Simple heuristic:
        # - Penalize plans with too many or too few steps
        # - Reward plans with dependencies between steps
        
        # Count dependencies
        total_deps = sum(len(step.depends_on) for step in plan.steps)
        
        # Calculate score based on number of steps and dependencies
        step_count = len(plan.steps)
        if step_count == 0:
            return 0.0
            
        step_score = min(step_count / 3.0, 1.0)  # Prefer plans with at least 3 steps
        dep_score = min(total_deps / step_count, 1.0)  # Prefer plans with dependencies
        
        return (step_score * 0.7) + (dep_score * 0.3)
    
    async def create_plan(self, query: str, tool_registry: ToolRegistry) -> Plan:
        """
        Create a plan using beam search.
        
        Args:
            query: The query or objective to fulfill
            tool_registry: Registry of available tools
            
        Returns:
            The best execution plan according to the evaluator
        """
        # Generate initial plan as a starting point
        initial_plan = await self.base_planner.create_plan(query, tool_registry)
        
        # Initialize beam with the initial plan
        current_beam = [initial_plan]
        best_plan = initial_plan
        best_score = self.evaluator(initial_plan)
        
        # Perform beam search iterations
        for depth in range(self.max_depth):
            candidates = []
            
            # Expand each plan in the current beam
            for plan in current_beam:
                # Generate variations of the plan
                variations = await self._generate_plan_variations(plan, query, tool_registry)
                candidates.extend(variations)
            
            # Score all candidate plans
            scored_candidates = [(plan, self.evaluator(plan)) for plan in candidates]
            
            # Sort by score (descending)
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Select top-k candidates for the next iteration
            current_beam = [plan for plan, _ in scored_candidates[:self.beam_width]]
            
            # Update best plan if we found a better one
            if scored_candidates and scored_candidates[0][1] > best_score:
                best_plan = scored_candidates[0][0]
                best_score = scored_candidates[0][1]
        
        return best_plan
    
    async def _generate_plan_variations(self, plan: Plan, query: str, tool_registry: ToolRegistry) -> List[Plan]:
        """
        Generate variations of a plan to explore alternative approaches.
        
        Args:
            plan: The base plan to create variations from
            query: The original query
            tool_registry: Available tools
            
        Returns:
            A list of plan variations
        """
        # This is a simplified implementation that creates a few variations:
        # 1. The original plan
        # 2. A plan with one more step added
        # 3. A plan with one step removed (if possible)
        # 4. A plan with reordered steps (if possible)
        
        variations = [plan]  # Start with the original plan
        
        # Variation 1: Add a step
        if len(plan.steps) < self.max_depth * 2:
            # Ask the LLM to suggest an additional step
            system_prompt = """You are an AI planner assistant. Given a plan and query, 
            suggest ONE additional step that would improve the plan. Return only a JSON object 
            with tool_name, tool_args, description, and depends_on fields."""
            
            plan_summary = f"Current plan: {plan.description} with {len(plan.steps)} steps"
            step_descriptions = "\n".join([f"Step {s.id}: {s.description}" for s in plan.steps])
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {query}\n\n{plan_summary}\n{step_descriptions}\n\nSuggest ONE additional step:"}
            ]
            
            try:
                response = await self.model.agenerate(messages=messages)
                new_step_text = response.generations[0].text
                
                import json
                import re

                # Extract JSON
                json_match = re.search(r'{.+}', new_step_text, re.DOTALL)
                if json_match:
                    new_step_data = json.loads(json_match.group(0))
                    
                    # Create new plan with the additional step
                    new_plan = Plan(description=plan.description)
                    
                    # Copy existing steps
                    for step in plan.steps:
                        new_plan.add_step(step.copy())
                    
                    # Add new step
                    step_id = str(len(new_plan.steps) + 1)
                    new_step = PlanStep(
                        id=step_id,
                        tool_name=new_step_data["tool_name"],
                        tool_args=new_step_data["tool_args"],
                        description=new_step_data["description"],
                        depends_on=new_step_data.get("depends_on", [])
                    )
                    new_plan.add_step(new_step)
                    variations.append(new_plan)
            except Exception:
                pass  # Skip this variation if it fails
        
        # Variation 2: Remove a step (if there's more than one)
        if len(plan.steps) > 1:
            # Try removing different steps to see if the plan still works
            for i in range(len(plan.steps)):
                new_plan = Plan(description=plan.description)
                for j, step in enumerate(plan.steps):
                    if j != i:  # Skip the step we're removing
                        new_step = step.copy()
                        # Update dependencies to account for the removed step
                        new_step.depends_on = [
                            dep for dep in step.depends_on if dep != plan.steps[i].id
                        ]
                        new_plan.add_step(new_step)
                variations.append(new_plan)
        
        # Variation 3: Reorder steps if possible (swap independent steps)
        if len(plan.steps) >= 2:
            # Find steps that can be swapped (no dependencies between them)
            for i in range(len(plan.steps) - 1):
                for j in range(i + 1, len(plan.steps)):
                    step_i = plan.steps[i]
                    step_j = plan.steps[j]
                    
                    # Check if steps can be swapped
                    if (step_j.id not in step_i.depends_on and 
                        step_i.id not in step_j.depends_on):
                        
                        new_plan = Plan(description=plan.description)
                        for k, step in enumerate(plan.steps):
                            if k == i:
                                new_plan.add_step(plan.steps[j].copy())
                            elif k == j:
                                new_plan.add_step(plan.steps[i].copy())
                            else:
                                new_plan.add_step(step.copy())
                        variations.append(new_plan)
        
        return variations