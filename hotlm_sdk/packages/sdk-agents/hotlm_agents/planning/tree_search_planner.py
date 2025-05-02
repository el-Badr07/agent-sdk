"""
Tree search planner implementation.

This module provides a planning algorithm that uses tree search to
explore and evaluate multiple potential plans.
"""

import asyncio
import heapq
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from hotlm_agents.planning.plan import Plan, PlanStep
from hotlm_agents.planning.planner import BasePlanner, LLMPlanner
from hotlm_agents.tooling.registry import ToolRegistry
from hotlm_core.models import BaseChatModel


class TreeNode:
    """
    Node in the planning search tree.
    
    Each node represents a partial plan with the steps collected so far.
    """
    
    def __init__(self, 
                 partial_plan: Plan,
                 depth: int = 0,
                 parent: Optional['TreeNode'] = None,
                 path_cost: float = 0.0):
        """
        Initialize a tree node.
        
        Args:
            partial_plan: The partial plan at this node
            depth: Depth of this node in the search tree
            parent: Parent node, or None for the root
            path_cost: Cumulative cost from root to this node
        """
        self.plan = partial_plan
        self.depth = depth
        self.parent = parent
        self.path_cost = path_cost
        self.heuristic = 0.0  # Estimated cost to goal
        
    def get_f_score(self) -> float:
        """
        Get the f-score (path_cost + heuristic) for this node.
        
        Returns:
            The f-score
        """
        return self.path_cost + self.heuristic
    
    def __lt__(self, other: 'TreeNode') -> bool:
        """
        Compare nodes by f-score for priority queue ordering.
        
        Args:
            other: Node to compare with
            
        Returns:
            True if this node has a lower f-score
        """
        return self.get_f_score() < other.get_f_score()


class TreeSearchPlanner(BasePlanner):
    """
    Planner that uses tree search to find an optimal plan.
    
    This planner uses a tree search algorithm (specifically A*) to explore
    the space of possible plans and find one that optimizes a specified objective.
    """
    
    def __init__(
        self,
        model: BaseChatModel,
        max_depth: int = 5,
        max_branches: int = 3,
        max_nodes: int = 100,
        heuristic_weight: float = 0.5
    ):
        """
        Initialize the tree search planner.
        
        Args:
            model: The language model to use for plan generation
            max_depth: Maximum depth of the search tree
            max_branches: Maximum branching factor at each node
            max_nodes: Maximum total number of nodes to expand
            heuristic_weight: Weight given to the heuristic in f-score calculation
        """
        self.model = model
        self.max_depth = max_depth
        self.max_branches = max_branches
        self.max_nodes = max_nodes
        self.heuristic_weight = heuristic_weight
        
        # Create a base LLM planner for generating initial plans
        self.base_planner = LLMPlanner(model)
    
    async def create_plan(self, query: str, tool_registry: ToolRegistry) -> Plan:
        """
        Create a plan using tree search.
        
        This implements an A*-like search algorithm to find an optimal plan.
        
        Args:
            query: The query or objective to fulfill
            tool_registry: Registry of available tools
            
        Returns:
            The best plan found by the search algorithm
        """
        # Generate an initial plan as starting point
        initial_plan = await self.base_planner.create_plan(query, tool_registry)
        
        # Create the root node
        root = TreeNode(initial_plan)
        root.heuristic = await self._calculate_heuristic(root.plan, query)
        
        # Initialize open and closed sets
        open_set = [root]  # Priority queue (heapq)
        open_set_plans = {self._plan_signature(root.plan)}
        closed_set = set()  # Set of visited plan signatures
        
        # Stats
        nodes_expanded = 0
        
        # A* search
        while open_set and nodes_expanded < self.max_nodes:
            # Get the node with lowest f-score
            current = heapq.heappop(open_set)
            open_set_plans.remove(self._plan_signature(current.plan))
            
            # Check if we've reached a goal state
            if self._is_goal_state(current.plan, query):
                return current.plan
            
            # Skip if we've seen this plan before
            plan_sig = self._plan_signature(current.plan)
            if plan_sig in closed_set:
                continue
                
            closed_set.add(plan_sig)
            nodes_expanded += 1
            
            # Don't expand beyond max depth
            if current.depth >= self.max_depth:
                continue
                
            # Generate successors
            successors = await self._generate_successors(current, query, tool_registry)
            
            # Add successors to open set
            for successor in successors[:self.max_branches]:
                successor_sig = self._plan_signature(successor.plan)
                if successor_sig not in closed_set and successor_sig not in open_set_plans:
                    heapq.heappush(open_set, successor)
                    open_set_plans.add(successor_sig)
        
        # Return the best plan found so far
        if open_set:
            return min(open_set, key=lambda x: x.get_f_score()).plan
        else:
            return initial_plan  # Fallback to initial plan if search fails
    
    async def _generate_successors(
        self, 
        node: TreeNode,
        query: str,
        tool_registry: ToolRegistry
    ) -> List[TreeNode]:
        """
        Generate successor nodes by expanding the current plan.
        
        Args:
            node: The current node to expand
            query: The original query
            tool_registry: Available tools
            
        Returns:
            List of successor nodes
        """
        # Generate variations to the current plan to create successors
        variations = []
        
        # Variation 1: Add a new step
        add_step_plan = await self._add_step_to_plan(node.plan, query, tool_registry)
        if add_step_plan:
            successor = TreeNode(
                partial_plan=add_step_plan,
                depth=node.depth + 1,
                parent=node,
                path_cost=node.path_cost + 1  # Simple cost model
            )
            successor.heuristic = await self._calculate_heuristic(successor.plan, query)
            variations.append(successor)
        
        # Variation 2: Remove a step (if there are steps to remove)
        if len(node.plan.steps) > 1:
            for i in range(len(node.plan.steps)):
                remove_step_plan = self._remove_step_from_plan(node.plan, i)
                successor = TreeNode(
                    partial_plan=remove_step_plan, 
                    depth=node.depth + 1,
                    parent=node,
                    path_cost=node.path_cost + 0.5  # Removing is less "costly"
                )
                successor.heuristic = await self._calculate_heuristic(successor.plan, query)
                variations.append(successor)
        
        # Variation 3: Reorder steps (swap consecutive steps if possible)
        if len(node.plan.steps) >= 2:
            for i in range(len(node.plan.steps) - 1):
                # Only swap consecutive steps that don't have dependencies between them
                if (node.plan.steps[i+1].id not in node.plan.steps[i].depends_on and
                    node.plan.steps[i].id not in node.plan.steps[i+1].depends_on):
                    swap_plan = self._swap_steps_in_plan(node.plan, i, i+1)
                    successor = TreeNode(
                        partial_plan=swap_plan,
                        depth=node.depth + 1,
                        parent=node,
                        path_cost=node.path_cost + 0.3  # Swapping is less costly than adding
                    )
                    successor.heuristic = await self._calculate_heuristic(successor.plan, query)
                    variations.append(successor)
        
        # Sort variations by combined score (f-score) and return
        variations.sort(key=lambda x: x.get_f_score())
        return variations
    
    async def _add_step_to_plan(
        self, 
        plan: Plan,
        query: str,
        tool_registry: ToolRegistry
    ) -> Optional[Plan]:
        """
        Add a new step to the plan using the LLM.
        
        Args:
            plan: Current plan
            query: Original query
            tool_registry: Available tools
            
        Returns:
            New plan with the added step, or None if failed
        """
        system_prompt = """You are an AI planner. Given a plan and query, suggest ONE additional 
        step that would improve the plan. Return ONLY a JSON object with tool_name, tool_args, 
        description, and depends_on fields."""
        
        # Generate text representation of current plan
        plan_summary = f"Plan: {plan.description}"
        steps_text = "\n".join([
            f"Step {s.id}: {s.description} (Tool: {s.tool_name})"
            for s in plan.steps
        ])
        
        # List available tools
        tools_text = "\n".join([
            f"- {tool.name}: {tool.description}"
            for tool in tool_registry.get_all_tools()
        ])
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""Query: {query}
            
Current Plan:
{plan_summary}
{steps_text}

Available Tools:
{tools_text}

Suggest ONE additional step to improve this plan. Return ONLY a JSON object.
"""}
        ]
        
        try:
            # Get suggestion from LLM
            response = await self.model.agenerate(messages=messages)
            suggestion_text = response.generations[0].text
            
            # Parse the suggestion
            import json
            import re

            # Try to extract JSON from the response
            json_match = re.search(r'{.*}', suggestion_text, re.DOTALL)
            if not json_match:
                return None
            
            suggestion = json.loads(json_match.group(0))
            
            # Create a new plan with the added step
            new_plan = Plan(description=plan.description)
            
            # Copy existing steps
            for step in plan.steps:
                new_plan.add_step(step.copy())
            
            # Add new step
            step_id = str(len(new_plan.steps) + 1)
            new_step = PlanStep(
                id=step_id,
                tool_name=suggestion["tool_name"],
                tool_args=suggestion["tool_args"],
                description=suggestion["description"],
                depends_on=suggestion.get("depends_on", [])
            )
            new_plan.add_step(new_step)
            
            return new_plan
            
        except Exception:
            # If anything goes wrong, return None
            return None
    
    def _remove_step_from_plan(self, plan: Plan, step_index: int) -> Plan:
        """
        Remove a step from the plan.
        
        Args:
            plan: Current plan
            step_index: Index of the step to remove
            
        Returns:
            New plan with the step removed
        """
        new_plan = Plan(description=plan.description)
        step_to_remove = plan.steps[step_index]
        
        # Add all steps except the one to remove
        for i, step in enumerate(plan.steps):
            if i != step_index:
                new_step = step.copy()
                # Update any dependencies that reference the removed step
                new_step.depends_on = [
                    dep for dep in step.depends_on 
                    if dep != step_to_remove.id
                ]
                new_plan.add_step(new_step)
        
        return new_plan
    
    def _swap_steps_in_plan(self, plan: Plan, index1: int, index2: int) -> Plan:
        """
        Swap two steps in the plan.
        
        Args:
            plan: Current plan
            index1: Index of first step
            index2: Index of second step
            
        Returns:
            New plan with the steps swapped
        """
        new_plan = Plan(description=plan.description)
        
        # Create a mapping from old step IDs to new positions
        id_mapping = {}
        
        # Copy steps with swapped positions
        for i, step in enumerate(plan.steps):
            if i == index1:
                new_plan.add_step(plan.steps[index2].copy())
                id_mapping[plan.steps[index2].id] = len(new_plan.steps)
            elif i == index2:
                new_plan.add_step(plan.steps[index1].copy())
                id_mapping[plan.steps[index1].id] = len(new_plan.steps)
            else:
                new_plan.add_step(step.copy())
                id_mapping[step.id] = len(new_plan.steps)
        
        return new_plan
    
    async def _calculate_heuristic(self, plan: Plan, query: str) -> float:
        """
        Calculate heuristic value for a plan (estimated cost to goal).
        
        Args:
            plan: The plan to evaluate
            query: The original query
            
        Returns:
            Heuristic value (lower is better)
        """
        # Simple heuristic based on plan complexity and alignment with query
        step_count = len(plan.steps)
        
        # Too few steps might not be enough to complete the task
        if step_count == 0:
            return float('inf')
        elif step_count <= 2:
            step_count_score = 0.5
        elif step_count <= 5:
            step_count_score = 0.2
        else:
            step_count_score = 0.3 + (step_count - 5) * 0.1  # Penalty for too many steps
        
        # Evaluate tool variety (using more diverse tools might be better)
        unique_tools = len(set(step.tool_name for step in plan.steps))
        tool_variety_score = max(0.0, 0.5 - (unique_tools / max(1, step_count)) * 0.5)
        
        # Evaluate dependencies (more dependencies might indicate a more structured plan)
        total_deps = sum(len(step.depends_on) for step in plan.steps)
        if step_count <= 1:
            dependency_score = 0.3
        else:
            ideal_deps = step_count - 1  # Ideally, all steps but one have dependencies
            dependency_score = 0.3 * (1 - min(1.0, total_deps / ideal_deps))
        
        # Combine scores (lower is better)
        heuristic = step_count_score + tool_variety_score + dependency_score
        
        # Normalize to a 0-1 range
        return min(1.0, heuristic)
    
    def _is_goal_state(self, plan: Plan, query: str) -> bool:
        """
        Check if a plan represents a goal state.
        
        Args:
            plan: The plan to check
            query: The original query
            
        Returns:
            True if the plan is a goal state
        """
        # A plan is a goal state if it has a reasonable number of steps,
        # uses a variety of tools, and has appropriate dependencies
        if len(plan.steps) < 2:
            return False
            
        # Check for tool variety
        unique_tools = len(set(step.tool_name for step in plan.steps))
        if unique_tools < min(2, len(plan.steps)):
            return False
            
        # Check for dependencies
        total_deps = sum(len(step.depends_on) for step in plan.steps)
        if len(plan.steps) > 2 and total_deps == 0:
            # No dependencies in a multi-step plan is suspicious
            return False
            
        # Consider it a goal state
        return True
    
    def _plan_signature(self, plan: Plan) -> str:
        """
        Generate a unique signature for a plan to detect duplicates.
        
        Args:
            plan: The plan to generate a signature for
            
        Returns:
            A unique signature string
        """
        # Create a signature based on tool names, arguments, and dependencies
        parts = []
        for step in plan.steps:
            # Convert args to a stable string representation
            args_str = str(sorted(step.tool_args.items()))
            depends_str = ",".join(sorted(step.depends_on))
            parts.append(f"{step.tool_name}:{args_str}:{depends_str}")
        
        return "|".join(sorted(parts))