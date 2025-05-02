"""
Planning strategy factory.

This module provides a factory for creating different planning strategy implementations.
"""

from typing import Any, Dict, Optional, Protocol, Type, runtime_checkable

from hotlm_agents.planning.plan import Plan
from hotlm_agents.planning.planner import BasePlanner, BeamSearchPlanner, LLMPlanner
from hotlm_agents.planning.tree_search_planner import TreeSearchPlanner
from hotlm_agents.tooling.registry import ToolRegistry
from hotlm_core.models import BaseChatModel


@runtime_checkable
class PlanningStrategy(Protocol):
    """Interface for planning strategies."""
    
    async def create_plan(self, query: str, tool_registry: ToolRegistry) -> Plan:
        """
        Create a plan to fulfill the given query or objective.
        
        Args:
            query: The query or objective to fulfill
            tool_registry: Registry of available tools
            
        Returns:
            An execution plan
        """
        ...


class PlanningStrategyFactory:
    """
    Factory for creating different planning strategy implementations.
    
    This factory enables users to easily swap between different planning algorithms
    such as basic LLM planning, beam search, tree search, or rehearsal planning.
    """
    
    _registry: Dict[str, Type[BasePlanner]] = {}
    
    @classmethod
    def register(cls, name: str, strategy_class: Type[BasePlanner]) -> None:
        """
        Register a planning strategy implementation.
        
        Args:
            name: Name of the strategy
            strategy_class: The planner class to register
        """
        cls._registry[name] = strategy_class
    
    @classmethod
    def create(cls, 
               strategy_type: str, 
               model: BaseChatModel, 
               **kwargs) -> PlanningStrategy:
        """
        Create a planning strategy of the specified type.
        
        Args:
            strategy_type: Type of planning strategy to create
            model: Language model to use for planning
            **kwargs: Additional arguments for the specific planner
            
        Returns:
            A planning strategy implementation
            
        Raises:
            ValueError: If strategy_type is not registered
        """
        # Register built-in strategies if not already registered
        if not cls._registry:
            cls._register_defaults()
        
        if strategy_type not in cls._registry:
            raise ValueError(
                f"Unknown planning strategy: {strategy_type}. "
                f"Available strategies: {', '.join(cls._registry.keys())}"
            )
        
        strategy_class = cls._registry[strategy_type]
        return strategy_class(model=model, **kwargs)
    
    @classmethod
    def _register_defaults(cls) -> None:
        """Register the default planning strategies."""
        cls.register("basic", LLMPlanner)
        cls.register("beam_search", BeamSearchPlanner)
        cls.register("tree_search", TreeSearchPlanner)
        
        # Note: Rehearsal planning will be implemented and registered
        # in future releases.

    @classmethod
    def list_available_strategies(cls) -> Dict[str, str]:
        """
        Get a list of all available planning strategies.
        
        Returns:
            Dictionary mapping strategy names to short descriptions
        """
        # Register built-in strategies if not already registered
        if not cls._registry:
            cls._register_defaults()
            
        descriptions = {
            "basic": "Basic LLM-based planning with a single pass",
            "beam_search": "Beam search exploring multiple plan variations in parallel",
            "tree_search": "A* tree search algorithm for optimal plan discovery"
        }
        
        return {name: descriptions.get(name, "No description available") 
                for name in cls._registry.keys()}


# Register default planning strategies
PlanningStrategyFactory._register_defaults()