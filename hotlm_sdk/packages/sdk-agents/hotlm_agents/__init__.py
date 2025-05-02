"""
HotLM Agents package - A framework for creating, configuring, and executing AI agents.

This package provides abstractions and tools for building AI agents
with various capabilities, including reasoning and tool usage.
"""

__version__ = "0.1.0"

from hotlm_agents.agents.base import BaseAgent, SimpleAgent
from hotlm_agents.agents.reasoning import (
    BranchingReasoningAgent,
    PlanAndExecuteAgent,
    ReasoningAgent,
)

# Import core classes
from hotlm_agents.execution.loop import AgentLoop
from hotlm_agents.execution.state import AgentFinish, LoopState, ToolExecution

# Top-level imports for easy access
from hotlm_agents.tooling.decorator import ParamStyle, tool

__all__ = [
    "tool",
    "ParamStyle",
    "BaseAgent",
    "SimpleAgent",
    "ReasoningAgent",
    "PlanAndExecuteAgent",
    "BranchingReasoningAgent",
    "AgentLoop",
    "LoopState",
    "AgentFinish",
    "ToolExecution",
]
