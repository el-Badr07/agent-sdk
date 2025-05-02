from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from hotlm_core.schema import BaseMessage


@dataclass
class AgentFinish:
    """Class to represent the end of an agent run with a final output."""
    output: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolExecution:
    """Class to track a single tool execution step."""
    tool_name: str
    tool_args: Dict[str, Any]
    result: Optional[str] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class LoopState:
    """State object for AgentLoop iterations with improved tracking capabilities."""
    def __init__(self):
        # Core state
        self.messages: List[BaseMessage] = []
        self.done: bool = False
        self.final_output: Optional[str] = None
        
        # Enhanced tracking
        self.intermediate_steps: List[Tuple[str, Dict[str, Any]]] = []
        self.tool_executions: List[ToolExecution] = []
        self.iteration_count: int = 0
        self.metadata: Dict[str, Any] = {}
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
    def add_tool_execution(self, tool_execution: ToolExecution) -> None:
        """Add a tool execution step to the state."""
        self.tool_executions.append(tool_execution)
        # Also maintain legacy format for compatibility
        self.intermediate_steps.append((tool_execution.tool_name, tool_execution.tool_args))
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get a structured history of the execution for analysis."""
        history = []
        
        for i, msg in enumerate(self.messages):
            entry = {
                "step": i,
                "type": msg.__class__.__name__,
                "content": msg.content,
            }
            if hasattr(msg, "metadata") and msg.metadata:
                entry["metadata"] = msg.metadata
            
            history.append(entry)
        
        return history
    
    @property
    def execution_time(self) -> Optional[float]:
        """Get the total execution time if available."""
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return None
    
    def get_tool_usage_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary statistics about tool usage."""
        summary = {}
        
        for execution in self.tool_executions:
            tool_name = execution.tool_name
            if tool_name not in summary:
                summary[tool_name] = {
                    "count": 0,
                    "total_time": 0.0,
                    "success_count": 0,
                    "error_count": 0,
                }
            
            summary[tool_name]["count"] += 1
            
            if execution.execution_time is not None:
                summary[tool_name]["total_time"] += execution.execution_time
                
            if execution.error:
                summary[tool_name]["error_count"] += 1
            else:
                summary[tool_name]["success_count"] += 1
        
        # Calculate average execution times
        for tool_name, stats in summary.items():
            if stats["count"] > 0:
                stats["avg_time"] = stats["total_time"] / stats["count"]
            else:
                stats["avg_time"] = 0.0
        
        return summary