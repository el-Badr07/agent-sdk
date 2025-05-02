"""
Planning data structures for the planner/executor pattern.

This module defines the core data structures used for representing and
managing execution plans in the two-step "plan then act" pattern.
"""

import enum
import uuid
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field


class StepStatus(enum.Enum):
    """Represents the status of a plan step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"

class PlanStep(BaseModel):
    """
    A single step in an execution plan.
    
    A step represents a single tool invocation with its inputs and dependencies.
    """
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    tool_name: str
    tool_args: Dict[str, Any]
    description: str
    depends_on: List[str] = Field(default_factory=list)
    status: StepStatus = Field(default=StepStatus.PENDING)
    result: Optional[str] = None
    error: Optional[str] = None
    
    def is_complete(self) -> bool:
        """Check if the step has been completed (succeeded or failed)."""
        return self.status in (StepStatus.SUCCEEDED, StepStatus.FAILED, StepStatus.SKIPPED)
    
    def is_successful(self) -> bool:
        """Check if the step has been completed successfully."""
        return self.status == StepStatus.SUCCEEDED
    
    def is_failed(self) -> bool:
        """Check if the step has failed."""
        return self.status == StepStatus.FAILED
    
    def is_pending(self) -> bool:
        """Check if the step is pending execution."""
        return self.status == StepStatus.PENDING
    
    def is_runnable(self, completed_step_ids: Set[str]) -> bool:
        """
        Check if the step can be run.
        
        Args:
            completed_step_ids: Set of IDs of completed steps
            
        Returns:
            True if the step can be run, False otherwise
        """
        if not self.is_pending():
            return False
        
        # Check if all dependencies are satisfied (completed)
        return all(dep_id in completed_step_ids for dep_id in self.depends_on)

class Plan(BaseModel):
    """
    A complete execution plan consisting of multiple steps.
    
    A plan represents a sequence of steps to be executed, with dependencies
    between steps.
    """
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    steps: List[PlanStep] = Field(default_factory=list)
    
    def add_step(self, step: PlanStep) -> None:
        """
        Add a step to the plan.
        
        Args:
            step: The step to add
        """
        self.steps.append(step)
    
    def get_step(self, step_id: str) -> Optional[PlanStep]:
        """
        Get a step by its ID.
        
        Args:
            step_id: The ID of the step to get
            
        Returns:
            The step if found, None otherwise
        """
        for step in self.steps:
            if step.id == step_id:
                return step
        return None
    
    def is_complete(self) -> bool:
        """
        Check if the plan is complete (all steps have been executed).
        
        Returns:
            True if the plan is complete, False otherwise
        """
        return all(step.is_complete() for step in self.steps)
    
    def has_failed_steps(self) -> bool:
        """
        Check if the plan has any failed steps.
        
        Returns:
            True if the plan has any failed steps, False otherwise
        """
        return any(step.is_failed() for step in self.steps)
    
    def get_runnable_steps(self) -> List[PlanStep]:
        """
        Get steps that are ready to run.
        
        Returns:
            A list of steps that can be run
        """
        completed_step_ids = {step.id for step in self.steps if step.is_complete()}
        return [step for step in self.steps if step.is_runnable(completed_step_ids)]


class PlanResult(BaseModel):
    """Result of executing a plan."""
    plan: Plan
    success: bool
    error: Optional[str] = None
    summary: Optional[str] = None