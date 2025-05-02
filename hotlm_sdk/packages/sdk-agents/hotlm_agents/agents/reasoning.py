"""Implementation of an agent with structured reasoning abilities."""

import json
import re
from typing import Any, Dict, List, Optional, Union

from hotlm_agents.agents.base import AgentOutput, BaseAgent
from hotlm_core.callbacks import CallbackManager
from hotlm_core.config import RunnableConfig
from hotlm_core.memory import BaseMemory
from hotlm_core.models import BaseChatModel
from hotlm_core.prompts import BaseChatPromptTemplate, ChatPromptTemplate
from hotlm_core.schema import AIMessage, HumanMessage, SystemMessage
from hotlm_core.tools import BaseTool


class ReasoningAgent(BaseAgent):
    """
    An agent that uses chain-of-thought reasoning to solve problems step by step.
    
    This agent explicitly breaks down problems, considers different approaches,
    and makes deliberate decisions about tool usage and final answers.
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        tools: Optional[List[BaseTool]] = None,
        memory: Optional[BaseMemory] = None,
        system_message: Optional[str] = None,
        custom_prompt_template: Optional[BaseChatPromptTemplate] = None,
        max_iterations: int = 10,
        reasoning_strategy: str = "cot",  # "cot" (chain of thought) or "react" (reasoning and acting)
        callbacks: Optional[List[Any]] = None,
        return_intermediate_steps: bool = False,
        verbose: bool = False,
        agent_id: Optional[str] = None,
    ):
        """
        Initialize a ReasoningAgent with explicit reasoning capabilities.
        
        Args:
            llm: Language model to use for generating agent responses
            tools: List of tools available to the agent
            memory: Optional memory to store conversation history
            system_message: Optional system message override (if not provided, a default will be used)
            custom_prompt_template: Optional custom prompt template (if not provided, will use get_prompt_template)
            max_iterations: Maximum number of iterations before forcing termination
            reasoning_strategy: Strategy for reasoning ("cot" or "react")
            callbacks: Optional list of callback handlers
            return_intermediate_steps: Whether to include intermediate steps in result
            verbose: Whether to log detailed information during execution
            agent_id: Optional unique identifier for this agent instance
        """
        self.reasoning_strategy = reasoning_strategy
        self.verbose = verbose
        
        # Use provided prompt template or generate one
        prompt_template = custom_prompt_template or self.get_prompt_template()
        
        # If no system message provided, use default based on reasoning strategy
        if system_message is None:
            if reasoning_strategy == "react":
                system_message = self._get_react_system_message(tools or [])
            else:  # Default to chain-of-thought
                system_message = self._get_cot_system_message(tools or [])
        
        super().__init__(
            llm=llm,
            prompt_template=prompt_template,
            tools=tools,
            memory=memory,
            system_message=system_message,
            max_iterations=max_iterations,
            early_stopping_method="generate",  # Reasoning agents should generate final answers
            callbacks=callbacks,
            return_intermediate_steps=return_intermediate_steps,
            agent_id=agent_id,
        )
        
        # Update metadata
        self.metadata.update({
            "reasoning_strategy": reasoning_strategy,
            "verbose": verbose,
        })
    
    def get_prompt_template(self) -> BaseChatPromptTemplate:
        """Return a prompt template designed for reasoning agents."""
        return ChatPromptTemplate.from_messages([
            SystemMessage(content="{system_message}"),
            {"role": "placeholder", "content": "{messages}"},
        ])
    
    def _get_cot_system_message(self, tools: List[BaseTool]) -> str:
        """Generate a system message for Chain-of-Thought reasoning."""
        tool_descriptions = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
        
        system_message = """You are a thoughtful assistant that helps solve problems by thinking step-by-step.

For each request, follow this process:
1. Think - Break down the problem. Consider what you know and what you need to find out.
2. Plan - Decide what steps you'll take to solve the problem.
3. Execute - Work through your plan, using tools when necessary.
4. Verify - Check if your solution addresses the original problem.

When you need to use a tool, format your response like this:
```json
{"tool": "tool_name", "args": {"arg1": "value1", "arg2": "value2"}}
```

Available tools:
{tool_descriptions}

Always explain your reasoning clearly. If you're uncertain, explain what you're unsure about and what additional information might help.
"""
        
        return system_message.format(tool_descriptions=tool_descriptions if tools else "No tools available.")
    
    def _get_react_system_message(self, tools: List[BaseTool]) -> str:
        """Generate a system message for ReAct (reasoning and acting) approach."""
        tool_descriptions = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
        
        system_message = """You are a ReAct (Reasoning and Acting) agent that solves tasks using tools and reasoning.

For each task, follow this structured format:
1. Thought: Think about what you need to do and why
2. Action: Decide which tool to use and with what parameters
3. Observation: Review what you learned from the tool
4. ... (repeat the above steps as needed)
5. Thought: I now have the information to answer the question
6. Answer: Your final response to the user's request

When specifying an Action, use this JSON format:
```json
{"tool": "tool_name", "args": {"arg1": "value1", "arg2": "value2"}}
```

Available tools:
{tool_descriptions}

Always start with a Thought and end with an Answer. Each Thought should clearly explain your reasoning process.
"""
        
        return system_message.format(tool_descriptions=tool_descriptions if tools else "No tools available.")


class PlanAndExecuteAgent(ReasoningAgent):
    """
    An agent that explicitly separates planning and execution phases.
    
    This agent first creates a complete plan, then executes each step of the plan
    in sequence, potentially adjusting the plan based on new information.
    """
    
    DEFAULT_SYSTEM_TEMPLATE = """You are an AI assistant that solves problems by first planning and then executing.

Follow this structured approach:

1. UNDERSTAND THE PROBLEM
- Understand what is being asked
- Identify key information and constraints

2. CREATE A DETAILED PLAN
- Break the problem into concrete steps
- Specify which tools you'll use for each step
- Be explicit and detailed

3. EXECUTE THE PLAN STEP BY STEP
- Execute exactly one step at a time
- When you need to use a tool, format your response as:
  ```json
  {"tool": "tool_name", "args": {"arg1": "value1", "arg2": "value2"}}
  ```
- After receiving tool outputs, interpret them before moving to the next step
- Only move to the next step once the current step is completed

4. ADJUST THE PLAN IF NEEDED
- If you encounter unexpected results, adapt your plan
- Explain any changes to the original plan

5. SUMMARIZE RESULTS
- Provide your final answer clearly
- Explain how you solved the problem

Available tools: {tool_descriptions}

Begin by understanding the problem and creating a plan. Then execute one step at a time.
"""

    def __init__(self, *args, **kwargs):
        """Initialize a PlanAndExecuteAgent with the specialized system template."""
        # Pass our specialized default template but allow overrides
        super().__init__(*args, **kwargs)


class BranchingReasoningAgent(ReasoningAgent):
    """
    An agent that can explore multiple solution paths simultaneously.
    
    This agent can branch its reasoning to explore different approaches to a problem,
    evaluate each branch, and select the most promising solution.
    """
    
    DEFAULT_SYSTEM_TEMPLATE = """You are an AI assistant that solves complex problems by exploring multiple approaches.

Follow this structured approach:

1. UNDERSTAND THE PROBLEM
- Identify what is being asked 
- List key information and constraints

2. GENERATE MULTIPLE APPROACHES
- Consider at least 2-3 different methods to solve the problem
- Briefly outline each approach

3. EVALUATE AND SELECT
- Consider the tradeoffs of each approach
- Select the most promising approach(es) to pursue

4. EXECUTE THE CHOSEN APPROACH(ES)
- Work through the selected approach step by step
- When you need to use a tool, format your response as:
  ```json
  {"tool": "tool_name", "args": {"arg1": "value1", "arg2": "value2"}}
  ```
- If the chosen approach fails, try an alternative approach

5. VERIFY AND PRESENT RESULTS
- Validate your solution
- Present your final answer clearly
- Explain which approach worked best and why

Available tools: {tool_descriptions}

When faced with uncertainty, it's better to explore multiple approaches rather than committing prematurely to one.
"""

    def __init__(self, *args, **kwargs):
        """Initialize a BranchingReasoningAgent with the specialized system template."""
        # Default to a higher number of max iterations since we're exploring multiple paths
        if "max_iterations" not in kwargs:
            kwargs["max_iterations"] = 20
        super().__init__(*args, **kwargs)