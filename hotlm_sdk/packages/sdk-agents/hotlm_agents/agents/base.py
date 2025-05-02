"""BaseAgent class holding configuration and providing a simple run interface."""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, TypeVar, Union

from hotlm_agents.execution.loop import AgentLoop
from hotlm_agents.execution.state import LoopState
from hotlm_core.callbacks import CallbackManager
from hotlm_core.config import RunnableConfig
from hotlm_core.memory import BaseMemory
from hotlm_core.models import BaseChatModel
from hotlm_core.prompts import BaseChatPromptTemplate
from hotlm_core.tools import BaseTool

AgentOutput = TypeVar('AgentOutput', str, Dict[str, Any])


class BaseAgent(ABC):
    """High-level Agent class that wraps an AgentLoop for convenience."""
    
    def __init__(
        self,
        llm: BaseChatModel,
        prompt_template: BaseChatPromptTemplate,
        tools: Optional[List[BaseTool]] = None,
        memory: Optional[BaseMemory] = None,
        system_message: Optional[str] = None,
        max_iterations: int = 10,
        early_stopping_method: str = "force",
        callbacks: Optional[List[Any]] = None,
        return_intermediate_steps: bool = False,
        agent_id: Optional[str] = None,
    ):
        """
        Initialize a BaseAgent with the necessary components.
        
        Args:
            llm: Language model to use for generating agent responses
            prompt_template: Template to format messages for the LLM
            tools: List of tools available to the agent
            memory: Optional memory to store conversation history
            system_message: Optional system message to include at the start
            max_iterations: Maximum number of iterations before forcing termination
            early_stopping_method: How to handle max iterations ("force" or "generate")
            callbacks: Optional list of callback handlers
            return_intermediate_steps: Whether to include intermediate steps in result
            agent_id: Optional unique identifier for this agent instance
        """
        self.loop = AgentLoop(
            llm=llm,
            prompt_template=prompt_template,
            tools=tools or [],
            memory=memory,
            system_message=system_message,
            max_iterations=max_iterations,
            early_stopping_method=early_stopping_method,
            return_intermediate_steps=return_intermediate_steps,
        )
        
        self.agent_id = agent_id or id(self)
        self.callbacks = callbacks or []
        self.callback_manager = CallbackManager(self.callbacks) if self.callbacks else None
        
        # Agent metadata for tracking
        self.metadata: Dict[str, Any] = {
            "agent_id": self.agent_id,
            "agent_type": self.__class__.__name__,
            "tool_count": len(tools or []),
            "tool_names": [tool.name for tool in (tools or [])],
        }
    
    def run(
        self, 
        user_input: str, 
        config: Optional[RunnableConfig] = None
    ) -> AgentOutput:
        """
        Execute the agent synchronously and return the final output.
        
        Args:
            user_input: User's input to the agent
            config: Optional configuration for the run
            
        Returns:
            Either a string response or a dictionary with the response and intermediate steps
        """
        if self.callback_manager:
            if not config:
                config = RunnableConfig()
            config.callback_manager = self.callback_manager
            
        return self.loop.run(user_input, config=config, callbacks=self.callbacks)
    
    async def arun(
        self, 
        user_input: str, 
        config: Optional[RunnableConfig] = None
    ) -> AgentOutput:
        """
        Execute the agent asynchronously and return the final output.
        
        Args:
            user_input: User's input to the agent
            config: Optional configuration for the run
            
        Returns:
            Either a string response or a dictionary with the response and intermediate steps
        """
        if self.callback_manager:
            if not config:
                config = RunnableConfig()
            config.callback_manager = self.callback_manager
            
        return await self.loop.arun(user_input, config=config, callbacks=self.callbacks)
    
    def stream(
        self, 
        user_input: str, 
        config: Optional[RunnableConfig] = None
    ) -> Iterator[str]:
        """
        Stream the agent's thinking process and final response.
        
        Args:
            user_input: User's input to the agent
            config: Optional configuration for the run
            
        Returns:
            Iterator yielding chunks of the agent's response
        """
        if self.callback_manager:
            if not config:
                config = RunnableConfig()
            config.callback_manager = self.callback_manager
            
        return self.loop.stream(user_input, config=config, callbacks=self.callbacks)
    
    async def astream(
        self, 
        user_input: str, 
        config: Optional[RunnableConfig] = None
    ) -> AsyncIterator[str]:
        """
        Asynchronously stream the agent's thinking process and final response.
        
        Args:
            user_input: User's input to the agent
            config: Optional configuration for the run
            
        Returns:
            Async iterator yielding chunks of the agent's response
        """
        if self.callback_manager:
            if not config:
                config = RunnableConfig()
            config.callback_manager = self.callback_manager
            
        return self.loop.astream(user_input, config=config, callbacks=self.callbacks)
    
    @abstractmethod
    def get_prompt_template(self) -> BaseChatPromptTemplate:
        """
        Get the prompt template for this agent.
        
        This method should be implemented by subclasses to provide the specific
        prompt template required by the agent.
        
        Returns:
            A prompt template for the agent
        """
        raise NotImplementedError("Subclasses must implement get_prompt_template")


class SimpleAgent(BaseAgent):
    """A concrete implementation of BaseAgent that uses a default prompt template."""
    
    def get_prompt_template(self) -> BaseChatPromptTemplate:
        """Return a default prompt template that works with most simple tasks."""
        from hotlm_core.prompts import ChatPromptTemplate
        from hotlm_core.schema import SystemMessage
        
        system_template = """You are a helpful AI assistant that can use tools to solve problems.
        When you need to use a tool, format your response as:
        
        ```json
        {"tool": "tool_name", "args": {"arg1": "value1", "arg2": "value2"}}
        ```
        
        Otherwise, just respond directly to the user's request.
        """
        
        return ChatPromptTemplate.from_messages([
            SystemMessage(content=system_template),
            {"role": "placeholder", "content": "{messages}"},
        ])
