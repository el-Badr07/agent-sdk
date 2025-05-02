import asyncio
import json
import logging
import re
import time
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Union

from hotlm_core.callbacks import CallbackManager, ConsoleCallbackHandler
from hotlm_core.config import RunnableConfig
from hotlm_core.errors import HotLMToolError, HotLMValidationError
from hotlm_core.memory import BaseMemory
from hotlm_core.models import BaseChatModel
from hotlm_core.prompts import BaseChatPromptTemplate
from hotlm_core.schema import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    ToolObservation,
)
from hotlm_core.tools import BaseTool

from .state import AgentFinish, LoopState

logger = logging.getLogger("hotlm_agents")

class AgentLoop:
    """Core agent execution loop: LLM call, tool handling, termination."""
    def __init__(
        self,
        llm: BaseChatModel,
        prompt_template: BaseChatPromptTemplate,
        tools: List[BaseTool],
        memory: Optional[BaseMemory] = None,
        system_message: Optional[str] = None,
        max_iterations: int = 10,
        early_stopping_method: str = "force",  # Options: "force", "generate"
        return_intermediate_steps: bool = False,
    ):
        """Initialize an AgentLoop for executing agent reasoning cycles.
        
        Args:
            llm: Language model to use for generating agent responses
            prompt_template: Template to format messages for the LLM
            tools: List of tools available to the agent
            memory: Optional memory to store conversation history
            system_message: Optional system message to include at the start
            max_iterations: Maximum number of iterations before forcing termination
            early_stopping_method: How to handle max iterations ("force" or "generate")
            return_intermediate_steps: Whether to include intermediate steps in result
        """
        self.llm = llm
        self.prompt_template = prompt_template
        self.tools = {tool.name: tool for tool in tools}
        self.memory = memory
        self.system_message = system_message
        self.max_iterations = max_iterations
        self.early_stopping_method = early_stopping_method
        self.return_intermediate_steps = return_intermediate_steps

        # Support tool descriptions in prompt
        self.tool_descriptions = [
            f"{tool.name}: {tool.description}" for tool in tools
        ]

    def run(
        self,
        user_input: str,
        config: Optional[RunnableConfig] = None,
        callbacks: Optional[List[Any]] = None,
        **kwargs: Any, # Allow passing additional inputs
    ) -> Union[str, Dict[str, Any]]:
        """Synchronously execute the agent loop until terminal response.
        
        Args:
            user_input: User's input to the agent (often used as the primary input)
            config: Optional configuration for the run
            callbacks: Optional callback handlers to use for this run
            **kwargs: Additional key-value pairs to include in the initial state
                      and potentially used by memory loading/saving.
            
        Returns:
            Either a string response or a dictionary with the response and intermediate steps
        """
        # Set up callback manager if provided
        if callbacks:
            callback_manager = CallbackManager(callbacks)
            config = config or RunnableConfig()
            config.callback_manager = callback_manager
            
        # Initialize state with user_input and any other kwargs
        initial_inputs = {"input": user_input, **kwargs}
        state = self._initialize_state(initial_inputs)
        
        # Main loop
        try:
            iteration_count = 0
            while not state.done and iteration_count < self.max_iterations:
                iteration_count += 1
                # Pass initial_inputs for memory loading context if needed
                # In a non-streaming run, the 'current_inputs' for memory loading
                # are typically the same as the initial inputs that started the run.
                state = self._execute_step(state, initial_inputs, config)
                
            # Handle max iterations if needed
            if iteration_count >= self.max_iterations and not state.done:
                if self.early_stopping_method == "force":
                    logger.warning(f"Reached max iterations ({self.max_iterations}). Forcing termination.")
                    state.done = True
                    state.final_output = f"I've spent too much time on this task without finding a complete solution. Here's what I know so far: {state.messages[-1].content}"
                elif self.early_stopping_method == "generate":
                    # Ask LLM to provide a final answer based on what it knows so far
                    final_prompt_msgs = self.prompt_template.invoke({
                        "messages": state.messages + [
                            SystemMessage(content="You have run out of steps to solve this problem. Provide your best answer based on what you've learned so far.")
                        ]
                    })
                    final_ai_message = self.llm.invoke(final_prompt_msgs, config=config)
                    state.done = True
                    state.final_output = final_ai_message.content
            
            # Update memory if needed, using the initial inputs and final output
            if self.memory and state.done and state.final_output:
                # Construct output dict based on memory's expected key or default
                output_key = getattr(self.memory, 'output_key', None) or "output"
                final_outputs = {output_key: state.final_output}
                # Save context using the initial inputs and the final outputs
                self.memory.save_context(initial_inputs, final_outputs)
                
            # Return appropriate format
            if self.return_intermediate_steps:
                return {
                    "output": state.final_output,
                    "intermediate_steps": state.intermediate_steps,
                }
            return state.final_output
            
        except Exception as e:
            logger.exception(f"Error in agent execution: {e}")
            return f"I encountered an error while processing your request: {str(e)}"
    
    async def arun(
        self, 
        user_input: str, 
        config: Optional[RunnableConfig] = None,
        callbacks: Optional[List[Any]] = None,
        **kwargs: Any, # Allow passing additional inputs
    ) -> Union[str, Dict[str, Any]]:
        """Asynchronously execute the agent loop until terminal response."""
        # Set up callback manager if provided
        if callbacks:
            callback_manager = CallbackManager(callbacks)
            config = config or RunnableConfig()
            config.callback_manager = callback_manager
            
        # Initialize state with user_input and any other kwargs
        initial_inputs = {"input": user_input, **kwargs}
        state = self._initialize_state(initial_inputs) # Assuming _initialize_state is sync
        
        # Main loop
        try:
            iteration_count = 0
            while not state.done and iteration_count < self.max_iterations:
                iteration_count += 1
                 # Pass initial_inputs for memory loading context if needed
                state = await self._aexecute_step(state, initial_inputs, config)
                
            # Handle max iterations if needed
            if iteration_count >= self.max_iterations and not state.done:
                if self.early_stopping_method == "force":
                    logger.warning(f"Reached max iterations ({self.max_iterations}). Forcing termination.")
                    state.done = True
                    state.final_output = f"I've spent too much time on this task without finding a complete solution. Here's what I know so far: {state.messages[-1].content}"
                elif self.early_stopping_method == "generate":
                    # Ask LLM to provide a final answer based on what it knows so far
                    final_prompt_msgs = self.prompt_template.invoke({
                        "messages": state.messages + [
                            SystemMessage(content="You have run out of steps to solve this problem. Provide your best answer based on what you've learned so far.")
                        ]
                    })
                    final_ai_message = await self.llm.ainvoke(final_prompt_msgs, config=config)
                    state.done = True
                    state.final_output = final_ai_message.content
            
            # Update memory if needed, using the initial inputs and final output
            if self.memory and state.done and state.final_output:
                 # Construct output dict based on memory's expected key or default
                output_key = getattr(self.memory, 'output_key', None) or "output"
                final_outputs = {output_key: state.final_output}
                # Save context using the initial inputs and the final outputs
                # Use asave_context if available, otherwise fallback or raise error
                if hasattr(self.memory, 'asave_context'):
                    await self.memory.asave_context(initial_inputs, final_outputs)
                else:
                    # Fallback to sync save_context if async version doesn't exist
                    # Consider adding a warning or making async memory mandatory for arun
                    self.memory.save_context(initial_inputs, final_outputs)
                
            # Return appropriate format
            if self.return_intermediate_steps:
                return {
                    "output": state.final_output,
                    "intermediate_steps": state.intermediate_steps,
                }
            return state.final_output
            
        except Exception as e:
             # ... existing error handling ...
            logger.exception(f"Error in agent execution: {e}")
            return f"I encountered an error while processing your request: {str(e)}"

    def stream(
        self, 
        user_input: str, 
        config: Optional[RunnableConfig] = None,
        callbacks: Optional[List[Any]] = None,
         **kwargs: Any, # Allow passing additional inputs
    ) -> Iterator[str]:
        """Stream the agent's thinking process and final response."""
        # Set up callback manager if provided
        if callbacks:
            callback_manager = CallbackManager(callbacks)
            config = config or RunnableConfig()
            config.callback_manager = callback_manager
            
        # Initialize state with user_input and any other kwargs
        initial_inputs = {"input": user_input, **kwargs}
        state = self._initialize_state(initial_inputs)
        
        # Yield initial thinking marker
        yield "Thinking...\n"
        
        # Main loop
        try:
            iteration_count = 0
            while not state.done and iteration_count < self.max_iterations:
                iteration_count += 1
                
                # Execute step
                previous_message_count = len(state.messages)
                 # Pass initial_inputs for memory loading context if needed
                state = self._execute_step(state, initial_inputs, config)
                
                # Yield new messages
                for msg in state.messages[previous_message_count:]:
                    if isinstance(msg, AIMessage):
                        yield f"Agent: {msg.content}\n"
                    elif isinstance(msg, ToolMessage):
                        yield f"Tool result: {msg.content}\n"
                
            # Handle max iterations if needed
            if iteration_count >= self.max_iterations and not state.done:
                yield "Reached maximum number of thinking steps. Concluding with current understanding.\n"
                if self.early_stopping_method == "force":
                    state.done = True
                    state.final_output = f"I've spent too much time on this task without finding a complete solution. Here's what I know so far: {state.messages[-1].content}"
                elif self.early_stopping_method == "generate":
                    yield "Finalizing answer based on current information...\n"
                    final_prompt_msgs = self.prompt_template.invoke({
                        "messages": state.messages + [
                            SystemMessage(content="You have run out of steps to solve this problem. Provide your best answer based on what you've learned so far.")
                        ]
                    })
                    final_ai_message = self.llm.invoke(final_prompt_msgs, config=config)
                    state.done = True
                    state.final_output = final_ai_message.content
            
            # Update memory if needed
            if self.memory and state.done and state.final_output:
                output_key = getattr(self.memory, 'output_key', None) or "output"
                final_outputs = {output_key: state.final_output}
                self.memory.save_context(initial_inputs, final_outputs)
                
            # Yield final answer
            if state.done and state.final_output:
                yield f"\nFinal answer: {state.final_output}\n"
                
        except Exception as e:
            logger.exception(f"Error in agent execution: {e}")
            yield f"\nError: I encountered an error while processing your request: {str(e)}\n"

    async def astream(
        self, 
        user_input: str, 
        config: Optional[RunnableConfig] = None,
        callbacks: Optional[List[Any]] = None,
         **kwargs: Any, # Allow passing additional inputs
    ) -> AsyncIterator[str]:
        """Asynchronously stream the agent's thinking process and final response."""
        # Set up callback manager if provided
        if callbacks:
            callback_manager = CallbackManager(callbacks)
            config = config or RunnableConfig()
            config.callback_manager = callback_manager
            
        # Initialize state with user_input and any other kwargs
        initial_inputs = {"input": user_input, **kwargs}
        state = self._initialize_state(initial_inputs) # Assuming sync init
        
        # Yield initial thinking marker
        yield "Thinking...\n"
        
        # Main loop
        try:
            iteration_count = 0
            while not state.done and iteration_count < self.max_iterations:
                iteration_count += 1
                
                # Execute step
                previous_message_count = len(state.messages)
                 # Pass initial_inputs for memory loading context if needed
                state = await self._aexecute_step(state, initial_inputs, config)
                
                # Yield new messages
                for msg in state.messages[previous_message_count:]:
                    if isinstance(msg, AIMessage):
                        yield f"Agent: {msg.content}\n"
                    elif isinstance(msg, ToolMessage):
                        yield f"Tool result: {msg.content}\n"
                
            # Handle max iterations if needed
            if iteration_count >= self.max_iterations and not state.done:
                yield "Reached maximum number of thinking steps. Concluding with current understanding.\n"
                if self.early_stopping_method == "force":
                    state.done = True
                    state.final_output = f"I've spent too much time on this task without finding a complete solution. Here's what I know so far: {state.messages[-1].content}"
                elif self.early_stopping_method == "generate":
                    yield "Finalizing answer based on current information...\n"
                    final_prompt_msgs = self.prompt_template.invoke({
                        "messages": state.messages + [
                            SystemMessage(content="You have run out of steps to solve this problem. Provide your best answer based on what you've learned so far.")
                        ]
                    })
                    final_ai_message = await self.llm.ainvoke(final_prompt_msgs, config=config)
                    state.done = True
                    state.final_output = final_ai_message.content
            
            # Update memory if needed
            if self.memory and state.done and state.final_output:
                output_key = getattr(self.memory, 'output_key', None) or "output"
                final_outputs = {output_key: state.final_output}
                if hasattr(self.memory, 'asave_context'):
                    await self.memory.asave_context(initial_inputs, final_outputs)
                else:
                    self.memory.save_context(initial_inputs, final_outputs)
                
            # Yield final answer
            if state.done and state.final_output:
                yield f"\nFinal answer: {state.final_output}\n"
                
        except Exception as e:
            logger.exception(f"Error in agent execution: {e}")
            yield f"\nError: I encountered an error while processing your request: {str(e)}\n"

    def _initialize_state(self, initial_inputs: Dict[str, Any]) -> LoopState:
        """Initialize the agent state with initial inputs and system message."""
        state = LoopState()
        
        # Add system message if provided
        if self.system_message:
            system_content = self.system_message
            # If we have tools, append their descriptions to system message
            if self.tools and "tools" not in system_content.lower():
                tool_descriptions = "\n".join(self.tool_descriptions)
                system_content += f"\n\nYou have access to the following tools:\n{tool_descriptions}"
            state.messages.append(SystemMessage(content=system_content))
            
        # Add primary user input (assuming 'input' key exists)
        # Memory loading might use other keys from initial_inputs later
        primary_input_key = getattr(self.memory, 'input_key', None) or "input"
        primary_input = initial_inputs.get(primary_input_key, str(initial_inputs))
        state.messages.append(HumanMessage(content=str(primary_input)))
        return state

    def _execute_step(self, state: LoopState, current_inputs: Dict[str, Any], config: Optional[RunnableConfig] = None) -> LoopState:
        """Execute a single step of the agent loop."""
        # Load memory variables if available, passing current_inputs for context
        memory_vars = {}
        if self.memory:
            # Pass the dictionary representing the current step's context
            memory_vars = self.memory.load_memory_variables(current_inputs)

        # Format messages into prompt, including loaded memory variables
        # Ensure all values passed to the template are appropriate (e.g., strings, lists of messages)
        prompt_inputs = {**current_inputs, **memory_vars, "messages": state.messages}
        # Filter or format as needed by the specific prompt template
        # Example: Ensure 'history' is a string if the template expects it
        if self.memory and not getattr(self.memory, 'return_messages', False) and 'history' in prompt_inputs:
             if not isinstance(prompt_inputs['history'], str):
                 # This assumes get_buffer_string exists and works correctly
                 # from hotlm_core.schema import get_buffer_string
                 # prompt_inputs['history'] = get_buffer_string(prompt_inputs['history'])
                 prompt_inputs['history'] = str(prompt_inputs['history']) # Simple fallback

        # Remove keys that might not be expected by the template if necessary
        # e.g., remove the raw 'input' if memory provides 'history'
        # if 'history' in prompt_inputs and 'input' in prompt_inputs:
        #     del prompt_inputs['input']

        prompt_msgs = self.prompt_template.invoke(prompt_inputs)

        # Call LLM
        ai_message = self.llm.invoke(prompt_msgs, config=config)
        state.messages.append(ai_message)

        # Parse tool calls - try multiple formats
        tool_calls = self._parse_tool_calls(ai_message.content)

        if tool_calls:
            # ... existing tool execution logic ...
            for tool_name, tool_args in tool_calls:
                tool = self.tools.get(tool_name)
                if tool:
                    try:
                        # Add to intermediate steps
                        state.intermediate_steps.append((tool_name, tool_args))

                        # Execute tool
                        start_time = time.time()
                        result = tool.invoke(tool_args, config=config)
                        execution_time = time.time() - start_time

                        # Add observation
                        state.messages.append(
                            ToolMessage(
                                content=str(result),
                                tool_call_id=tool_name, # Use name as ID for now
                                metadata={"execution_time": execution_time}
                            )
                        )
                    except Exception as e:
                        error_msg = f"Error executing tool {tool_name}: {str(e)}"
                        logger.warning(error_msg)
                        state.messages.append(ToolMessage(content=error_msg, tool_call_id=tool_name))
                else:
                    # ... existing tool not found logic ...
                    state.messages.append(
                        ToolMessage(
                            content=f"Error: Tool '{tool_name}' not found. Available tools: {', '.join(self.tools.keys())}",
                            tool_call_id=tool_name
                        )
                    )
        else:
            # Check for explicit agent finish
            agent_finish = self._parse_agent_finish(ai_message.content)
            if agent_finish:
                state.done = True
                state.final_output = agent_finish.output
            else:
                # No tool calls or explicit finish, treat as final answer
                state.done = True
                state.final_output = ai_message.content

        return state

    async def _aexecute_step(self, state: LoopState, current_inputs: Dict[str, Any], config: Optional[RunnableConfig] = None) -> LoopState:
        """Execute a single step of the agent loop asynchronously."""
         # Load memory variables if available, passing current_inputs for context
        memory_vars = {}
        if self.memory:
            # Use aload_memory_variables if available
            if hasattr(self.memory, 'aload_memory_variables'):
                memory_vars = await self.memory.aload_memory_variables(current_inputs)
            else:
                # Fallback to sync load
                memory_vars = self.memory.load_memory_variables(current_inputs)


        # Format messages into prompt, including loaded memory variables
        prompt_inputs = {**current_inputs, **memory_vars, "messages": state.messages}
        # Apply similar filtering/formatting as in _execute_step if needed
        if self.memory and not getattr(self.memory, 'return_messages', False) and 'history' in prompt_inputs:
             if not isinstance(prompt_inputs['history'], str):
                 prompt_inputs['history'] = str(prompt_inputs['history']) # Simple fallback

        prompt_msgs = self.prompt_template.invoke(prompt_inputs) # Assuming template invoke is sync

        # Call LLM
        ai_message = await self.llm.ainvoke(prompt_msgs, config=config)
        state.messages.append(ai_message)

        # Parse tool calls - try multiple formats
        tool_calls = self._parse_tool_calls(ai_message.content)

        if tool_calls:
             # ... existing async tool execution logic ...
            tool_tasks = []
            tool_results = {} # To store results in order

            for idx, (tool_name, tool_args) in tool_calls:
                tool = self.tools.get(tool_name)
                if tool:
                    state.intermediate_steps.append((tool_name, tool_args))
                    # Create task for each tool call
                    tool_tasks.append(
                        self._arun_tool(tool, tool_name, tool_args, idx, config)
                    )
                else:
                    # Handle tool not found immediately
                    error_msg = f"Error: Tool '{tool_name}' not found. Available tools: {', '.join(self.tools.keys())}"
                    tool_results[idx] = ToolMessage(content=error_msg, tool_call_id=tool_name)


            # Run tool tasks concurrently
            if tool_tasks:
                results = await asyncio.gather(*tool_tasks)
                for idx, result_message in results:
                    tool_results[idx] = result_message

            # Add results to state messages in the order they were called
            for i in range(len(tool_calls)):
                 if i in tool_results:
                     state.messages.append(tool_results[i])

        else:
             # Check for explicit agent finish
            agent_finish = self._parse_agent_finish(ai_message.content)
            if agent_finish:
                state.done = True
                state.final_output = agent_finish.output
            else:
                # No tool calls or explicit finish, treat as final answer
                state.done = True
                state.final_output = ai_message.content

        return state

    async def _arun_tool(self, tool: BaseTool, tool_name: str, tool_args: Any, call_index: int, config: Optional[RunnableConfig]) -> tuple[int, ToolMessage]:
        """Helper to run a single tool asynchronously and return its result message."""
        try:
            start_time = time.time()
            # Use ainvoke if available, otherwise run sync invoke in thread
            if hasattr(tool, 'ainvoke'):
                result = await tool.ainvoke(tool_args, config=config)
            else:
                result = await asyncio.to_thread(tool.invoke, tool_args, config=config)
            execution_time = time.time() - start_time
            return call_index, ToolMessage(
                content=str(result),
                tool_call_id=tool_name,
                metadata={"execution_time": execution_time}
            )
        except Exception as e:
            error_msg = f"Error executing tool {tool_name}: {str(e)}"
            logger.warning(error_msg)
            return call_index, ToolMessage(content=error_msg, tool_call_id=tool_name)

    def _parse_tool_calls(self, content: str) -> List[tuple]:
        """Parse tool calls from LLM output using various formats."""
        tool_calls = []
        
        # Try to parse JSON formatted tool call
        json_pattern = r'```json\s*(\{.*?\})\s*```'
        json_matches = re.findall(json_pattern, content, re.DOTALL)
        
        for json_str in json_matches:
            try:
                data = json.loads(json_str)
                if "tool" in data and "args" in data:
                    tool_calls.append((data["tool"], data["args"]))
                    continue
            except json.JSONDecodeError:
                pass
        
        # Try function call format: CALL_TOOL[name](args)
        func_pattern = r'CALL_TOOL\[(\w+)\]\((.*?)\)'
        func_matches = re.findall(func_pattern, content, re.DOTALL)
        
        for name, args_str in func_matches:
            try:
                # Try to parse as JSON first
                args = json.loads(args_str)
            except json.JSONDecodeError:
                try:
                    # Fall back to eval (less secure but more flexible)
                    args = eval(f"dict({args_str})")
                except Exception:
                    # If all else fails, treat as string
                    args = {"input": args_str}
            tool_calls.append((name, args))
        
        # Try another common format: <tool>name</tool> <args>args_dict</args>
        xml_pattern = r'<tool>(.*?)</tool>\s*<args>(.*?)</args>'
        xml_matches = re.findall(xml_pattern, content, re.DOTALL)
        
        for name, args_str in xml_matches:
            try:
                # Try to parse as JSON first
                args = json.loads(args_str)
            except json.JSONDecodeError:
                try:
                    # Fall back to eval (less secure but more flexible)
                    args = eval(f"dict({args_str})")
                except Exception:
                    # If all else fails, treat as string
                    args = {"input": args_str}
            tool_calls.append((name, args))
            
        return tool_calls

    def _parse_agent_finish(self, content: str) -> Optional[AgentFinish]:
        """Parse explicit agent finish from LLM output."""
        final_answer_patterns = [
            r'Final Answer:\s*(.*)',
            r'FINAL_ANSWER\s*(.*)',
            r'<final_answer>(.*?)</final_answer>'
        ]
        
        for pattern in final_answer_patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                return AgentFinish(output=match.group(1).strip())
        
        return None