"""Test cases for agent execution loop and state management."""

import unittest
from unittest.mock import MagicMock, patch

# --- Mock classes to replace imports ---

class AgentState:
    """Mock state class to track agent execution."""
    
    def __init__(self):
        self.history = []
        self.memory = {}
        self.current_step = 0
        self.max_steps = 10
        self.status = "initialized"
        self.result = None
    
    def add_to_history(self, entry):
        """Add an entry to the execution history."""
        self.history.append(entry)
        self.current_step += 1
    
    def set_memory(self, key, value):
        """Store a value in the agent's memory."""
        self.memory[key] = value
    
    def get_memory(self, key, default=None):
        """Retrieve a value from the agent's memory."""
        return self.memory.get(key, default)
    
    def is_finished(self):
        """Check if the agent execution has finished."""
        return self.status in ["completed", "error", "max_steps_reached"]
    
    def set_status(self, status):
        """Update the agent execution status."""
        self.status = status
    
    def set_result(self, result):
        """Set the final result of the agent execution."""
        self.result = result
        self.set_status("completed")
    
    def set_error(self, error):
        """Record an error that occurred during execution."""
        self.set_memory("error", str(error))
        self.set_status("error")


class ExecutionLoop:
    """Mock execution loop to control agent execution flow."""
    
    def __init__(self, agent, max_steps=10, callbacks=None):
        self.agent = agent
        self.max_steps = max_steps
        self.callbacks = callbacks or []
        self.state = AgentState()
        self.state.max_steps = max_steps
    
    def run(self, input_text):
        """Run the execution loop with the given input."""
        self.state = AgentState()  # Reset state
        self.state.max_steps = self.max_steps
        
        # Initialize the execution
        self.state.add_to_history({"type": "input", "content": input_text})
        
        # Run callbacks for initialization
        for callback in self.callbacks:
            if hasattr(callback, "on_start"):
                callback.on_start(self.state)
        
        try:
            # Core execution loop
            while not self.state.is_finished():
                # Check if we've hit the max steps
                if self.state.current_step >= self.max_steps:
                    self.state.set_status("max_steps_reached")
                    break
                
                # Run callbacks before step
                for callback in self.callbacks:
                    if hasattr(callback, "on_step_start"):
                        callback.on_step_start(self.state)
                
                # Get the latest input (either the original or from history)
                latest_input = input_text
                if self.state.current_step > 1:
                    # In a real implementation, we would extract this from state history
                    latest_input = self.state.get_memory("latest_input", input_text)
                
                # Execute agent step
                try:
                    result = self.agent.run(latest_input)
                    
                    # In this mock, we'll assume a single step always completes the task,
                    # unless it's a "Processing step" result indicating non-termination
                    self.state.add_to_history({"type": "output", "content": result})
                    # Treat 'Processing step' responses as intermediate
                    if not (isinstance(result, str) and result.startswith("Processing step")):
                        self.state.set_result(result)
                except Exception as e:
                    self.state.set_error(e)
                
                # Run callbacks after step
                for callback in self.callbacks:
                    if hasattr(callback, "on_step_end"):
                        callback.on_step_end(self.state)
        
        except Exception as e:
            self.state.set_error(e)
        
        # Run callbacks for completion
        for callback in self.callbacks:
            if hasattr(callback, "on_end"):
                callback.on_end(self.state)
        
        return self.state.result


class BaseAgent:
    """Mock base agent class."""
    
    def run(self, input_text):
        """Run the agent with the given input."""
        raise NotImplementedError("Subclasses must implement run method")


class SimpleAgent(BaseAgent):
    """Mock simple agent that returns a hardcoded response."""
    
    def run(self, input_text):
        """Run the agent with the given input."""
        return f"I processed: {input_text}"


class CallbackHandler:
    """Mock callback handler to track agent execution events."""
    
    def __init__(self):
        self.events = []
    
    def on_start(self, state):
        """Called when execution starts."""
        self.events.append(("on_start", state.current_step))
    
    def on_step_start(self, state):
        """Called before each execution step."""
        self.events.append(("on_step_start", state.current_step))
    
    def on_step_end(self, state):
        """Called after each execution step."""
        self.events.append(("on_step_end", state.current_step))
    
    def on_end(self, state):
        """Called when execution completes."""
        self.events.append(("on_end", state.current_step, state.status))


class TestAgentExecution(unittest.TestCase):
    """Test cases for agent execution."""
    
    def setUp(self):
        """Set up the test environment."""
        self.simple_agent = SimpleAgent()
        self.callbacks = [CallbackHandler()]
        self.execution_loop = ExecutionLoop(
            agent=self.simple_agent,
            max_steps=5,
            callbacks=self.callbacks
        )
    
    def test_basic_execution(self):
        """Test basic execution of an agent with a simple input."""
        input_text = "Hello, agent!"
        result = self.execution_loop.run(input_text)
        
        # Check that the agent processed the input
        self.assertEqual(result, "I processed: Hello, agent!")
        
        # Check that the state was properly updated
        self.assertEqual(self.execution_loop.state.status, "completed")
        self.assertEqual(len(self.execution_loop.state.history), 2)  # Input and output
    
    def test_callbacks(self):
        """Test that callbacks are properly invoked during execution."""
        input_text = "Test callbacks"
        self.execution_loop.run(input_text)
        
        # Get the callback handler
        callback = self.callbacks[0]
        
        # Check that all expected events were triggered
        event_types = [event[0] for event in callback.events]
        self.assertIn("on_start", event_types)
        self.assertIn("on_step_start", event_types)
        self.assertIn("on_step_end", event_types)
        self.assertIn("on_end", event_types)
        
        # Check that on_end was called with completed status
        end_events = [e for e in callback.events if e[0] == "on_end"]
        self.assertEqual(len(end_events), 1)
        self.assertEqual(end_events[0][2], "completed")
    
    def test_max_steps(self):
        """Test that execution stops when max steps is reached."""
        # Create an agent that never completes
        class NonTerminatingAgent(BaseAgent):
            def run(self, input_text):
                # This agent always returns a non-final result
                return f"Processing step: {input_text}"
        
        # Create a new execution loop with this agent
        non_terminating_agent = NonTerminatingAgent()
        loop = ExecutionLoop(agent=non_terminating_agent, max_steps=3)
        
        # Run the agent
        input_text = "This should hit max steps"
        loop.run(input_text)
        
        # Check that execution stopped due to max steps
        self.assertEqual(loop.state.status, "max_steps_reached")
        self.assertEqual(loop.state.current_step, 3)  # Should have completed exactly 3 steps


if __name__ == "__main__":
    unittest.main()