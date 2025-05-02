"""Example: Basic Agent with LLM Integration (No Memory/Tools yet)."""

import asyncio
import os

from dotenv import load_dotenv

# Import necessary components
from hotlm_agents.agents import BaseAgent
from hotlm_core.schema import HumanMessage

# Assuming AgentLoop is not directly used here, BaseAgent handles it
# from hotlm_agents.execution import AgentLoop
from hotlm_integrations.llms.openai import OpenAIAdapter  # Using OpenAI for example

# Load environment variables (e.g., API keys)
load_dotenv()

async def main():
    print("Starting basic LLM agent example...")

    # 1. Initialize the LLM Adapter
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        return
    adapter = OpenAIAdapter(api_key=api_key)

    # 2. Initialize the Agent (using BaseAgent which wraps AgentLoop)
    # BaseAgent needs llm, tools, and prompt. We need a basic prompt.
    # Let's create a simple prompt template inline for this example.
    from hotlm_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    # Basic prompt - expects 'messages' input variable from the loop
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="messages"),
    ])

    agent = BaseAgent(llm=adapter, tools=[], prompt=prompt)

    # 3. Run the agent
    inputs = {"input": "Hello, world! Tell me a short joke."}
    print(f"Agent Input: {inputs['input']}")

    try:
        # BaseAgent's arun method handles the loop execution
        result = await agent.arun(inputs["input"]) # Pass just the user input string
        print(f"Agent Output: {result}")
    except Exception as e:
        print(f"An error occurred: {e}")
        # Add more specific error handling if needed, e.g., for API keys

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
