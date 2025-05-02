"""Example: Agent with LLM Integration and Conversation Buffer Memory."""

import asyncio
import os

from dotenv import load_dotenv

# Import necessary components
from hotlm_agents.agents import BaseAgent
from hotlm_agents.memory_layers import ConversationBufferMemory
from hotlm_core.memory.history import (
    InMemoryChatMessageHistory,  # Use core implementation
)
from hotlm_core.prompts import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from hotlm_core.schema import SystemMessage
from hotlm_integrations.llms.openai import OpenAIAdapter

# Load environment variables
load_dotenv()

async def main():
    print("Starting agent with memory example...")

    # 1. Initialize LLM Adapter
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        return
    adapter = OpenAIAdapter(api_key=api_key)

    # 2. Initialize Memory
    # Use the core InMemoryChatMessageHistory as the store
    chat_history = InMemoryChatMessageHistory()
    # Configure buffer memory to return messages and use the history store
    # The memory_key="history" matches the variable name in the prompt
    memory = ConversationBufferMemory(
        chat_memory=chat_history,
        memory_key="history", # Key for prompt template
        return_messages=True # LLM expects messages, not a string
    )

    # 3. Create Prompt Template with Memory
    # This prompt includes a placeholder for the memory variables (history)
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"), # Where memory messages go
        HumanMessagePromptTemplate.from_template("{input}") # The current user input
    ])

    # 4. Initialize Agent with LLM, Memory, and Prompt
    # BaseAgent automatically uses the AgentLoop with the provided components
    agent = BaseAgent(llm=adapter, memory=memory, tools=[], prompt=prompt)

    # 5. Run multiple interactions to demonstrate memory
    try:
        print("--- Interaction 1 ---")
        inputs1 = {"input": "My name is Bob."}
        result1 = await agent.arun(inputs1["input"]) # Pass only the input string
        print(f"Input: {inputs1['input']}")
        print(f"Output: {result1}")
        print("-" * 20)

        # Wait a moment to ensure interactions are distinct if needed
        await asyncio.sleep(1)

        print("--- Interaction 2 ---")
        inputs2 = {"input": "What is my name?"}
        result2 = await agent.arun(inputs2["input"]) # Agent should use memory here
        print(f"Input: {inputs2['input']}")
        print(f"Output: {result2}") # Should hopefully remember "Bob"
        print("-" * 20)

        print("--- Current Memory State ---")
        # Load memory variables to see what's stored
        current_memory = memory.load_memory_variables({}) # Pass empty dict as no specific input needed
        print(current_memory.get("history", "Memory is empty or key mismatch."))

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())

