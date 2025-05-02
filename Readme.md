# Agent SDK

A comprehensive, modular framework for building and testing AI agents with pluggable tools, memory, and LLM integrations.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Repository Structure](#repository-structure)
4. [Installation](#installation)
5. [Getting Started](#getting-started)
6. [Core Modules](#core-modules)
   - [Agents](#agents)
   - [Tools](#tools)
   - [Memory](#memory)
   - [Integrations](#integrations)
   - [Execution & Runners](#execution--runners)
7. [Examples](#examples)
8. [Testing](#testing)
9. [Contributing](#contributing)
10. [License](#license)

---

## Overview

The Agent SDK provides building blocks for creating intelligent AI agents that:

- Orchestrate interactions with Large Language Models (LLMs)
- Invoke external tools via function-calling
- Maintain context using memory resolvers
- Execute tasks and workflows programmatically
- Can be extended with custom modules and integrations

It is designed for flexibility, testability, and rapid prototyping of agent-driven applications.

## Key Features

- **Pluggable Agents**: Define custom agent behaviors and planners.
- **Tool System**: Wrap any Python function or REST API as a `BaseTool` with JSON schema.
- **Memory**: Recency and relevance-based memory resolvers to manage conversational context.
- **LLM Adapters**: Interfaces for OpenAI, Anthropic, Google Vertex, Groq, and Hugging Face.
- **Execution Engines**: Synchronous and asynchronous runners for streaming and batch tasks.
- **Vectorstores**: Qdrant support for embeddings storage and retrieval.
- **Testing**: Built-in pytest integration for unit and integration tests.

## Repository Structure

```
agent-sdk/
├── hotlm_sdk/              # Core Python SDK modules
│   ├── agents/             # Agent & planner implementations
│   ├── tools_impl/         # Built-in tool wrappers (e.g., math_tools)
│   ├── memory/             # Memory resolver strategies
│   ├── integrations/llms/  # LLM provider adapters
│   ├── vectorstores/       # Vector DB interfaces (Qdrant, etc.)
│   └── execution/          # Runners and executors
├── examples/               # Sample scripts and usage demos
├── tests/                  # Unit and integration tests
├── pyproject.toml          # Project metadata and dependencies
└── Readme.md               # This documentation
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-org/agent-sdk.git
   cd agent-sdk
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # on Windows: .\.venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables** (for LLM APIs):
   ```bash
   export OPENAI_API_KEY=your_openai_key
   export ANTHROPIC_API_KEY=your_anthropic_key
   export GOOGLE_API_KEY=your_google_key
   export GROQ_API_KEY=your_groq_key
   ```

## Getting Started

Create a simple agent that answers math questions:

```python
from hotlm_sdk.integrations.llms.openai import OpenAIModel
from hotlm_sdk.agents.planners.llm_planner import LLMPlanner
from hotlm_sdk.agents.tools_impl.math_tools import AddTool

# Initialize LLM adapter
llm = OpenAIModel(model="gpt-4")

# Register tools
tools = [AddTool()]

# Create the planner agent
planner = LLMPlanner(llm=llm, tools=tools)

# Run the agent
result = planner.run("What is 7 + 5?")
print(result)  # → "The answer is 12."
```

## Core Modules

### Agents

- **LLMPlanner**: Builds prompts, handles function-calls, and loops through tool execution until completion.
- **Execution**: Runners that manage synchronous or streaming LLM calls and collect outputs.

### Tools

- **BaseTool**: Abstract base for all tools. Implements `__call__`.
- **Built-ins**: `AddTool`, `SubtractTool`, `MultiplyTool`, `DivideTool` in `tools_impl/math_tools.py`.
- **Custom Tools**: Add your own under `tools_impl/` by subclassing `BaseTool` and providing a JSON schema.

### Memory

- **MemoryResolver** (abstract): Defines `load_memory()`.
- **RecencyMemoryResolver**: Retrieves the most recent messages.
- **RelevanceMemoryResolver**: Retrieves semantically similar tokens using embeddings.

### Integrations

Adapters implement `BaseLanguageModel.generate()` and `chat()`:

- `openai.py` → OpenAI API
- `anthropic.py` → Anthropic Claude
- `google.py` → Google Vertex AI
- `groq.py` → Groq REST API
- `huggingface.py` → Hugging Face Transformers

Each supports optional function-calling parameters.

### Execution & Runners

- **SyncRunner**: Blocks until completion. Good for batch tasks.
- **StreamRunner**: Yields tokens as they arrive from the LLM.

## Examples

See the `examples/` folder for:

- **basic_math.py**: Demonstrates planner with math tools.
- **document_processor.py**: Integrates a PDF parsing tool.
- **chatbot_demo.py**: Full chat agent with memory and tools.

## Testing

Run pytest:

```bash
pytest --maxfail=1 --disable-warnings -q
```

Add new tests under `tests/` following the naming convention `test_*.py`.

## Contributing

1. Fork the repo.
2. Create a feature branch: `git checkout -b feature/YourFeature`
3. Commit changes: `git commit -m "Add awesome feature"`
4. Push branch: `git push origin feature/YourFeature`
5. Open a Pull Request.

Please follow the existing code style (Black, isort, flake8) and add tests for new features.
