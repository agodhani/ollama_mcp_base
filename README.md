# Ollama MCP Base Template

A ready-to-clone base template for building [Model Context Protocol (MCP)](https://modelcontextprotocol.io) servers powered by [Ollama](https://ollama.com). Clone it, add your tools, and run.

---

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| [Python](https://python.org) | ≥ 3.11 | |
| [uv](https://docs.astral.sh/uv/) | latest | package/env manager |
| [Ollama](https://ollama.com) | latest | must be running locally |

Install Ollama and pull a model before starting:

```bash
ollama pull llama3.2
```

---

## Setup

```bash
# 1. Clone / copy the template
git clone <your-repo-url>
cd ollama-mcp-base-template

# 2. Install dependencies
uv sync

# 3. Run the server
uv run python main.py
# or via the registered script entry point:
uv run serve
```

---

## Project Structure

```
.
├── main.py                  # Entry point — subclass BaseMCPServer here
├── pyproject.toml
│
├── config/
│   └── settings.py          # All configuration, env-var overrides
│
├── server/
│   └── base_server.py       # BaseMCPServer + OllamaIntegrationMixin
│
├── client/
│   └── base_client.py       # BaseMCPClient for connecting to servers
│
└── models/
    └── ollama_manager.py    # OllamaManager: pull/create/list/delete models
```

---

## Adding Your Own Tools

Open [main.py](main.py) and extend `MyMCPServer.register_tools()`:

```python
async def register_tools(self):
    self.add_tool(
        name="get_weather",
        description="Get the current weather for a city",
        input_schema={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
            },
            "required": ["city"],
        },
        handler=self._get_weather,
    )

async def _get_weather(self, args: dict) -> str:
    city = args["city"]
    # ... your implementation
    return f"It is sunny in {city}."
```

### Adding Resources

Resources expose readable data (files, DB queries, API snapshots) to the client.

```python
async def register_resources(self):
    self.add_resource(
        uri="data://config",
        name="Server Config",
        description="Current server configuration as JSON",
        mime_type="application/json",
        handler=self._config_resource,
    )

async def _config_resource(self) -> str:
    import json
    return json.dumps({"model": self.model_name, "host": self.ollama_host})
```

### Adding Prompt Templates

Prompts are reusable message templates the client can request with variables filled in.

```python
async def register_prompts(self):
    self.add_prompt(
        name="code_review",
        description="Review a code snippet",
        arguments=[{"name": "code", "description": "Code to review", "required": True}],
        handler=self._code_review_prompt,
    )

async def _code_review_prompt(self, args: dict) -> list:
    return [
        {"role": "user", "content": f"Please review this code:\n\n{args['code']}"}
    ]
```

---

## Using Ollama Inference Inside a Tool

Mix in `OllamaIntegrationMixin` to call Ollama directly from your tool handlers:

```python
from server.base_server import BaseMCPServer, OllamaIntegrationMixin

class MyServer(OllamaIntegrationMixin, BaseMCPServer):
    async def register_tools(self):
        self.add_tool(
            name="summarize",
            description="Summarize text using the configured Ollama model",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            handler=self._summarize,
        )

    async def _summarize(self, args: dict) -> str:
        return await self.call_ollama(
            prompt=f"Summarize the following:\n\n{args['text']}",
            system="You are a concise summarizer. Reply in 2-3 sentences.",
        )
```

`call_ollama()` — single-turn generation
`call_ollama_chat()` — multi-turn with conversation history

---

## Using the Client

`BaseMCPClient` spawns a server process and lets you interact with it programmatically.

```python
import asyncio
from client.base_client import BaseMCPClient

async def main():
    async with BaseMCPClient(server_script_path="./main.py") as client:
        await client.discover()

        # List what the server exposes
        print([t.name for t in client.list_tools()])

        # Call a tool
        result = await client.call_tool("echo", {"text": "hello"})
        print(result)

        # Read a resource
        content = await client.read_resource("data://config")

asyncio.run(main())
```

---

## Configuration

All settings can be overridden with environment variables (no code changes needed).

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `llama3.2` | Default model for inference |
| `OLLAMA_TIMEOUT` | `120` | Request timeout (seconds) |
| `OLLAMA_TEMPERATURE` | `0.7` | Sampling temperature |
| `SERVER_NAME` | `base-mcp-server` | MCP server name shown to clients |
| `SERVER_VERSION` | `1.0.0` | Server version string |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

You can export these in your shell or place them in a `.env` file and source it:

```bash
export OLLAMA_MODEL=mistral
export LOG_LEVEL=DEBUG
uv run python main.py
```

---

## Managing Ollama Models

Use `OllamaManager` to manage models programmatically:

```python
import asyncio
from models.ollama_manager import OllamaManager

async def main():
    manager = OllamaManager()

    if not await manager.health_check():
        raise RuntimeError("Ollama is not running")

    models = await manager.list_models()
    print([m["name"] for m in models])

    await manager.pull_model("mistral")

asyncio.run(main())
```

### Custom Models via Modelfile

```python
await manager.create_model(
    name="my-coding-assistant",
    modelfile_path="./modelfiles/coding-assistant.modelfile",
)
```

Example `Modelfile`:

```
FROM llama3.2

SYSTEM You are an expert Python developer. Be concise and practical.

PARAMETER temperature 0.5
PARAMETER top_p 0.9
```

---

## Quick Checklist When Cloning

- [ ] `uv sync` to install dependencies
- [ ] Rename `MyMCPServer` in [main.py](main.py)
- [ ] Replace the `echo` tool with your own tools
- [ ] Set `SERVER_NAME` and `OLLAMA_MODEL` in your environment
- [ ] Run `uv run serve` and verify tools appear in your MCP client
