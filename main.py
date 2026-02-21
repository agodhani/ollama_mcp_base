"""
Ollama MCP Base Template

Entry point for your MCP server. Clone this template, then:
  1. Rename MyMCPServer to something meaningful
  2. Replace the echo tool with your own tools in register_tools()
  3. Optionally override register_resources() and register_prompts()
  4. Run: uv run python main.py  (or: uv run serve)

To use Ollama inference inside a tool, mix in OllamaIntegrationMixin:
    class MyServer(OllamaIntegrationMixin, BaseMCPServer):
        async def _my_tool(self, args: dict) -> str:
            return await self.call_ollama(args["prompt"])
"""

import asyncio
import logging

from config.settings import settings
from server.base_server import BaseMCPServer

# Configure logging once at the application entry point (not inside library modules)
logging.basicConfig(
    level=getattr(logging, settings.server.log_level, logging.INFO),
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
)


class MyMCPServer(BaseMCPServer):
    """
    Your custom MCP server — rename and extend this class.

    Add tools in register_tools(), resources in register_resources(),
    and prompt templates in register_prompts().
    """

    async def register_tools(self):
        # ----------------------------------------------------------------
        # Add your tools here.  Remove or replace the echo example below.
        # ----------------------------------------------------------------
        self.add_tool(
            name="echo",
            description="Echoes back the provided text. Useful for testing.",
            input_schema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to echo back",
                    },
                },
                "required": ["text"],
            },
            handler=self._echo,
        )

    async def _echo(self, args: dict) -> str:
        return args["text"]

    # Uncomment to expose resources:
    # async def register_resources(self):
    #     self.add_resource(
    #         uri="data://status",
    #         name="Server Status",
    #         description="Current server status information",
    #         mime_type="application/json",
    #         handler=self._status_resource,
    #     )
    #
    # async def _status_resource(self) -> str:
    #     import json
    #     return json.dumps({"status": "ok", "model": self.model_name})


def main():
    server = MyMCPServer(
        name=settings.server.name,
        version=settings.server.version,
        model_name=settings.ollama.default_model,
        ollama_host=settings.ollama.host,
    )
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
