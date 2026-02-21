"""
Base MCP Server Implementation

This module provides a base class for creating MCP (Model Context Protocol) servers
that integrate with Ollama for AI model inference. The server handles:
- MCP protocol communication
- Tool/resource/prompt registration and execution
- Ollama model inference via OllamaIntegrationMixin

Key Concepts:
- MCP Server: Exposes tools and resources that clients can use
- Tools: Functions the AI can call (e.g. search, calculator)
- Resources: Data sources the AI can read (e.g. files, databases)
- Prompts: Reusable prompt templates

Usage:
    class MyServer(BaseMCPServer):
        async def register_tools(self):
            self.add_tool(
                name="my_tool",
                description="Does something useful",
                input_schema={
                    "type": "object",
                    "properties": {"input": {"type": "string"}},
                    "required": ["input"],
                },
                handler=self._my_tool_handler,
            )

        async def _my_tool_handler(self, args: dict) -> str:
            return f"Result: {args['input']}"

    server = MyServer(model_name="llama3.2")
    asyncio.run(server.run())
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

import httpx
import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

logger = logging.getLogger(__name__)


class ToolDefinition:
    """Defines a tool exposed via MCP."""

    __slots__ = ("name", "description", "input_schema", "handler")

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: Callable,
    ):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.handler = handler


class ResourceDefinition:
    """Defines a resource accessible via MCP."""

    __slots__ = ("uri", "name", "description", "mime_type", "handler")

    def __init__(
        self,
        uri: str,
        name: str,
        description: str,
        mime_type: str,
        handler: Callable,
    ):
        self.uri = uri
        self.name = name
        self.description = description
        self.mime_type = mime_type
        self.handler = handler


class PromptDefinition:
    """Defines a reusable prompt template."""

    __slots__ = ("name", "description", "arguments", "handler")

    def __init__(
        self,
        name: str,
        description: str,
        arguments: List[Dict[str, Any]],
        handler: Callable,
    ):
        self.name = name
        self.description = description
        self.arguments = arguments
        self.handler = handler


class BaseMCPServer(ABC):
    """
    Base class for MCP servers with optional Ollama integration.

    Subclass this and implement register_tools() to create your server.
    register_resources() and register_prompts() are optional overrides.

    Architecture:
    1. Server initializes and registers capabilities
    2. Client connects and discovers available tools/resources/prompts
    3. Client sends requests; server executes handlers and returns results
    """

    def __init__(
        self,
        name: str = "base-mcp-server",
        version: str = "1.0.0",
        model_name: str = "llama3.2",
        ollama_host: str = "http://localhost:11434",
    ):
        self.name = name
        self.version = version
        self.model_name = model_name
        self.ollama_host = ollama_host

        self.server = Server(name)

        self.tools: Dict[str, ToolDefinition] = {}
        self.resources: Dict[str, ResourceDefinition] = {}
        self.prompts: Dict[str, PromptDefinition] = {}

        self._setup_handlers()

        logger.info(f"Initialized {name} v{version} with model {model_name}")

    def _setup_handlers(self):
        """Register MCP protocol handlers for tools, resources, and prompts."""

        @self.server.list_tools()
        async def list_tools() -> List[types.Tool]:
            return [
                types.Tool(
                    name=t.name,
                    description=t.description,
                    inputSchema=t.input_schema,
                )
                for t in self.tools.values()
            ]

        @self.server.call_tool()
        async def call_tool(
            name: str, arguments: Dict[str, Any]
        ) -> List[types.TextContent]:
            if name not in self.tools:
                raise ValueError(f"Unknown tool: {name}")
            logger.info(f"Executing tool: {name}")
            try:
                result = await self.tools[name].handler(arguments)
                return [types.TextContent(type="text", text=str(result))]
            except Exception as e:
                logger.error(f"Tool '{name}' failed: {e}")
                raise

        @self.server.list_resources()
        async def list_resources() -> List[types.Resource]:
            return [
                types.Resource(
                    uri=r.uri,
                    name=r.name,
                    description=r.description,
                    mimeType=r.mime_type,
                )
                for r in self.resources.values()
            ]

        @self.server.read_resource()
        async def read_resource(uri: str) -> str:
            if uri not in self.resources:
                raise ValueError(f"Unknown resource: {uri}")
            logger.info(f"Reading resource: {uri}")
            try:
                return await self.resources[uri].handler()
            except Exception as e:
                logger.error(f"Resource '{uri}' read failed: {e}")
                raise

        @self.server.list_prompts()
        async def list_prompts() -> List[types.Prompt]:
            return [
                types.Prompt(
                    name=p.name,
                    description=p.description,
                    arguments=p.arguments,
                )
                for p in self.prompts.values()
            ]

        @self.server.get_prompt()
        async def get_prompt(
            name: str, arguments: Optional[Dict[str, str]] = None
        ) -> types.GetPromptResult:
            if name not in self.prompts:
                raise ValueError(f"Unknown prompt: {name}")
            logger.info(f"Getting prompt: {name}")
            try:
                messages = await self.prompts[name].handler(arguments or {})
                return types.GetPromptResult(
                    description=self.prompts[name].description,
                    messages=messages,
                )
            except Exception as e:
                logger.error(f"Prompt '{name}' generation failed: {e}")
                raise

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------

    def add_tool(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: Callable,
    ):
        """
        Register a new tool.

        Args:
            name: Tool identifier (lowercase_with_underscores)
            description: Clear description of what the tool does
            input_schema: JSON Schema object for tool parameters
            handler: Async function(args: dict) -> str

        Example:
            self.add_tool(
                name="get_weather",
                description="Get current weather for a location",
                input_schema={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                    },
                    "required": ["location"],
                },
                handler=self._get_weather,
            )
        """
        self.tools[name] = ToolDefinition(name, description, input_schema, handler)
        logger.info(f"Registered tool: {name}")

    def add_resource(
        self,
        uri: str,
        name: str,
        description: str,
        mime_type: str,
        handler: Callable,
    ):
        """
        Register a new resource.

        Args:
            uri: Unique resource identifier (e.g. "file:///data/config.json")
            name: Human-readable name
            description: What this resource contains
            mime_type: Content type (e.g. "application/json", "text/plain")
            handler: Async function() -> str that returns resource content
        """
        self.resources[uri] = ResourceDefinition(
            uri, name, description, mime_type, handler
        )
        logger.info(f"Registered resource: {uri}")

    def add_prompt(
        self,
        name: str,
        description: str,
        arguments: List[Dict[str, Any]],
        handler: Callable,
    ):
        """
        Register a new prompt template.

        Args:
            name: Prompt identifier
            description: What this prompt is for
            arguments: List of argument definitions
            handler: Async function(args: dict) -> list[PromptMessage]
        """
        self.prompts[name] = PromptDefinition(name, description, arguments, handler)
        logger.info(f"Registered prompt: {name}")

    # ------------------------------------------------------------------
    # Lifecycle hooks (override to customize)
    # ------------------------------------------------------------------

    @abstractmethod
    async def register_tools(self):
        """
        Register your custom tools here using self.add_tool(...).

        This method is required — every server should expose at least one tool.
        """
        pass

    async def register_resources(self):
        """
        Register your custom resources here using self.add_resource(...).

        Override this method if your server exposes resources.
        """
        pass

    async def register_prompts(self):
        """
        Register your custom prompt templates here using self.add_prompt(...).

        Override this method if your server exposes reusable prompts.
        """
        pass

    async def initialize(self):
        """
        Initialize the server by registering all capabilities.

        Called automatically before the server starts. Override for
        custom initialization logic (e.g. DB connections, auth setup).
        """
        logger.info("Initializing server...")
        await self.register_tools()
        await self.register_resources()
        await self.register_prompts()
        logger.info(
            f"Server ready — {len(self.tools)} tools, "
            f"{len(self.resources)} resources, {len(self.prompts)} prompts"
        )

    async def run(self):
        """
        Start the MCP server over stdio transport.

        This is the standard MCP communication channel and is compatible
        with Claude Desktop, the MCP CLI, and other MCP clients.
        """
        await self.initialize()
        logger.info(f"Starting {self.name} server...")

        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )


class OllamaIntegrationMixin:
    """
    Mixin that adds Ollama inference to a BaseMCPServer subclass.

    Usage:
        class MyServer(OllamaIntegrationMixin, BaseMCPServer):
            async def register_tools(self):
                self.add_tool("analyze", ..., handler=self._analyze)

            async def _analyze(self, args: dict) -> str:
                return await self.call_ollama(args["text"])

    Requires the host class to have `self.model_name` and `self.ollama_host`
    attributes (both provided by BaseMCPServer).
    """

    async def call_ollama(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str:
        """
        Call Ollama for single-turn text generation.

        Args:
            prompt: User prompt
            model: Model to use (defaults to self.model_name)
            system: Optional system prompt
            temperature: Sampling temperature (0 = deterministic, 1 = creative)

        Returns:
            Model response text
        """
        model = model or self.model_name  # type: ignore[attr-defined]

        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system

        logger.info(f"Calling Ollama model: {model}")

        try:
            async with httpx.AsyncClient(
                timeout=self.ollama_host and 120.0  # type: ignore[attr-defined]
            ) as client:
                response = await client.post(
                    f"{self.ollama_host}/api/generate",  # type: ignore[attr-defined]
                    json=payload,
                )
                response.raise_for_status()
                return response.json().get("response", "")
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            raise

    async def call_ollama_chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str:
        """
        Call Ollama using chat format (supports conversation history).

        Args:
            messages: List of {"role": "user"|"assistant"|"system", "content": "..."}
            model: Model to use (defaults to self.model_name)
            temperature: Sampling temperature

        Returns:
            Model response text
        """
        model = model or self.model_name  # type: ignore[attr-defined]

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }

        logger.info(f"Calling Ollama chat: {model}")

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.ollama_host}/api/chat",  # type: ignore[attr-defined]
                    json=payload,
                )
                response.raise_for_status()
                return response.json().get("message", {}).get("content", "")
        except Exception as e:
            logger.error(f"Ollama chat call failed: {e}")
            raise
