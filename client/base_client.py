"""
Base MCP Client Implementation

Provides a base class for connecting to MCP servers, discovering their
capabilities, and calling tools/resources/prompts.

Key Concepts:
- Client spawns a server process (or connects to one) via stdio
- Client discovers capabilities with discover()
- Client calls tools, reads resources, and uses prompts
- Use as an async context manager for automatic connection lifecycle

Example Usage:
    async with BaseMCPClient(server_script_path="./server/base_server.py") as client:
        await client.discover()

        result = await client.call_tool("echo", {"text": "hello"})
        content = await client.read_resource("file:///data.json")
        messages = await client.get_prompt("my_prompt", {"var": "value"})
"""

import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import mcp
from mcp.client.stdio import StdioServerParameters, stdio_client

logger = logging.getLogger(__name__)


@dataclass
class ToolInfo:
    """Information about an available tool."""

    name: str
    description: str
    input_schema: Dict[str, Any]


@dataclass
class ResourceInfo:
    """Information about an available resource."""

    uri: str
    name: str
    description: str
    mime_type: str


@dataclass
class PromptInfo:
    """Information about an available prompt."""

    name: str
    description: str
    arguments: List[Dict[str, Any]] = field(default_factory=list)


class BaseMCPClient:
    """
    Base class for MCP clients.

    Handles the full connection lifecycle to an MCP server via stdio and
    provides typed methods to interact with server capabilities.

    Use as a context manager:
        async with BaseMCPClient("./my_server.py") as client:
            await client.discover()
            result = await client.call_tool("search", {"query": "hello"})

    Or manage the lifecycle manually:
        client = BaseMCPClient("./my_server.py")
        await client.connect()
        await client.discover()
        ...
        await client.disconnect()
    """

    def __init__(
        self,
        server_script_path: str,
        server_args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
    ):
        """
        Args:
            server_script_path: Path to the Python server script to spawn
            server_args: Additional CLI arguments passed to the server script
            env: Extra environment variables for the server process.
                 Pass None (default) to inherit the current environment.
        """
        self.server_script_path = server_script_path
        self.server_args = server_args or []
        self.env = env  # None = inherit parent env; {} = empty env (usually wrong)

        self.session: Optional[mcp.ClientSession] = None
        self._exit_stack: Optional[AsyncExitStack] = None

        # Populated after discover()
        self.tools: Dict[str, ToolInfo] = {}
        self.resources: Dict[str, ResourceInfo] = {}
        self.prompts: Dict[str, PromptInfo] = {}

        logger.info(f"Initialized client for server: {server_script_path}")

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()

    async def connect(self):
        """
        Spawn the server process and establish the MCP session.

        Uses AsyncExitStack so both the stdio transport and the MCP session
        are cleaned up properly on disconnect().
        """
        logger.info("Connecting to server...")

        server_params = StdioServerParameters(
            command="python",
            args=[self.server_script_path] + self.server_args,
            env=self.env,
        )

        self._exit_stack = AsyncExitStack()

        read_stream, write_stream = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.session = await self._exit_stack.enter_async_context(
            mcp.ClientSession(read_stream, write_stream)
        )
        await self.session.initialize()

        logger.info("Connected to server")

    async def disconnect(self):
        """Close the MCP session and stop the server process."""
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self.session = None
            logger.info("Disconnected from server")

    async def discover(self):
        """
        Discover all capabilities available on the server.

        Populates self.tools, self.resources, and self.prompts.
        Call this after connect() to learn what the server exposes.
        """
        if not self.session:
            raise RuntimeError("Not connected. Call connect() first.")

        logger.info("Discovering server capabilities...")

        tools_result = await self.session.list_tools()
        self.tools = {
            t.name: ToolInfo(
                name=t.name,
                description=t.description,
                input_schema=t.inputSchema,
            )
            for t in tools_result.tools
        }
        logger.info(f"Discovered {len(self.tools)} tools")

        resources_result = await self.session.list_resources()
        self.resources = {
            r.uri: ResourceInfo(
                uri=r.uri,
                name=r.name,
                description=r.description,
                mime_type=r.mimeType,
            )
            for r in resources_result.resources
        }
        logger.info(f"Discovered {len(self.resources)} resources")

        prompts_result = await self.session.list_prompts()
        self.prompts = {
            p.name: PromptInfo(
                name=p.name,
                description=p.description,
                arguments=p.arguments if hasattr(p, "arguments") else [],
            )
            for p in prompts_result.prompts
        }
        logger.info(f"Discovered {len(self.prompts)} prompts")

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a tool on the server.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments matching the tool's input schema

        Returns:
            Tool execution result (text content)

        Raises:
            RuntimeError: If not connected
            ValueError: If tool_name is not in discovered tools
        """
        if not self.session:
            raise RuntimeError("Not connected. Call connect() first.")
        if tool_name not in self.tools:
            raise ValueError(
                f"Tool '{tool_name}' not found. "
                f"Available: {', '.join(self.tools)}"
            )

        logger.info(f"Calling tool: {tool_name}")
        result = await self.session.call_tool(tool_name, arguments)
        return result.content[0].text if result.content else None

    async def read_resource(self, uri: str) -> Optional[str]:
        """
        Read a resource from the server.

        Args:
            uri: URI of the resource to read

        Returns:
            Resource content as a string

        Raises:
            RuntimeError: If not connected
            ValueError: If uri is not in discovered resources
        """
        if not self.session:
            raise RuntimeError("Not connected. Call connect() first.")
        if uri not in self.resources:
            raise ValueError(
                f"Resource '{uri}' not found. "
                f"Available: {', '.join(self.resources)}"
            )

        logger.info(f"Reading resource: {uri}")
        result = await self.session.read_resource(uri)
        return result.contents[0].text if result.contents else None

    async def get_prompt(
        self,
        prompt_name: str,
        arguments: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, str]]:
        """
        Get a prompt template from the server with variables filled in.

        Args:
            prompt_name: Name of the prompt
            arguments: Values for prompt variables

        Returns:
            List of {"role": str, "content": str} message dicts

        Raises:
            RuntimeError: If not connected
            ValueError: If prompt_name is not in discovered prompts
        """
        if not self.session:
            raise RuntimeError("Not connected. Call connect() first.")
        if prompt_name not in self.prompts:
            raise ValueError(
                f"Prompt '{prompt_name}' not found. "
                f"Available: {', '.join(self.prompts)}"
            )

        logger.info(f"Getting prompt: {prompt_name}")
        result = await self.session.get_prompt(prompt_name, arguments=arguments or {})

        return [
            {
                "role": msg.role,
                "content": (
                    msg.content.text
                    if hasattr(msg.content, "text")
                    else str(msg.content)
                ),
            }
            for msg in result.messages
        ]

    # ------------------------------------------------------------------
    # Convenience accessors (require discover() to have been called)
    # ------------------------------------------------------------------

    def list_tools(self) -> List[ToolInfo]:
        return list(self.tools.values())

    def list_resources(self) -> List[ResourceInfo]:
        return list(self.resources.values())

    def list_prompts(self) -> List[PromptInfo]:
        return list(self.prompts.values())

    def get_tool_info(self, tool_name: str) -> Optional[ToolInfo]:
        return self.tools.get(tool_name)

    def get_resource_info(self, uri: str) -> Optional[ResourceInfo]:
        return self.resources.get(uri)

    def get_prompt_info(self, prompt_name: str) -> Optional[PromptInfo]:
        return self.prompts.get(prompt_name)
