"""
Base MCP Client Implementation

This module provides a base class for creating MCP clients that connect to
MCP servers. The client can:
- Discover available tools, resources, and prompts
- Call tools on the server
- Read resources from the server
- Use prompt templates

Key Concepts:
- Client connects to server via stdio (standard input/output)
- Client discovers capabilities by listing tools/resources/prompts
- Client sends requests and receives responses
- Client can be used in applications or other automation workflows
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import mcp
from mcp.client.stdio import stdio_client, StdioServerParameters

logging.basicConfig(level=logging.INFO)
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
    arguments: List[Dict[str, Any]]


class BaseMCPClient:
    """
    Base class for MCP clients.
    
    This class handles connection to MCP servers and provides methods
    to interact with server capabilities.
    
    Architecture:
    1. Client spawns server process (or connects to running server)
    2. Client performs capability discovery
    3. Client can call tools, read resources, use prompts
    4. Client manages connection lifecycle
    
    Example Usage:
        client = BaseMCPClient(
            server_script_path="./my_server.py"
        )
        
        async with client:
            # Discover capabilities
            await client.discover()
            
            # Use a tool
            result = await client.call_tool(
                "search",
                {"query": "MCP protocol"}
            )
            
            # Read a resource
            content = await client.read_resource("file:///data.json")
    """
    
    def __init__(
        self,
        server_script_path: str,
        server_args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the MCP client.
        
        Args:
            server_script_path: Path to the server script to run
            server_args: Additional arguments to pass to server
            env: Environment variables for the server process
        """
        self.server_script_path = server_script_path
        self.server_args = server_args or []
        self.env = env or {}
        
        # These will be set when connected
        self.client = None
        self.session = None
        self.read_stream = None
        self.write_stream = None
        
        # Cached capability information
        self.tools: Dict[str, ToolInfo] = {}
        self.resources: Dict[str, ResourceInfo] = {}
        self.prompts: Dict[str, PromptInfo] = {}
        
        logger.info(f"Initialized client for server: {server_script_path}")
    
    async def __aenter__(self):
        """Context manager entry - connect to server."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - disconnect from server."""
        await self.disconnect()
    
    async def connect(self):
        """
        Connect to the MCP server.
        
        This spawns the server process and establishes communication.
        """
        logger.info("Connecting to server...")
        
        server_params = StdioServerParameters(
            command="python",
            args=[self.server_script_path] + self.server_args,
            env=self.env
        )
        
        # Create stdio client connection
        self.read_stream, self.write_stream = await stdio_client(server_params)
        
        # Create MCP client instance
        self.client = mcp.ClientSession(self.read_stream, self.write_stream)
        
        # Initialize the session
        await self.client.__aenter__()
        
        logger.info("Connected to server")
    
    async def disconnect(self):
        """Disconnect from the MCP server."""
        if self.client:
            await self.client.__aexit__(None, None, None)
            logger.info("Disconnected from server")
    
    async def discover(self):
        """
        Discover all capabilities available on the server.
        
        This populates the tools, resources, and prompts caches.
        Call this after connecting to learn what the server offers.
        """
        logger.info("Discovering server capabilities...")
        
        # Discover tools
        tools_result = await self.client.list_tools()
        self.tools = {
            tool.name: ToolInfo(
                name=tool.name,
                description=tool.description,
                input_schema=tool.inputSchema
            )
            for tool in tools_result.tools
        }
        logger.info(f"Discovered {len(self.tools)} tools")
        
        # Discover resources
        resources_result = await self.client.list_resources()
        self.resources = {
            resource.uri: ResourceInfo(
                uri=resource.uri,
                name=resource.name,
                description=resource.description,
                mime_type=resource.mimeType
            )
            for resource in resources_result.resources
        }
        logger.info(f"Discovered {len(self.resources)} resources")
        
        # Discover prompts
        prompts_result = await self.client.list_prompts()
        self.prompts = {
            prompt.name: PromptInfo(
                name=prompt.name,
                description=prompt.description,
                arguments=prompt.arguments if hasattr(prompt, 'arguments') else []
            )
            for prompt in prompts_result.prompts
        }
        logger.info(f"Discovered {len(self.prompts)} prompts")
    
    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Any:
        """
        Call a tool on the server.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool
        
        Returns:
            Tool execution result
        
        Raises:
            ValueError: If tool doesn't exist
        """
        if tool_name not in self.tools:
            available = ", ".join(self.tools.keys())
            raise ValueError(
                f"Tool '{tool_name}' not found. "
                f"Available tools: {available}"
            )
        
        logger.info(f"Calling tool: {tool_name}")
        
        result = await self.client.call_tool(tool_name, arguments)
        
        # Extract text content from result
        if result.content:
            return result.content[0].text
        return None
    
    async def read_resource(self, uri: str) -> str:
        """
        Read a resource from the server.
        
        Args:
            uri: URI of the resource to read
        
        Returns:
            Resource content
        
        Raises:
            ValueError: If resource doesn't exist
        """
        if uri not in self.resources:
            available = ", ".join(self.resources.keys())
            raise ValueError(
                f"Resource '{uri}' not found. "
                f"Available resources: {available}"
            )
        
        logger.info(f"Reading resource: {uri}")
        
        result = await self.client.read_resource(uri)
        
        # Extract content from result
        if result.contents:
            return result.contents[0].text
        return None
    
    async def get_prompt(
        self,
        prompt_name: str,
        arguments: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, str]]:
        """
        Get a prompt template from the server.
        
        Args:
            prompt_name: Name of the prompt
            arguments: Values for prompt variables
        
        Returns:
            List of message dicts with 'role' and 'content'
        
        Raises:
            ValueError: If prompt doesn't exist
        """
        if prompt_name not in self.prompts:
            available = ", ".join(self.prompts.keys())
            raise ValueError(
                f"Prompt '{prompt_name}' not found. "
                f"Available prompts: {available}"
            )
        
        logger.info(f"Getting prompt: {prompt_name}")
        
        result = await self.client.get_prompt(
            prompt_name,
            arguments=arguments or {}
        )
        
        # Convert to standard message format
        messages = []
        for msg in result.messages:
            messages.append({
                "role": msg.role,
                "content": msg.content.text if hasattr(msg.content, 'text') else str(msg.content)
            })
        
        return messages
    
    def list_tools(self) -> List[ToolInfo]:
        """Get list of available tools (must call discover() first)."""
        return list(self.tools.values())
    
    def list_resources(self) -> List[ResourceInfo]:
        """Get list of available resources (must call discover() first)."""
        return list(self.resources.values())
    
    def list_prompts(self) -> List[PromptInfo]:
        """Get list of available prompts (must call discover() first)."""
        return list(self.prompts.values())
    
    def get_tool_info(self, tool_name: str) -> Optional[ToolInfo]:
        """Get information about a specific tool."""
        return self.tools.get(tool_name)
    
    def get_resource_info(self, uri: str) -> Optional[ResourceInfo]:
        """Get information about a specific resource."""
        return self.resources.get(uri)
    
    def get_prompt_info(self, prompt_name: str) -> Optional[PromptInfo]:
        """Get information about a specific prompt."""
        return self.prompts.get(prompt_name)