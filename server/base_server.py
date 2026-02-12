"""
Base MCP Server Implementation

This module provides a base class for creating MCP (Model Context Protocol) servers
that integrate with Ollama for AI model inference. The server handles:
- MCP protocol communication
- Tool/resource registration and execution
- Ollama model management
- Request/response handling

Key Concepts:
- MCP Server: Exposes tools and resources that clients can use
- Tools: Functions that the AI can call (like search, calculator, etc.)
- Resources: Data sources the AI can access (like files, databases)
- Prompts: Reusable prompt templates
"""

import asyncio
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """
    Defines a tool that can be exposed via MCP.
    
    Attributes:
        name: Unique identifier for the tool
        description: What the tool does (helps AI understand when to use it)
        input_schema: JSON schema defining expected parameters
        handler: Async function that executes the tool
    """
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable


@dataclass
class ResourceDefinition:
    """
    Defines a resource that can be accessed via MCP.
    
    Resources are data sources like files, databases, or API endpoints
    that the AI can read from.
    """
    uri: str
    name: str
    description: str
    mime_type: str
    handler: Callable


@dataclass
class PromptDefinition:
    """
    Defines a reusable prompt template.
    
    Prompts can include variables that get filled in at runtime.
    """
    name: str
    description: str
    arguments: List[Dict[str, Any]]
    handler: Callable


class BaseMCPServer(ABC):
    """
    Base class for MCP servers with Ollama integration.
    
    This class provides the foundation for building custom MCP servers.
    Subclass this and implement the abstract methods to create your own
    specialized server.
    
    Architecture:
    1. Server initializes and registers capabilities
    2. Client connects and discovers available tools/resources
    3. Client sends requests (tool calls, resource reads, etc.)
    4. Server processes requests, potentially using Ollama for AI inference
    5. Server returns results to client
    
    Example Usage:
        class MyServer(BaseMCPServer):
            def register_tools(self):
                self.add_tool(...)
            
            def register_resources(self):
                self.add_resource(...)
        
        server = MyServer(model_name="llama2")
        await server.run()
    """
    
    def __init__(
        self,
        name: str = "base-mcp-server",
        version: str = "1.0.0",
        model_name: str = "llama3.1:8b",
        ollama_host: str = "http://localhost:11434"
    ):
        """
        Initialize the MCP server.
        
        Args:
            name: Server name (displayed to clients)
            version: Server version
            model_name: Ollama model to use for inference
            ollama_host: Ollama API endpoint
        """
        self.name = name
        self.version = version
        self.model_name = model_name
        self.ollama_host = ollama_host
        
        # MCP server instance
        self.server = Server(name)
        
        # Storage for registered capabilities
        self.tools: Dict[str, ToolDefinition] = {}
        self.resources: Dict[str, ResourceDefinition] = {}
        self.prompts: Dict[str, PromptDefinition] = {}
        
        # Setup handlers
        self._setup_handlers()
        
        logger.info(f"Initialized {name} v{version} with model {model_name}")
    
    def _setup_handlers(self):
        """
        Set up MCP protocol handlers.
        
        These handlers respond to client requests for:
        - Listing available tools
        - Executing tool calls
        - Listing available resources
        - Reading resource content
        - Listing available prompts
        - Getting prompt templates
        """
        
        # Tool handlers
        @self.server.list_tools()
        async def list_tools() -> List[types.Tool]:
            """Return list of all registered tools."""
            return [
                types.Tool(
                    name=tool.name,
                    description=tool.description,
                    inputSchema=tool.input_schema
                )
                for tool in self.tools.values()
            ]
        
        @self.server.call_tool()
        async def call_tool(
            name: str,
            arguments: Dict[str, Any]
        ) -> List[types.TextContent]:
            """Execute a tool and return results."""
            if name not in self.tools:
                raise ValueError(f"Unknown tool: {name}")
            
            tool = self.tools[name]
            logger.info(f"Executing tool: {name} with args: {arguments}")
            
            try:
                result = await tool.handler(arguments)
                return [types.TextContent(
                    type="text",
                    text=str(result)
                )]
            except Exception as e:
                logger.error(f"Tool execution failed: {e}")
                raise
        
        # Resource handlers
        @self.server.list_resources()
        async def list_resources() -> List[types.Resource]:
            """Return list of all registered resources."""
            return [
                types.Resource(
                    uri=resource.uri,
                    name=resource.name,
                    description=resource.description,
                    mimeType=resource.mime_type
                )
                for resource in self.resources.values()
            ]
        
        @self.server.read_resource()
        async def read_resource(uri: str) -> str:
            """Read and return resource content."""
            if uri not in self.resources:
                raise ValueError(f"Unknown resource: {uri}")
            
            resource = self.resources[uri]
            logger.info(f"Reading resource: {uri}")
            
            try:
                content = await resource.handler()
                return content
            except Exception as e:
                logger.error(f"Resource read failed: {e}")
                raise
        
        # Prompt handlers
        @self.server.list_prompts()
        async def list_prompts() -> List[types.Prompt]:
            """Return list of all registered prompts."""
            return [
                types.Prompt(
                    name=prompt.name,
                    description=prompt.description,
                    arguments=prompt.arguments
                )
                for prompt in self.prompts.values()
            ]
        
        @self.server.get_prompt()
        async def get_prompt(
            name: str,
            arguments: Optional[Dict[str, str]] = None
        ) -> types.GetPromptResult:
            """Get a prompt template with variables filled in."""
            if name not in self.prompts:
                raise ValueError(f"Unknown prompt: {name}")
            
            prompt = self.prompts[name]
            logger.info(f"Getting prompt: {name}")
            
            try:
                messages = await prompt.handler(arguments or {})
                return types.GetPromptResult(
                    description=prompt.description,
                    messages=messages
                )
            except Exception as e:
                logger.error(f"Prompt generation failed: {e}")
                raise
    
    def add_tool(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: Callable
    ):
        """
        Register a new tool.
        
        Args:
            name: Tool identifier (use lowercase with underscores)
            description: Clear description of what the tool does
            input_schema: JSON schema for tool parameters
            handler: Async function that implements the tool
        
        Example:
            server.add_tool(
                name="get_weather",
                description="Get weather for a location",
                input_schema={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                },
                handler=self.get_weather_handler
            )
        """
        self.tools[name] = ToolDefinition(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=handler
        )
        logger.info(f"Registered tool: {name}")
    
    def add_resource(
        self,
        uri: str,
        name: str,
        description: str,
        mime_type: str,
        handler: Callable
    ):
        """
        Register a new resource.
        
        Args:
            uri: Unique resource identifier (e.g., "file:///data/config.json")
            name: Human-readable name
            description: What this resource contains
            mime_type: Content type (e.g., "application/json", "text/plain")
            handler: Async function that returns resource content
        """
        self.resources[uri] = ResourceDefinition(
            uri=uri,
            name=name,
            description=description,
            mime_type=mime_type,
            handler=handler
        )
        logger.info(f"Registered resource: {uri}")
    
    def add_prompt(
        self,
        name: str,
        description: str,
        arguments: List[Dict[str, Any]],
        handler: Callable
    ):
        """
        Register a new prompt template.
        
        Args:
            name: Prompt identifier
            description: What this prompt is for
            arguments: List of argument definitions
            handler: Async function that generates prompt messages
        """
        self.prompts[name] = PromptDefinition(
            name=name,
            description=description,
            arguments=arguments,
            handler=handler
        )
        logger.info(f"Registered prompt: {name}")
    
    @abstractmethod
    async def register_tools(self):
        """
        Override this to register your custom tools.
        
        Example:
            async def register_tools(self):
                self.add_tool(
                    name="search",
                    description="Search the web",
                    input_schema={...},
                    handler=self.search_handler
                )
        """
        pass
    
    @abstractmethod
    async def register_resources(self):
        """
        Override this to register your custom resources.
        
        Example:
            async def register_resources(self):
                self.add_resource(
                    uri="file:///config.json",
                    name="Configuration",
                    description="Server configuration",
                    mime_type="application/json",
                    handler=self.config_handler
                )
        """
        pass
    
    async def register_prompts(self):
        """
        Override this to register your custom prompts.
        
        This is optional - only implement if you want prompt templates.
        """
        pass
    
    async def initialize(self):
        """
        Initialize the server by registering all capabilities.
        
        This is called automatically before the server starts.
        You can override this if you need custom initialization logic.
        """
        logger.info("Initializing server...")
        await self.register_tools()
        await self.register_resources()
        await self.register_prompts()
        logger.info(
            f"Server initialized with {len(self.tools)} tools, "
            f"{len(self.resources)} resources, {len(self.prompts)} prompts"
        )
    
    async def run(self):
        """
        Start the MCP server.
        
        This runs the server using stdio (standard input/output) transport,
        which is the standard way MCP servers communicate with clients.
        """
        await self.initialize()
        
        logger.info(f"Starting {self.name} server...")
        
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


class OllamaIntegrationMixin:
    """
    Mixin class that adds Ollama integration capabilities.
    
    This provides methods for calling Ollama models for inference,
    which is useful when your tools need AI capabilities.
    
    Usage:
        class MyServer(BaseMCPServer, OllamaIntegrationMixin):
            async def my_tool(self, args):
                response = await self.call_ollama("Analyze this text")
                return response
    """
    
    async def call_ollama(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.7,
        stream: bool = False
    ) -> str:
        """
        Call Ollama for AI inference.
        
        Args:
            prompt: The user prompt/question
            model: Model to use (defaults to self.model_name)
            system: System prompt to set context/behavior
            temperature: Randomness (0.0 = deterministic, 1.0 = creative)
            stream: Whether to stream the response
        
        Returns:
            Model's response text
        """
        import httpx
        
        model = model or self.model_name
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature
            }
        }
        
        if system:
            payload["system"] = system
        
        logger.info(f"Calling Ollama model: {model}")
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.ollama_host}/api/generate",
                    json=payload
                )
                response.raise_for_status()
                result = response.json()
                return result.get("response", "")
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            raise
    
    async def call_ollama_chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7
    ) -> str:
        """
        Call Ollama using chat format (for conversation history).
        
        Args:
            messages: List of message dicts with 'role' and 'content'
                      Example: [
                          {"role": "system", "content": "You are helpful"},
                          {"role": "user", "content": "Hello!"}
                      ]
            model: Model to use
            temperature: Randomness level
        
        Returns:
            Model's response text
        """
        import httpx
        
        model = model or self.model_name
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        
        logger.info(f"Calling Ollama chat model: {model}")
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.ollama_host}/api/chat",
                    json=payload
                )
                response.raise_for_status()
                result = response.json()
                return result.get("message", {}).get("content", "")
        except Exception as e:
            logger.error(f"Ollama chat call failed: {e}")
            raise