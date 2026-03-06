"""MCP client for connecting to MCP servers."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
import subprocess

from policy_cli.core.types import Tool, ToolResult, SimpleToolResult, ToolError
from policy_cli.tools.base import BaseTool

logger = logging.getLogger(__name__)


class MCPServerStatus(Enum):
    """Status of MCP server connection."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""

    name: str
    command: List[str]
    env: Optional[Dict[str, str]] = None
    timeout: int = 30
    trusted: bool = False


@dataclass
class MCPTool:
    """MCP tool definition."""

    name: str
    description: str
    input_schema: Dict[str, Any]
    server_name: str


class MCPToolWrapper(BaseTool):
    """Wrapper for MCP tools to integrate with tool registry."""

    def __init__(self, mcp_tool: MCPTool, client: 'MCPClient'):
        self.mcp_tool = mcp_tool
        self.client = client
        super().__init__(
            name=f"{mcp_tool.server_name}.{mcp_tool.name}",
            description=f"[{mcp_tool.server_name}] {mcp_tool.description}"
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return self.mcp_tool.input_schema

    async def execute(
        self,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        try:
            result = await self.client.call_tool(
                self.mcp_tool.server_name,
                self.mcp_tool.name,
                parameters
            )
            return SimpleToolResult(content=result)
        except Exception as e:
            return SimpleToolResult(
                content=f"MCP tool execution failed: {str(e)}",
                error=str(e)
            )


class MCPClient:
    """Client for managing MCP server connections."""

    def __init__(self):
        self.servers: Dict[str, MCPServerConfig] = {}
        self.processes: Dict[str, subprocess.Popen] = {}
        self.status: Dict[str, MCPServerStatus] = {}
        self.tools: Dict[str, List[MCPTool]] = {}

    async def add_server(self, config: MCPServerConfig) -> None:
        """Add an MCP server configuration."""
        self.servers[config.name] = config
        self.status[config.name] = MCPServerStatus.DISCONNECTED

    async def connect_server(self, server_name: str) -> bool:
        """Connect to an MCP server."""
        if server_name not in self.servers:
            logger.error(f"Server {server_name} not configured")
            return False

        config = self.servers[server_name]
        self.status[server_name] = MCPServerStatus.CONNECTING

        try:
            # Start the MCP server process
            process = await asyncio.create_subprocess_exec(
                *config.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=config.env
            )

            self.processes[server_name] = process

            # Initialize MCP connection
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {},
                        "prompts": {}
                    },
                    "clientInfo": {
                        "name": "policy-cli",
                        "version": "0.1.0"
                    }
                }
            }

            # Send initialization request
            await self._send_request(server_name, init_request)

            # Wait for response
            response = await self._read_response(server_name)

            if response and response.get("result"):
                # Successfully initialized, now discover tools
                await self._discover_tools(server_name)
                self.status[server_name] = MCPServerStatus.CONNECTED
                logger.info(f"Connected to MCP server: {server_name}")
                return True
            else:
                logger.error(f"Failed to initialize MCP server: {server_name}")
                self.status[server_name] = MCPServerStatus.ERROR
                return False

        except Exception as e:
            logger.error(f"Error connecting to MCP server {server_name}: {e}")
            self.status[server_name] = MCPServerStatus.ERROR
            return False

    async def disconnect_server(self, server_name: str) -> None:
        """Disconnect from an MCP server."""
        if server_name in self.processes:
            process = self.processes[server_name]
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()

            del self.processes[server_name]

        self.status[server_name] = MCPServerStatus.DISCONNECTED
        if server_name in self.tools:
            del self.tools[server_name]

        logger.info(f"Disconnected from MCP server: {server_name}")

    async def _discover_tools(self, server_name: str) -> None:
        """Discover tools available on an MCP server."""
        tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }

        await self._send_request(server_name, tools_request)
        response = await self._read_response(server_name)

        if response and response.get("result") and response["result"].get("tools"):
            tools = []
            for tool_def in response["result"]["tools"]:
                mcp_tool = MCPTool(
                    name=tool_def["name"],
                    description=tool_def.get("description", ""),
                    input_schema=tool_def.get("inputSchema", {}),
                    server_name=server_name
                )
                tools.append(mcp_tool)

            self.tools[server_name] = tools
            logger.info(f"Discovered {len(tools)} tools from {server_name}")

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        parameters: Dict[str, Any]
    ) -> str:
        """Call a tool on an MCP server."""
        if server_name not in self.processes:
            raise ToolError(f"Server {server_name} not connected", tool_name)

        # Prepare tool call request
        tool_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": parameters
            }
        }

        await self._send_request(server_name, tool_request)
        response = await self._read_response(server_name)

        if response and response.get("result"):
            result = response["result"]
            if result.get("isError"):
                raise ToolError(
                    result.get("content", [{}])[0].get("text", "Unknown error"),
                    tool_name
                )
            else:
                # Extract content from result
                content = result.get("content", [])
                if content and isinstance(content, list):
                    return content[0].get("text", "")
                return str(result)
        elif response and response.get("error"):
            error = response["error"]
            raise ToolError(
                f"MCP error: {error.get('message', 'Unknown error')}",
                tool_name
            )
        else:
            raise ToolError("No response from MCP server", tool_name)

    async def _send_request(self, server_name: str, request: Dict[str, Any]) -> None:
        """Send a JSON-RPC request to an MCP server."""
        if server_name not in self.processes:
            raise Exception(f"Server {server_name} not connected")

        process = self.processes[server_name]
        request_json = json.dumps(request) + "\n"

        if process.stdin:
            process.stdin.write(request_json.encode())
            await process.stdin.drain()

    async def _read_response(self, server_name: str) -> Optional[Dict[str, Any]]:
        """Read a JSON-RPC response from an MCP server."""
        if server_name not in self.processes:
            return None

        process = self.processes[server_name]

        try:
            if process.stdout:
                line = await process.stdout.readline()
                if line:
                    return json.loads(line.decode().strip())
        except (json.JSONDecodeError, asyncio.TimeoutError) as e:
            logger.error(f"Error reading response from {server_name}: {e}")

        return None

    def get_server_tools(self, server_name: str) -> List[MCPTool]:
        """Get tools available on a specific server."""
        return self.tools.get(server_name, [])

    def get_all_tools(self) -> List[MCPTool]:
        """Get all tools from all connected servers."""
        all_tools = []
        for tools in self.tools.values():
            all_tools.extend(tools)
        return all_tools

    def get_server_status(self, server_name: str) -> MCPServerStatus:
        """Get the status of a specific server."""
        return self.status.get(server_name, MCPServerStatus.DISCONNECTED)

    async def shutdown(self) -> None:
        """Shutdown all MCP server connections."""
        for server_name in list(self.processes.keys()):
            await self.disconnect_server(server_name)


class MCPManager:
    """High-level manager for MCP integration."""

    def __init__(self):
        self.client = MCPClient()

    async def load_servers_from_config(self, config: Dict[str, Any]) -> None:
        """Load MCP servers from configuration."""
        mcp_servers = config.get("mcp_servers", {})

        for server_name, server_config in mcp_servers.items():
            mcp_config = MCPServerConfig(
                name=server_name,
                command=server_config["command"],
                env=server_config.get("env"),
                timeout=server_config.get("timeout", 30),
                trusted=server_config.get("trusted", False)
            )

            await self.client.add_server(mcp_config)

    async def connect_all_servers(self) -> Dict[str, bool]:
        """Connect to all configured MCP servers."""
        results = {}

        for server_name in self.client.servers:
            success = await self.client.connect_server(server_name)
            results[server_name] = success

        return results

    def get_tool_wrappers(self) -> List[MCPToolWrapper]:
        """Get tool wrappers for all connected MCP tools."""
        wrappers = []

        for tools in self.client.tools.values():
            for mcp_tool in tools:
                wrapper = MCPToolWrapper(mcp_tool, self.client)
                wrappers.append(wrapper)

        return wrappers

    async def shutdown(self) -> None:
        """Shutdown MCP manager."""
        await self.client.shutdown()