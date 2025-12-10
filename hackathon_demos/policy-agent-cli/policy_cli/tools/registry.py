"""Tool registry for managing available tools."""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Any
import logging
from dataclasses import dataclass

from policy_cli.core.types import Tool, ToolError
from policy_cli.tools.base import BaseTool

logger = logging.getLogger(__name__)


@dataclass
class ToolRegistration:
    """Information about a registered tool."""

    tool: Tool
    source: str  # "builtin", "mcp", "discovered"
    server_name: Optional[str] = None  # For MCP tools
    trusted: bool = True


class ToolRegistry:
    """Registry for managing all available tools."""

    def __init__(self):
        self._tools: Dict[str, ToolRegistration] = {}
        self._tool_groups: Dict[str, Set[str]] = {
            "builtin": set(),
            "mcp": set(),
            "discovered": set(),
        }

    def register_tool(
        self,
        tool: Tool,
        source: str = "builtin",
        server_name: Optional[str] = None,
        trusted: bool = True,
        replace_existing: bool = False
    ) -> None:
        """Register a tool in the registry."""
        if tool.name in self._tools and not replace_existing:
            existing = self._tools[tool.name]
            if source == "mcp" and existing.source != "mcp":
                # MCP tools can have conflicts, use fully qualified names
                qualified_name = f"{server_name}.{tool.name}" if server_name else tool.name
                self._register_with_name(qualified_name, tool, source, server_name, trusted)
            else:
                logger.warning(f"Tool '{tool.name}' is already registered. Skipping.")
            return

        self._register_with_name(tool.name, tool, source, server_name, trusted)

    def _register_with_name(
        self,
        name: str,
        tool: Tool,
        source: str,
        server_name: Optional[str],
        trusted: bool
    ) -> None:
        """Register tool with specific name."""
        registration = ToolRegistration(
            tool=tool,
            source=source,
            server_name=server_name,
            trusted=trusted
        )

        self._tools[name] = registration
        self._tool_groups[source].add(name)

        logger.debug(f"Registered tool '{name}' from source '{source}'")

    def unregister_tool(self, name: str) -> bool:
        """Unregister a tool by name."""
        if name not in self._tools:
            return False

        registration = self._tools[name]
        del self._tools[name]
        self._tool_groups[registration.source].discard(name)

        logger.debug(f"Unregistered tool '{name}'")
        return True

    def unregister_mcp_server_tools(self, server_name: str) -> int:
        """Unregister all tools from a specific MCP server."""
        tools_to_remove = [
            name for name, reg in self._tools.items()
            if reg.source == "mcp" and reg.server_name == server_name
        ]

        for name in tools_to_remove:
            self.unregister_tool(name)

        logger.debug(f"Unregistered {len(tools_to_remove)} tools from server '{server_name}'")
        return len(tools_to_remove)

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        registration = self._tools.get(name)
        return registration.tool if registration else None

    def get_tool_registration(self, name: str) -> Optional[ToolRegistration]:
        """Get full tool registration by name."""
        return self._tools.get(name)

    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def list_tools(self, source: Optional[str] = None, trusted_only: bool = False) -> List[str]:
        """List all tool names, optionally filtered by source or trust status."""
        tools = []

        for name, registration in self._tools.items():
            if source and registration.source != source:
                continue
            if trusted_only and not registration.trusted:
                continue
            tools.append(name)

        return sorted(tools)

    def get_tools_by_source(self, source: str) -> Dict[str, Tool]:
        """Get all tools from a specific source."""
        return {
            name: reg.tool
            for name, reg in self._tools.items()
            if reg.source == source
        }

    def get_function_declarations(
        self,
        include_sources: Optional[List[str]] = None,
        trusted_only: bool = True
    ) -> List[Dict[str, Any]]:
        """Get function declarations for AI model."""
        declarations = []

        for name, registration in self._tools.items():
            if include_sources and registration.source not in include_sources:
                continue
            if trusted_only and not registration.trusted:
                continue

            if hasattr(registration.tool, 'to_function_declaration'):
                declarations.append(registration.tool.to_function_declaration())
            else:
                # Fallback for tools that don't implement the method
                declarations.append({
                    "name": registration.tool.name,
                    "description": registration.tool.description,
                    "parameters": registration.tool.parameters_schema
                })

        return declarations

    def clear_discovered_tools(self) -> None:
        """Remove all discovered tools (for refresh)."""
        discovered_tools = list(self._tool_groups["discovered"])
        for tool_name in discovered_tools:
            self.unregister_tool(tool_name)

    def clear_mcp_tools(self) -> None:
        """Remove all MCP tools (for refresh)."""
        mcp_tools = list(self._tool_groups["mcp"])
        for tool_name in mcp_tools:
            self.unregister_tool(tool_name)

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            "total_tools": len(self._tools),
            "by_source": {
                source: len(tools)
                for source, tools in self._tool_groups.items()
            },
            "trusted_tools": len([
                reg for reg in self._tools.values() if reg.trusted
            ]),
            "untrusted_tools": len([
                reg for reg in self._tools.values() if not reg.trusted
            ])
        }

    def __len__(self) -> int:
        """Return number of registered tools."""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """Check if tool name is in registry."""
        return name in self._tools

    def __iter__(self):
        """Iterate over tool names."""
        return iter(self._tools.keys())