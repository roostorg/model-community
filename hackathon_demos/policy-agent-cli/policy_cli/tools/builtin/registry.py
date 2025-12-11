"""Registry for built-in tools."""

from __future__ import annotations

from typing import List

from policy_cli.tools.registry import ToolRegistry
from policy_cli.tools.builtin.file_tools import (
    ReadFileTool,
    WriteFileTool,
    ListDirectoryTool,
    GlobTool,
    EditFileTool,
)
from policy_cli.tools.builtin.shell_tools import (
    ShellTool,
    GrepTool,
)
from policy_cli.tools.builtin.web_tools import (
    WebFetchTool,
    WebSearchTool,
    UrlAnalyzerTool,
)
from policy_cli.tools.builtin.policy_tools import (
    RunPolicyTool,
    ModifyPolicyTool,
)


def register_builtin_tools(registry: ToolRegistry, include_tools: List[str] = None) -> None:
    """Register all built-in tools with the registry."""

    # Define all available built-in tools
    builtin_tools = {
        "read_file": ReadFileTool,
        "write_file": WriteFileTool,
        "list_directory": ListDirectoryTool,
        "glob_files": GlobTool,
        "edit_file": EditFileTool,
        "shell": ShellTool,
        "grep": GrepTool,
        "web_fetch": WebFetchTool,
        "web_search": WebSearchTool,
        "analyze_url": UrlAnalyzerTool,
        "run_policy": RunPolicyTool,
        "modify_policy": ModifyPolicyTool,
    }

    # Register requested tools (or all if none specified)
    tools_to_register = include_tools if include_tools else list(builtin_tools.keys())

    for tool_name in tools_to_register:
        if tool_name in builtin_tools:
            tool_class = builtin_tools[tool_name]
            tool_instance = tool_class()
            registry.register_tool(
                tool=tool_instance,
                source="builtin",
                trusted=True
            )
        else:
            print(f"Warning: Unknown built-in tool '{tool_name}'")


def get_builtin_tool_names() -> List[str]:
    """Get list of all available built-in tool names."""
    return [
        "read_file",
        "write_file",
        "list_directory",
        "glob_files",
        "edit_file",
        "shell",
        "grep",
        "web_fetch",
        "web_search",
        "analyze_url",
        "run_policy",
        "modify_policy",
    ]


def get_default_tools() -> List[str]:
    """Get list of default tools to enable."""
    return [
        "read_file",
        "write_file",
        "list_directory",
        "glob_files",
        "edit_file",
        "shell",
        "grep",
        "web_fetch",
        "run_policy",
        "modify_policy",
    ]


def get_safe_tools() -> List[str]:
    """Get list of tools that are safe to run without confirmation."""
    return [
        "read_file",
        "list_directory",
        "glob_files",
        "grep",
        "web_fetch",
        "analyze_url",
        "run_policy",
        "modify_policy",
    ]