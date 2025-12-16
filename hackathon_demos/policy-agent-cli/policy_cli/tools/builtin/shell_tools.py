"""Shell execution tools."""

from __future__ import annotations

from typing import Dict, Any, Optional
import logging
import asyncio

from policy_cli.tools.base import BaseTool, ShellToolMixin
from policy_cli.core.types import ToolResult, SimpleToolResult

logger = logging.getLogger(__name__)


class ShellTool(BaseTool, ShellToolMixin):
    """Tool for executing shell commands."""

    def __init__(self):
        super().__init__(
            name="shell",
            description="Execute shell commands and return the output. Use with caution."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 30)",
                    "default": 30
                },
                "working_directory": {
                    "type": "string",
                    "description": "Working directory for command execution (optional)"
                }
            },
            "required": ["command"]
        }

    async def execute(
        self,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        command = parameters["command"]
        timeout = parameters.get("timeout", 30)
        working_directory = parameters.get("working_directory")

        try:
            # Change to working directory if specified
            old_cwd = None
            if working_directory:
                import os
                old_cwd = os.getcwd()
                abs_wd = os.path.abspath(working_directory)
                if not os.path.exists(abs_wd):
                    return SimpleToolResult(
                        content=f"Working directory not found: {working_directory}",
                        error="Directory not found"
                    )
                os.chdir(abs_wd)

            try:
                stdout, stderr, returncode = await self._execute_shell_command(
                    command, timeout=timeout
                )

                # Format output
                output_parts = []
                if stdout.strip():
                    output_parts.append(f"STDOUT:\n{stdout}")
                if stderr.strip():
                    output_parts.append(f"STDERR:\n{stderr}")

                output = "\n\n".join(output_parts) if output_parts else "(No output)"

                if returncode != 0:
                    content = f"Command failed with exit code {returncode}\n\n{output}"
                    return SimpleToolResult(
                        content=content,
                        display_content=f"Command failed: {command}",
                        error=f"Exit code {returncode}"
                    )
                else:
                    content = f"Command executed successfully\n\n{output}"
                    return SimpleToolResult(
                        content=content,
                        display_content=f"Executed: {command}"
                    )

            finally:
                # Restore original working directory
                if old_cwd:
                    import os
                    os.chdir(old_cwd)

        except Exception as e:
            return SimpleToolResult(
                content=f"Shell command execution failed: {str(e)}",
                error=str(e)
            )


class GrepTool(BaseTool):
    """Tool for searching text in files."""

    def __init__(self):
        super().__init__(
            name="grep",
            description="Search for text patterns in files using grep-like functionality."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Text pattern to search for"
                },
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of files to search in"
                },
                "directory": {
                    "type": "string",
                    "description": "Directory to search in (alternative to files list)"
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Search recursively in subdirectories",
                    "default": False
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Case sensitive search",
                    "default": True
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Number of context lines to show around matches",
                    "default": 2
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 100
                }
            },
            "required": ["pattern"]
        }

    async def execute(
        self,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        pattern = parameters["pattern"]
        files = parameters.get("files", [])
        directory = parameters.get("directory")
        recursive = parameters.get("recursive", False)
        case_sensitive = parameters.get("case_sensitive", True)
        context_lines = parameters.get("context_lines", 2)
        max_results = parameters.get("max_results", 100)

        try:
            import re
            import os

            # Compile regex pattern
            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                regex = re.compile(pattern, flags)
            except re.error as e:
                return SimpleToolResult(
                    content=f"Invalid regex pattern: {str(e)}",
                    error="Invalid regex"
                )

            search_files = []

            # Determine files to search
            if files:
                search_files = files
            elif directory:
                abs_dir = os.path.abspath(directory)
                if not os.path.exists(abs_dir):
                    return SimpleToolResult(
                        content=f"Directory not found: {directory}",
                        error="Directory not found"
                    )

                if recursive:
                    for root, dirs, filenames in os.walk(abs_dir):
                        for filename in filenames:
                            # Skip binary files
                            if self._is_text_file(filename):
                                search_files.append(os.path.join(root, filename))
                else:
                    for item in os.listdir(abs_dir):
                        full_path = os.path.join(abs_dir, item)
                        if os.path.isfile(full_path) and self._is_text_file(item):
                            search_files.append(full_path)
            else:
                return SimpleToolResult(
                    content="Must specify either files list or directory to search",
                    error="No search target specified"
                )

            # Search in files
            results = []
            result_count = 0

            for file_path in search_files:
                if result_count >= max_results:
                    break

                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()

                    file_results = []
                    for line_num, line in enumerate(lines, 1):
                        if regex.search(line):
                            # Get context lines
                            start_line = max(0, line_num - 1 - context_lines)
                            end_line = min(len(lines), line_num + context_lines)

                            context_block = []
                            for i in range(start_line, end_line):
                                prefix = ">>> " if i == line_num - 1 else "    "
                                context_block.append(f"{prefix}{i+1}: {lines[i].rstrip()}")

                            file_results.append({
                                "line_num": line_num,
                                "context": "\n".join(context_block)
                            })

                            result_count += 1
                            if result_count >= max_results:
                                break

                    if file_results:
                        results.append({
                            "file": file_path,
                            "matches": file_results
                        })

                except (UnicodeDecodeError, PermissionError):
                    # Skip files that can't be read
                    continue

            # Format results
            if not results:
                content = f"No matches found for pattern: {pattern}"
            else:
                content_parts = [f"Found {result_count} matches for pattern: {pattern}\n"]

                for file_result in results:
                    content_parts.append(f"\n📄 {file_result['file']}:")
                    for match in file_result['matches']:
                        content_parts.append(f"\nLine {match['line_num']}:")
                        content_parts.append(match['context'])

                content = "\n".join(content_parts)

            return SimpleToolResult(
                content=content,
                display_content=f"Searched for '{pattern}': {result_count} matches in {len(results)} files"
            )

        except Exception as e:
            return SimpleToolResult(
                content=f"Grep search failed: {str(e)}",
                error=str(e)
            )

    def _is_text_file(self, filename: str) -> bool:
        """Check if file is likely a text file based on extension."""
        text_extensions = {
            '.txt', '.py', '.js', '.ts', '.html', '.css', '.json', '.xml',
            '.yml', '.yaml', '.md', '.rst', '.c', '.cpp', '.h', '.hpp',
            '.java', '.go', '.rs', '.sh', '.bash', '.zsh', '.fish',
            '.sql', '.csv', '.ini', '.cfg', '.conf', '.log'
        }

        _, ext = os.path.splitext(filename.lower())
        return ext in text_extensions or not ext  # Include extensionless files