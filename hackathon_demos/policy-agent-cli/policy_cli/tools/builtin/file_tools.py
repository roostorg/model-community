"""File operation tools."""

from __future__ import annotations

import os
import os.path
import glob
from typing import Dict, Any, Optional
import logging

from policy_cli.tools.base import BaseTool, FileToolMixin
from policy_cli.core.types import ToolResult, SimpleToolResult

logger = logging.getLogger(__name__)


class ReadFileTool(BaseTool, FileToolMixin):
    """Tool for reading file contents."""

    def __init__(self):
        super().__init__(
            name="read_file",
            description="Read the contents of a file. Supports text files up to 10MB."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to read"
                },
                "encoding": {
                    "type": "string",
                    "description": "File encoding (default: auto-detect)",
                    "default": "auto"
                }
            },
            "required": ["file_path"]
        }

    async def execute(
        self,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        file_path = parameters["file_path"]

        try:
            content = self._read_file_safe(file_path)
            return SimpleToolResult(
                content=f"File content:\n```\n{content}\n```",
                display_content=f"Read {len(content)} characters from {file_path}"
            )
        except Exception as e:
            return SimpleToolResult(
                content=f"Failed to read file: {str(e)}",
                error=str(e)
            )


class WriteFileTool(BaseTool, FileToolMixin):
    """Tool for writing files."""

    def __init__(self):
        super().__init__(
            name="write_file",
            description="Write content to a file. Creates directories as needed."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                },
                "create_dirs": {
                    "type": "boolean",
                    "description": "Create parent directories if they don't exist",
                    "default": True
                }
            },
            "required": ["file_path", "content"]
        }

    async def execute(
        self,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        file_path = parameters["file_path"]
        content = parameters["content"]
        create_dirs = parameters.get("create_dirs", True)

        try:
            self._write_file_safe(file_path, content, create_dirs)
            return SimpleToolResult(
                content=f"Successfully wrote {len(content)} characters to {file_path}",
                display_content=f"Wrote file: {file_path}"
            )
        except Exception as e:
            return SimpleToolResult(
                content=f"Failed to write file: {str(e)}",
                error=str(e)
            )


class ListDirectoryTool(BaseTool):
    """Tool for listing directory contents."""

    def __init__(self):
        super().__init__(
            name="list_directory",
            description="List the contents of a directory with file information."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "directory_path": {
                    "type": "string",
                    "description": "Path to the directory to list"
                },
                "show_hidden": {
                    "type": "boolean",
                    "description": "Include hidden files (starting with .)",
                    "default": False
                },
                "recursive": {
                    "type": "boolean",
                    "description": "List directories recursively",
                    "default": False
                }
            },
            "required": ["directory_path"]
        }

    async def execute(
        self,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        directory_path = parameters["directory_path"]
        show_hidden = parameters.get("show_hidden", False)
        recursive = parameters.get("recursive", False)

        try:
            abs_path = os.path.abspath(directory_path)

            if not os.path.exists(abs_path):
                return SimpleToolResult(
                    content=f"Directory not found: {directory_path}",
                    error="Directory not found"
                )

            if not os.path.isdir(abs_path):
                return SimpleToolResult(
                    content=f"Path is not a directory: {directory_path}",
                    error="Not a directory"
                )

            entries = []

            if recursive:
                for root, dirs, files in os.walk(abs_path):
                    # Filter hidden files/dirs if requested
                    if not show_hidden:
                        dirs[:] = [d for d in dirs if not d.startswith('.')]
                        files = [f for f in files if not f.startswith('.')]

                    # Add directories
                    for dir_name in dirs:
                        full_path = os.path.join(root, dir_name)
                        rel_path = os.path.relpath(full_path, abs_path)
                        entries.append(f"📁 {rel_path}/")

                    # Add files
                    for file_name in files:
                        full_path = os.path.join(root, file_name)
                        rel_path = os.path.relpath(full_path, abs_path)
                        try:
                            size = os.path.getsize(full_path)
                            entries.append(f"📄 {rel_path} ({self._format_size(size)})")
                        except OSError:
                            entries.append(f"📄 {rel_path} (size unknown)")
            else:
                # Non-recursive listing
                items = os.listdir(abs_path)
                if not show_hidden:
                    items = [item for item in items if not item.startswith('.')]

                for item in sorted(items):
                    full_path = os.path.join(abs_path, item)
                    if os.path.isdir(full_path):
                        entries.append(f"📁 {item}/")
                    else:
                        try:
                            size = os.path.getsize(full_path)
                            entries.append(f"📄 {item} ({self._format_size(size)})")
                        except OSError:
                            entries.append(f"📄 {item} (size unknown)")

            if not entries:
                content = f"Directory is empty: {directory_path}"
            else:
                content = f"Contents of {directory_path}:\n" + "\n".join(entries)

            return SimpleToolResult(
                content=content,
                display_content=f"Listed {len(entries)} items in {directory_path}"
            )

        except Exception as e:
            return SimpleToolResult(
                content=f"Failed to list directory: {str(e)}",
                error=str(e)
            )

    def _format_size(self, size: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f}{unit}"
            size /= 1024
        return f"{size:.1f}TB"


class GlobTool(BaseTool):
    """Tool for finding files using glob patterns."""

    def __init__(self):
        super().__init__(
            name="glob_files",
            description="Find files matching glob patterns (e.g., '*.py', '**/*.js')."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match files"
                },
                "directory": {
                    "type": "string",
                    "description": "Directory to search in (default: current directory)",
                    "default": "."
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Use recursive search with **",
                    "default": True
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
        directory = parameters.get("directory", ".")
        recursive = parameters.get("recursive", True)

        try:
            abs_dir = os.path.abspath(directory)

            if not os.path.exists(abs_dir):
                return SimpleToolResult(
                    content=f"Directory not found: {directory}",
                    error="Directory not found"
                )

            # Change to target directory for glob search
            old_cwd = os.getcwd()
            os.chdir(abs_dir)

            try:
                matches = glob.glob(pattern, recursive=recursive)
                # Convert back to absolute paths
                matches = [os.path.join(abs_dir, match) for match in matches]
                matches = sorted(matches)

                if not matches:
                    content = f"No files found matching pattern: {pattern}"
                else:
                    # Make paths relative to original directory for display
                    display_matches = [os.path.relpath(match, old_cwd) for match in matches]
                    content = f"Files matching '{pattern}':\n" + "\n".join(f"📄 {match}" for match in display_matches)

                return SimpleToolResult(
                    content=content,
                    display_content=f"Found {len(matches)} files matching {pattern}"
                )

            finally:
                os.chdir(old_cwd)

        except Exception as e:
            return SimpleToolResult(
                content=f"Glob search failed: {str(e)}",
                error=str(e)
            )


class EditFileTool(BaseTool, FileToolMixin):
    """Tool for editing files with find/replace operations."""

    def __init__(self):
        super().__init__(
            name="edit_file",
            description="Edit a file by replacing text. Supports exact string replacement."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to edit"
                },
                "old_text": {
                    "type": "string",
                    "description": "Text to find and replace"
                },
                "new_text": {
                    "type": "string",
                    "description": "New text to replace with"
                },
                "count": {
                    "type": "integer",
                    "description": "Maximum number of replacements (default: all)",
                    "default": -1
                }
            },
            "required": ["file_path", "old_text", "new_text"]
        }

    async def execute(
        self,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        file_path = parameters["file_path"]
        old_text = parameters["old_text"]
        new_text = parameters["new_text"]
        count = parameters.get("count", -1)

        try:
            # Read current content
            content = self._read_file_safe(file_path)

            # Count occurrences before replacement
            occurrences = content.count(old_text)
            if occurrences == 0:
                return SimpleToolResult(
                    content=f"Text not found in file: '{old_text}'",
                    error="Text not found"
                )

            # Perform replacement
            new_content = content.replace(old_text, new_text, count)
            actual_replacements = occurrences if count == -1 else min(count, occurrences)

            # Write back to file
            self._write_file_safe(file_path, new_content, create_dirs=False)

            return SimpleToolResult(
                content=f"Successfully replaced {actual_replacements} occurrence(s) in {file_path}",
                display_content=f"Edited {file_path}: {actual_replacements} replacements"
            )

        except Exception as e:
            return SimpleToolResult(
                content=f"Failed to edit file: {str(e)}",
                error=str(e)
            )