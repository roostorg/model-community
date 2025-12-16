"""Base classes for tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from dataclasses import dataclass
import logging

from policy_cli.core.types import Tool, ToolResult, SimpleToolResult, ToolError

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """Base class for all tools."""

    def __init__(self, name: str, description: str):
        self._name = name
        self._description = description

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    @abstractmethod
    def parameters_schema(self) -> Dict[str, Any]:
        """JSON schema for tool parameters."""
        pass

    @abstractmethod
    async def execute(
        self,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """Execute the tool with given parameters."""
        pass

    def validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and process parameters. Override for custom validation."""
        # Basic validation - could be enhanced with jsonschema
        return parameters

    async def safe_execute(
        self,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """Execute tool with error handling."""
        try:
            validated_params = self.validate_parameters(parameters)
            return await self.execute(validated_params, context)
        except Exception as e:
            logger.error(f"Tool {self.name} execution failed: {e}")
            return SimpleToolResult(
                content=f"Tool execution failed: {str(e)}",
                error=str(e)
            )

    def to_function_declaration(self) -> Dict[str, Any]:
        """Convert tool to function declaration for AI model."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema
        }


class FileToolMixin:
    """Mixin for tools that work with files."""

    def _validate_file_path(self, path: str) -> str:
        """Validate and normalize file path."""
        import os
        import os.path

        if not path:
            raise ToolError("File path cannot be empty", getattr(self, 'name', 'unknown'))

        # Convert to absolute path
        abs_path = os.path.abspath(path)

        # Basic security check - don't allow certain system directories
        forbidden_dirs = ['/etc', '/usr', '/bin', '/sbin']
        for forbidden in forbidden_dirs:
            if abs_path.startswith(forbidden):
                raise ToolError(
                    f"Access to system directory {forbidden} is not allowed",
                    getattr(self, 'name', 'unknown')
                )

        return abs_path

    def _read_file_safe(self, path: str, max_size: int = 10 * 1024 * 1024) -> str:
        """Read file with size limit and encoding detection."""
        import os

        abs_path = self._validate_file_path(path)

        if not os.path.exists(abs_path):
            raise ToolError(f"File not found: {path}", getattr(self, 'name', 'unknown'))

        if not os.path.isfile(abs_path):
            raise ToolError(f"Path is not a file: {path}", getattr(self, 'name', 'unknown'))

        # Check file size
        size = os.path.getsize(abs_path)
        if size > max_size:
            raise ToolError(
                f"File too large: {size} bytes (max {max_size})",
                getattr(self, 'name', 'unknown')
            )

        # Try to read with different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252']
        for encoding in encodings:
            try:
                with open(abs_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue

        raise ToolError(
            f"Could not decode file with any supported encoding",
            getattr(self, 'name', 'unknown')
        )

    def _write_file_safe(self, path: str, content: str, create_dirs: bool = True) -> None:
        """Write file safely with directory creation."""
        import os
        import os.path

        abs_path = self._validate_file_path(path)

        if create_dirs:
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)

        try:
            with open(abs_path, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            raise ToolError(
                f"Failed to write file: {str(e)}",
                getattr(self, 'name', 'unknown')
            )


class ShellToolMixin:
    """Mixin for tools that execute shell commands."""

    def _validate_command(self, command: str) -> str:
        """Validate shell command for safety."""
        if not command.strip():
            raise ToolError("Command cannot be empty", getattr(self, 'name', 'unknown'))

        # Basic security checks
        dangerous_commands = [
            'rm -rf /',
            'format',
            'del /s',
            'shutdown',
            'reboot',
            'halt',
            'init 0',
            'init 6',
            ':(){ :|:& };:',  # Fork bomb
        ]

        command_lower = command.lower()
        for dangerous in dangerous_commands:
            if dangerous in command_lower:
                raise ToolError(
                    f"Dangerous command detected: {dangerous}",
                    getattr(self, 'name', 'unknown')
                )

        return command

    async def _execute_shell_command(
        self,
        command: str,
        timeout: int = 30,
        capture_output: bool = True
    ) -> tuple[str, str, int]:
        """Execute shell command with timeout."""
        import asyncio
        import subprocess

        validated_command = self._validate_command(command)

        try:
            if capture_output:
                process = await asyncio.create_subprocess_shell(
                    validated_command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=None  # Use current working directory
                )
            else:
                process = await asyncio.create_subprocess_shell(validated_command)

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )

            return (
                stdout.decode('utf-8', errors='replace') if stdout else '',
                stderr.decode('utf-8', errors='replace') if stderr else '',
                process.returncode or 0
            )

        except asyncio.TimeoutError:
            if process:
                process.kill()
                await process.wait()
            raise ToolError(
                f"Command timed out after {timeout} seconds",
                getattr(self, 'name', 'unknown')
            )
        except Exception as e:
            raise ToolError(
                f"Command execution failed: {str(e)}",
                getattr(self, 'name', 'unknown')
            )