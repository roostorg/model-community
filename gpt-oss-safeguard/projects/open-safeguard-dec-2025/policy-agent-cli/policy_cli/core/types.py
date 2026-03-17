"""Core type definitions for Policy CLI."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, AsyncGenerator, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum

from policy_cli.constants import CLI_MODEL, DEFAULT_TEMPERATURE
from policy_cli.core.events import Event


class FinishReason(Enum):
    """Reasons why content generation finished."""

    STOP = "stop"
    MAX_TOKENS = "max_tokens"
    FUNCTION_CALL = "function_call"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"


@dataclass
class Message:
    """Represents a message in the conversation."""

    role: str  # "user", "assistant", "tool"
    content: Optional[str] = None
    function_calls: Optional[List[FunctionCall]] = None
    function_response: Optional[FunctionResponse] = None
    metadata: Optional[Dict[str, Any]] = None
    tool_call_id: Optional[str] = None  # For tool response messages


@dataclass
class FunctionCall:
    """Represents a function call request."""

    id: str
    name: str
    parameters: Dict[str, Any]


@dataclass
class FunctionResponse:
    """Represents a function call response."""

    call_id: str
    content: str
    error: Optional[str] = None


@dataclass
class GenerationConfig:
    """Configuration for content generation."""

    model: str = CLI_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stream: bool = True
    tools: Optional[List[Dict[str, Any]]] = None


@dataclass
class GenerationResponse:
    """Response from content generation."""

    content: str
    finish_reason: FinishReason
    function_calls: Optional[List[FunctionCall]] = None
    usage: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class StreamingClient(Protocol):
    """Protocol for streaming AI clients."""

    async def generate_stream(
        self,
        messages: List[Message],
        config: GenerationConfig,
    ) -> AsyncGenerator[Event, None]:
        """Generate streaming response."""
        ...


class ToolResult(Protocol):
    """Protocol for tool execution results."""

    @property
    def content(self) -> str:
        """Main content for the LLM."""
        ...

    @property
    def display_content(self) -> Optional[str]:
        """Content for user display (optional)."""
        ...

    @property
    def error(self) -> Optional[str]:
        """Error message if execution failed."""
        ...


@dataclass
class SimpleToolResult:
    """Simple implementation of ToolResult."""

    content: str
    display_content: Optional[str] = None
    error: Optional[str] = None


class Tool(Protocol):
    """Protocol for tools that can be executed."""

    @property
    def name(self) -> str:
        """Tool name."""
        ...

    @property
    def description(self) -> str:
        """Tool description."""
        ...

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        """JSON schema for tool parameters."""
        ...

    async def execute(
        self,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """Execute the tool with given parameters."""
        ...


@dataclass
class AgentConfig:
    """Configuration for an agent."""

    name: str
    description: str
    model: str = CLI_MODEL
    tools: Optional[List[str]] = None  # Tool names to include
    max_turns: int = 50
    timeout_seconds: int = 300
    system_prompt: Optional[str] = None


@dataclass
class AgentResult:
    """Result from agent execution."""

    result: str
    terminate_reason: str
    turns_used: int
    tools_called: List[str]
    metadata: Optional[Dict[str, Any]] = None


class ConfigurationError(Exception):
    """Raised when there's a configuration error."""
    pass


class ToolError(Exception):
    """Raised when tool execution fails."""

    def __init__(self, message: str, tool_name: str, error_type: str = "execution_error"):
        super().__init__(message)
        self.tool_name = tool_name
        self.error_type = error_type


class AgentError(Exception):
    """Raised when agent execution fails."""

    def __init__(self, message: str, agent_name: str, error_type: str = "execution_error"):
        super().__init__(message)
        self.agent_name = agent_name
        self.error_type = error_type