"""Event system for streaming AI interactions."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


class EventType(Enum):
    """Types of events that can occur during AI interaction."""

    # Content events
    CONTENT_CHUNK = "content_chunk"
    CONTENT_FINISHED = "content_finished"

    # Tool events
    TOOL_CALL_REQUEST = "tool_call_request"
    TOOL_CALL_RESPONSE = "tool_call_response"
    TOOL_CALL_CONFIRMATION = "tool_call_confirmation"

    # Control events
    USER_CANCELLED = "user_cancelled"
    ERROR = "error"
    RETRY = "retry"
    FINISHED = "finished"

    # Agent events
    AGENT_ACTIVITY = "agent_activity"
    THOUGHT_CHUNK = "thought_chunk"


@dataclass
class BaseEvent(ABC):
    """Base class for all events."""

    type: EventType
    data: Dict[str, Any]
    timestamp: Optional[float] = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            import time
            self.timestamp = time.time()


class ContentChunkEvent(BaseEvent):
    """Event for content streaming chunks."""

    def __init__(self, text: str, **kwargs: Any) -> None:
        self.type = EventType.CONTENT_CHUNK
        self.data = {"text": text, **kwargs}
        self.timestamp = None
        self.__post_init__()

    @property
    def text(self) -> str:
        return self.data["text"]


class ToolCallRequestEvent(BaseEvent):
    """Event for tool call requests."""

    def __init__(
        self,
        call_id: str,
        tool_name: str,
        parameters: Dict[str, Any],
        **kwargs: Any
    ) -> None:
        self.type = EventType.TOOL_CALL_REQUEST
        self.data = {
            "call_id": call_id,
            "tool_name": tool_name,
            "parameters": parameters,
            **kwargs
        }
        self.timestamp = None
        self.__post_init__()

    @property
    def call_id(self) -> str:
        return self.data["call_id"]

    @property
    def tool_name(self) -> str:
        return self.data["tool_name"]

    @property
    def parameters(self) -> Dict[str, Any]:
        return self.data["parameters"]


class ToolCallResponseEvent(BaseEvent):
    """Event for tool call responses."""

    def __init__(
        self,
        call_id: str,
        result: Dict[str, Any],
        error: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        self.type = EventType.TOOL_CALL_RESPONSE
        self.data = {
            "call_id": call_id,
            "result": result,
            "error": error,
            **kwargs
        }
        self.timestamp = None
        self.__post_init__()

    @property
    def call_id(self) -> str:
        return self.data["call_id"]

    @property
    def result(self) -> Dict[str, Any]:
        return self.data["result"]

    @property
    def error(self) -> Optional[str]:
        return self.data.get("error")


class ErrorEvent(BaseEvent):
    """Event for errors."""

    def __init__(self, message: str, error_type: Optional[str] = None, **kwargs: Any) -> None:
        self.type = EventType.ERROR
        self.data = {
            "message": message,
            "error_type": error_type,
            **kwargs
        }
        self.timestamp = None
        self.__post_init__()

    @property
    def message(self) -> str:
        return self.data["message"]

    @property
    def error_type(self) -> Optional[str]:
        return self.data.get("error_type")


class FinishedEvent(BaseEvent):
    """Event indicating interaction completion."""

    def __init__(self, reason: str = "completed", **kwargs: Any) -> None:
        self.type = EventType.FINISHED
        self.data = {"reason": reason, **kwargs}
        self.timestamp = None
        self.__post_init__()

    @property
    def reason(self) -> str:
        return self.data["reason"]


class AgentActivityEvent(BaseEvent):
    """Event for agent activity updates."""

    def __init__(
        self,
        activity_type: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        self.type = EventType.AGENT_ACTIVITY
        self.data = {
            "activity_type": activity_type,
            "message": message,
            "details": details or {},
            **kwargs
        }
        self.timestamp = None
        self.__post_init__()

    @property
    def activity_type(self) -> str:
        return self.data["activity_type"]

    @property
    def message(self) -> str:
        return self.data["message"]

    @property
    def details(self) -> Dict[str, Any]:
        return self.data.get("details", {})


# Union type for all events
Event = (
    ContentChunkEvent |
    ToolCallRequestEvent |
    ToolCallResponseEvent |
    ErrorEvent |
    FinishedEvent |
    AgentActivityEvent |
    BaseEvent
)