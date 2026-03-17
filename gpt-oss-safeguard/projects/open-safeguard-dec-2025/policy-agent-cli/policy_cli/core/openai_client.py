"""OpenAI chat client implementation."""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Any, AsyncGenerator, Optional

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall

from policy_cli.constants import CLI_MODEL
from policy_cli.core.types import (
    Message,
    FunctionCall,
    GenerationConfig,
    StreamingClient
)
from policy_cli.core.events import (
    Event,
    ContentChunkEvent,
    ToolCallRequestEvent,
    FinishedEvent,
    EventType
)

logger = logging.getLogger(__name__)


class OpenAIChatClient:
    """OpenAI chat client with streaming support."""

    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = CLI_MODEL

    async def generate_stream(
        self,
        messages: List[Message],
        config: GenerationConfig
    ) -> AsyncGenerator[Event, None]:
        """Generate streaming response from OpenAI."""
        logger.info(f"Starting OpenAI stream with {len(messages)} messages")
        try:
            # Convert messages to OpenAI format
            openai_messages = self._convert_messages(messages)
            logger.debug(f"Converted to {len(openai_messages)} OpenAI messages")

            # Prepare request parameters
            request_params = {
                "model": config.model or self.model,
                "messages": openai_messages,
                "temperature": config.temperature,
                "stream": True,
            }

            if config.max_tokens:
                request_params["max_tokens"] = config.max_tokens

            if config.tools:
                request_params["tools"] = self._convert_tools(config.tools)
                request_params["tool_choice"] = "auto"

            # Make streaming request
            logger.debug(f"Making OpenAI API call with model: {request_params['model']}")
            stream = await self.client.chat.completions.create(**request_params)

            current_tool_calls: Dict[str, Dict[str, Any]] = {}
            chunk_count = 0

            async for chunk in stream:
                chunk_count += 1
                # print(f"\nchunk {chunk_count}:{chunk}\n")
                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                delta = choice.delta

                ## Marshall: the beauty of this async-for loop is that
                ##            1. if text token, if bubbles it up, so that iOS can stream it immediately.
                ##            2. but if it's related to a tool call, it will keep accumulating the tool call info until the tool-call-definition is complete.
                # Handle content chunks
                if delta.content:
                    logger.debug(f"Content chunk: {delta.content}")
                    yield ContentChunkEvent(text=delta.content)

                # Handle tool calls
                if delta.tool_calls: # marshall: why not "else if"? is it possible to have "content" and "tool_call" at the same time?
                    for tool_call_delta in delta.tool_calls:
                        tool_call_id = tool_call_delta.id or ""

                        # marshall: OpenAI can stream tool call subsequent chunks WITHOUT a valid id (UUID) nor index.
                        if tool_call_id and (tool_call_id not in current_tool_calls): # if tool_call_id == '', which happens in subsequent chunks, this would erroneously create a separate tool_call object.
                            current_tool_calls[tool_call_id] = {
                                "id": tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": "", # marshall: will be populated below
                                    "arguments": ""
                                }
                            }
                            # only update variable "current_call" if there is an id.  Otherwise, use the existing current_call.
                            current_call = current_tool_calls[tool_call_id]
                            

                        if tool_call_delta.function:
                            if tool_call_delta.function.name:
                                current_call["function"]["name"] = tool_call_delta.function.name
                            if tool_call_delta.function.arguments:
                                current_call["function"]["arguments"] += tool_call_delta.function.arguments

                # Check if we have complete tool calls to emit
                if choice.finish_reason == "tool_calls":
                    for tool_call in current_tool_calls.values():
                        if tool_call["function"]["name"] and tool_call["function"]["arguments"]: # marshall: why "and", some function can have no arguments (?)
                            try:
                                parameters = json.loads(tool_call["function"]["arguments"])
                                yield ToolCallRequestEvent(
                                    call_id=tool_call["id"],
                                    tool_name=tool_call["function"]["name"],
                                    parameters=parameters
                                )
                            except json.JSONDecodeError as e:
                                logger.error(f"Failed to parse tool arguments: {e}")
                    # Don't finish yet - let the caller execute tools and continue the loop
                    logger.info(f"\n\nStream paused after {chunk_count} chunks for tool execution")
                    break

                # Handle completion (only for stop/length, NOT tool_calls)
                elif choice.finish_reason in ["stop", "length"]:
                    logger.info(f"\n\nStream finished after {chunk_count} chunks with reason: {choice.finish_reason}")
                    yield FinishedEvent()
                    break

        except Exception as e:
            logger.error(f"Error in OpenAI stream: {e}", exc_info=True)
            yield FinishedEvent()

    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert internal messages to OpenAI format."""
        openai_messages = []

        for message in messages:
            openai_msg = {
                "role": message.role,
                "content": message.content or ""
            }

            # Handle tool calls
            if message.function_calls:
                openai_msg["tool_calls"] = []
                for func_call in message.function_calls:
                    openai_msg["tool_calls"].append({
                        "id": func_call.id,
                        "type": "function",
                        "function": {
                            "name": func_call.name,
                            "arguments": json.dumps(func_call.parameters)
                        }
                    })

            # Handle tool responses
            if message.role == "tool":
                openai_msg["tool_call_id"] = getattr(message, "tool_call_id", "")

            openai_messages.append(openai_msg)

        return openai_messages

    def _convert_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert function declarations to OpenAI tools format."""
        openai_tools = []

        for tool in tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                }
            }

            if "parameters" in tool:
                openai_tool["function"]["parameters"] = tool["parameters"]

            openai_tools.append(openai_tool)

        return openai_tools


class ChatSession:
    """Chat session management for OpenAI."""

    def __init__(self, client: OpenAIChatClient, config: GenerationConfig):
        self.client = client
        self.config = config
        self.messages: List[Message] = []

    async def send_message(self, content: str) -> AsyncGenerator[Event, None]:
        """Send a message and get streaming response."""
        # Add user message
        user_message = Message(role="user", content=content)
        self.messages.append(user_message)

        # Generate response
        async for event in self.client.generate_stream(self.messages, self.config):
            yield event

    def add_assistant_message(self, content: str, function_calls: Optional[List[FunctionCall]] = None) -> None:
        """Add assistant message to conversation."""
        message = Message(
            role="assistant",
            content=content,
            function_calls=function_calls or []
        )
        self.messages.append(message)

    def add_function_response(self, call_id: str, content: str, error: Optional[str] = None) -> None:
        """Add function response to conversation."""
        response_content = content
        if error:
            response_content = f"Error: {error}"

        message = Message(
            role="tool",
            content=response_content,
            tool_call_id=call_id
        )
        self.messages.append(message)

    def get_conversation_history(self) -> List[Message]:
        """Get the full conversation history."""
        return self.messages.copy()

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.messages.clear()