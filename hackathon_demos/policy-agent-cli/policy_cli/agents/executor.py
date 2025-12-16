"""Agent executor implementing the agentic feedback loop."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass
from enum import Enum

from policy_cli.constants import CLI_MODEL, DEFAULT_TEMPERATURE
from policy_cli.core.events import (
    Event,
    EventType,
    ToolCallRequestEvent,
    ToolCallResponseEvent,
    AgentActivityEvent,
    FinishedEvent,
    ErrorEvent
)
from policy_cli.core.types import (
    AgentConfig,
    AgentResult,
    AgentError,
    Message,
    GenerationConfig,
    FunctionCall,
    Tool,
    ToolResult
)
from policy_cli.core.openai_client import OpenAIChatClient, ChatSession
from policy_cli.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class TerminateMode(Enum):
    """Reasons for agent termination."""

    GOAL_ACHIEVED = "goal_achieved"
    MAX_TURNS = "max_turns"
    TIMEOUT = "timeout"
    ERROR = "error"
    USER_CANCELLED = "user_cancelled"


@dataclass
class AgentExecutionContext:
    """Context for agent execution."""

    config: AgentConfig
    tool_registry: ToolRegistry
    chat_client: OpenAIChatClient
    workspace_context: Optional[Dict[str, Any]] = None


class AgentExecutor:
    """
    Executes an agent using the two-phase approach:
    1. Work Phase: Agent loops calling tools until it has gathered enough information
    2. Extraction Phase: Final summarization of findings
    """

    def __init__(self, context: AgentExecutionContext):
        self.context = context
        self.config = context.config
        self.tool_registry = context.tool_registry
        self.chat_client = context.chat_client
        self.turns_used = 0
        self.tools_called: List[str] = []
        self.start_time: Optional[float] = None

    async def execute(
        self,
        initial_prompt: str,
        inputs: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        """
        Execute the agent with the gather context → take action → verify → repeat loop.
        """
        self.start_time = time.time()
        self.turns_used = 0
        self.tools_called = []

        try:
            # Phase 1: Work Phase - Agent gathers information and takes actions
            terminate_reason = await self._work_phase(initial_prompt, inputs or {})

            if terminate_reason != TerminateMode.GOAL_ACHIEVED:
                return AgentResult(
                    result=f"Agent execution terminated: {terminate_reason.value}",
                    terminate_reason=terminate_reason.value,
                    turns_used=self.turns_used,
                    tools_called=self.tools_called
                )

            # Phase 2: Extraction Phase - Summarize findings
            final_result = await self._extraction_phase()

            return AgentResult(
                result=final_result,
                terminate_reason=terminate_reason.value,
                turns_used=self.turns_used,
                tools_called=self.tools_called,
                metadata={
                    "execution_time": time.time() - (self.start_time or 0),
                    "agent_name": self.config.name
                }
            )

        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            return AgentResult(
                result=f"Agent execution failed: {str(e)}",
                terminate_reason=TerminateMode.ERROR.value,
                turns_used=self.turns_used,
                tools_called=self.tools_called
            )

    async def _work_phase(
        self,
        initial_prompt: str,
        inputs: Dict[str, Any]
    ) -> TerminateMode:
        """
        Work phase: Agent loops calling tools until goal is achieved or termination.
        Implements: gather context → take action → verify → repeat
        """
        # Setup generation config with available tools
        tools = self._get_available_tools()
        generation_config = GenerationConfig(
            model=self.config.model,
            tools=tools,
            temperature=DEFAULT_TEMPERATURE,  # Slightly creative for problem solving
            stream=True
        )

        # Create chat session
        chat_session = ChatSession(self.chat_client, generation_config)

        # Add system prompt if configured
        if self.config.system_prompt:
            chat_session.add_assistant_message(self.config.system_prompt)

        # Format initial prompt with inputs
        formatted_prompt = self._format_prompt(initial_prompt, inputs)

        while True:
            # Check termination conditions
            terminate_reason = self._check_termination()
            if terminate_reason:
                return terminate_reason

            logger.debug(f"Agent turn {self.turns_used + 1}/{self.config.max_turns}")

            # Send message and process response
            has_tool_calls = False
            response_content = ""

            async for event in chat_session.send_message(formatted_prompt):
                if event.type == EventType.CONTENT_CHUNK:
                    response_content += event.data["text"]

                elif event.type == EventType.TOOL_CALL_REQUEST:
                    has_tool_calls = True
                    await self._handle_tool_call(event, chat_session)

                elif event.type == EventType.ERROR:
                    logger.error(f"Error in agent turn: {event.data['message']}")
                    return TerminateMode.ERROR

                elif event.type == EventType.FINISHED:
                    break

            self.turns_used += 1

            # If no tool calls were made, the agent considers its work complete
            if not has_tool_calls:
                logger.debug("Agent finished work phase - no more tool calls")
                return TerminateMode.GOAL_ACHIEVED

            # For subsequent turns, use empty prompt (continue conversation)
            formatted_prompt = ""

    async def _extraction_phase(self) -> str:
        """
        Extraction phase: Summarize the work and extract final result.
        """
        extraction_prompt = f"""
        Based on the work you've done, please provide a final summary and answer.

        Task: {self.config.description}

        Please provide:
        1. A summary of what you discovered
        2. The final answer or result
        3. Any recommendations or next steps

        Be concise but thorough.
        """

        # Use a clean chat session for extraction (no tools needed)
        extraction_config = GenerationConfig(
            model=self.config.model,
            tools=None,  # No tools in extraction phase
            temperature=DEFAULT_TEMPERATURE,
            stream=True
        )

        chat_session = ChatSession(self.chat_client, extraction_config)
        result_content = ""

        async for event in chat_session.send_message(extraction_prompt):
            if event.type == EventType.CONTENT_CHUNK:
                result_content += event.data["text"]
            elif event.type == EventType.FINISHED:
                break

        return result_content.strip() or "No result generated"

    async def _handle_tool_call(
        self,
        event: ToolCallRequestEvent,
        chat_session: ChatSession
    ) -> None:
        """Handle a tool call request from the agent."""
        tool_name = event.tool_name
        call_id = event.call_id
        parameters = event.parameters

        logger.debug(f"Agent calling tool: {tool_name} with params: {parameters}")

        try:
            # Get tool from registry
            tool = self.tool_registry.get_tool(tool_name)
            if not tool:
                error_msg = f"Tool '{tool_name}' not found in registry"
                chat_session.add_function_response(call_id, "", error=error_msg)
                return

            # Execute tool
            result = await tool.execute(parameters, context=self.context.workspace_context)

            # Track tool usage
            if tool_name not in self.tools_called:
                self.tools_called.append(tool_name)

            # Add result to conversation
            content = result.content
            if result.error:
                content = f"Error: {result.error}\n{content}"

            chat_session.add_function_response(call_id, content, error=result.error)

            logger.debug(f"Tool {tool_name} completed successfully")

        except Exception as e:
            error_msg = f"Tool execution failed: {str(e)}"
            logger.error(f"Tool {tool_name} failed: {e}")
            chat_session.add_function_response(call_id, "", error=error_msg)

    def _check_termination(self) -> Optional[TerminateMode]:
        """Check if agent should terminate."""
        # Check max turns
        if self.turns_used >= self.config.max_turns:
            return TerminateMode.MAX_TURNS

        # Check timeout
        if self.start_time:
            elapsed = time.time() - self.start_time
            if elapsed > self.config.timeout_seconds:
                return TerminateMode.TIMEOUT

        return None

    def _get_available_tools(self) -> List[Dict[str, Any]]:
        """Get tool definitions for the agent."""
        # Filter tools based on agent config
        available_tools = []

        if self.config.tools:
            # Agent has specific tool allowlist
            for tool_name in self.config.tools:
                if self.tool_registry.has_tool(tool_name):
                    tool = self.tool_registry.get_tool(tool_name)
                    if tool and hasattr(tool, 'to_function_declaration'):
                        available_tools.append(tool.to_function_declaration())
        else:
            # Use all trusted tools
            available_tools = self.tool_registry.get_function_declarations(
                trusted_only=True
            )

        logger.debug(f"Agent has access to {len(available_tools)} tools")
        return available_tools

    def _format_prompt(self, prompt: str, inputs: Dict[str, Any]) -> str:
        """Format prompt with inputs using simple string formatting."""
        try:
            return prompt.format(**inputs)
        except KeyError as e:
            logger.warning(f"Missing input for prompt formatting: {e}")
            return prompt
        except Exception as e:
            logger.warning(f"Prompt formatting failed: {e}")
            return prompt


class SimpleAgent:
    """
    Simplified agent for quick tasks that don't need the full executor.
    """

    def __init__(self, chat_client: OpenAIChatClient, tool_registry: ToolRegistry):
        self.chat_client = chat_client
        self.tool_registry = tool_registry

    async def run_simple(
        self,
        prompt: str,
        tools: Optional[List[str]] = None,
        model: str = CLI_MODEL
    ) -> str:
        """Run a simple agent task."""
        # Get tool definitions
        if tools:
            tool_declarations = []
            for tool_name in tools:
                tool = self.tool_registry.get_tool(tool_name)
                if tool and hasattr(tool, 'to_function_declaration'):
                    tool_declarations.append(tool.to_function_declaration())
        else:
            tool_declarations = self.tool_registry.get_function_declarations(trusted_only=True)

        # Setup generation config
        config = GenerationConfig(
            model=model,
            tools=tool_declarations,
            stream=True
        )

        # Create chat session and run
        chat_session = ChatSession(self.chat_client, config)
        result_content = ""

        async for event in chat_session.send_message(prompt):
            if event.type == EventType.CONTENT_CHUNK:
                result_content += event.data["text"]

            elif event.type == EventType.TOOL_CALL_REQUEST:
                await self._handle_simple_tool_call(event, chat_session)

            elif event.type == EventType.FINISHED:
                break

        return result_content.strip()

    async def _handle_simple_tool_call(
        self,
        event: ToolCallRequestEvent,
        chat_session: ChatSession
    ) -> None:
        """Handle tool call for simple agent."""
        tool = self.tool_registry.get_tool(event.tool_name)
        if not tool:
            chat_session.add_function_response(
                event.call_id,
                "",
                error=f"Tool {event.tool_name} not found"
            )
            return

        try:
            result = await tool.execute(event.parameters)
            chat_session.add_function_response(
                event.call_id,
                result.content,
                error=result.error
            )
        except Exception as e:
            chat_session.add_function_response(
                event.call_id,
                "",
                error=str(e)
            )