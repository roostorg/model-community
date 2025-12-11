"""Main entry point for Policy CLI."""

from __future__ import annotations

import asyncio
import os
import sys
import logging
from typing import Optional

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.prompt import Prompt, Confirm
from rich.text import Text
from rich.panel import Panel

from policy_cli.constants import CLI_MODEL
from policy_cli.config import Config
from policy_cli.core.openai_client import OpenAIChatClient, ChatSession
from policy_cli.core.events import EventType
from policy_cli.core.types import GenerationConfig, AgentConfig, Message, FunctionCall
from policy_cli.tools.registry import ToolRegistry
from policy_cli.tools.builtin.registry import register_builtin_tools
from policy_cli.agents.executor import AgentExecutor, AgentExecutionContext
from policy_cli.mcp.client import MCPManager
from policy_cli.__about__ import __version__

logger = logging.getLogger(__name__)

# Set up console and logging
console = Console()

def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, show_path=False)]
    )


@click.group()
@click.version_option(version=__version__)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--config-dir", help="Configuration directory")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, config_dir: Optional[str]) -> None:
    """Policy CLI - AI-powered coding assistant."""
    ctx.ensure_object(dict)

    # Setup logging
    setup_logging(verbose)

    # Load configuration
    config = Config(config_dir)
    ctx.obj["config"] = config
    ctx.obj["verbose"] = verbose or config.get("ui.verbose", False)


@cli.command()
# @click.option("--prompt", "-p", help="Single prompt to execute")
@click.option("--model", "-m", help="Model to use")
@click.option("--agent", "-a", help="Agent configuration to use")
@click.option("--tools", help="Comma-separated list of tools to enable")
@click.pass_context
def chat(
    ctx: click.Context,
    # prompt: Optional[str],
    model: Optional[str],
    agent: Optional[str],
    tools: Optional[str]
) -> None:
    """Start interactive chat or execute a single prompt."""
    # asyncio.run(_chat_command(ctx, prompt, model, agent, tools))
    asyncio.run(_chat_command(ctx, model, agent, tools))


async def _chat_command(
    ctx: click.Context,
    # prompt: Optional[str],
    model: Optional[str],
    agent: Optional[str],
    tools: Optional[str]
) -> None:
    """Async implementation of chat command."""
    config: Config = ctx.obj["config"]
    verbose: bool = ctx.obj["verbose"]

    # Check for API key
    api_key = config.get_api_key()
    if not api_key:
        console.print("[red]Error: No API key found.[/red]")
        console.print("Set your API key with: [blue]policy config set api.api_key YOUR_KEY[/blue]")
        console.print("Or set the OPENAI_API_KEY environment variable.")
        sys.exit(1)

    # Initialize components
    chat_client = OpenAIChatClient(api_key)
    tool_registry = ToolRegistry()
    mcp_manager = MCPManager()

    try:
        # Register built-in tools
        enabled_tools = tools.split(",") if tools else config.get_enabled_tools()
        register_builtin_tools(tool_registry, enabled_tools)

        # Load and connect MCP servers
        await mcp_manager.load_servers_from_config(config._config)
        mcp_results = await mcp_manager.connect_all_servers()

        # Register MCP tools
        for tool_wrapper in mcp_manager.get_tool_wrappers():
            tool_registry.register_tool(
                tool=tool_wrapper,
                source="mcp",
                server_name=tool_wrapper.mcp_tool.server_name,
                trusted=config.get_mcp_servers().get(tool_wrapper.mcp_tool.server_name, {}).get("trusted", False)
            )

        if verbose:
            stats = tool_registry.get_stats()
            console.print(f"[green]Loaded {stats['total_tools']} tools[/green]")
            for source, count in stats['by_source'].items():
                if count > 0:
                    console.print(f"  {source}: {count} tools")

        # Setup generation config
        model_config = config.get_model_config()
        generation_config = GenerationConfig(
            model=model or model_config["model"],
            temperature=model_config["temperature"],
            max_tokens=model_config["max_tokens"],
            tools=tool_registry.get_function_declarations(trusted_only=True)
        )

        # if prompt:
        #     # Single prompt mode
        #     await _execute_single_prompt(
        #         prompt, chat_client, tool_registry, generation_config, verbose
        #     )
        # else:
        #     # Interactive mode
        #     await _interactive_chat(
        #         chat_client, tool_registry, generation_config, config, verbose
        #     )
        await _interactive_chat(
            chat_client, tool_registry, generation_config, config, verbose
        )

    finally:
        await mcp_manager.shutdown()


# async def _execute_single_prompt(
#     prompt: str,
#     chat_client: OpenAIChatClient,
#     tool_registry: ToolRegistry,
#     generation_config: GenerationConfig,
#     verbose: bool
# ) -> None:
#     """Execute a single prompt and exit."""
#     chat_session = ChatSession(chat_client, generation_config)

#     console.print(Panel(f"[blue]Prompt:[/blue] {prompt}", title="Policy CLI"))

#     # Add user message to conversation
#     user_message = Message(role="user", content=prompt)
#     chat_session.messages.append(user_message)

#     # Agentic loop: Keep generating until LLM finishes (not just tool calls)
#     while True:
#         assistant_content = ""
#         assistant_tool_calls = []

#         async for event in chat_session.client.generate_stream(chat_session.messages, chat_session.config):
#             there_is_a_tool_response = False

#             if event.type == EventType.CONTENT_CHUNK:
#                 console.print(event.data["text"], end="")
#                 assistant_content += event.data["text"]

#             elif event.type == EventType.TOOL_CALL_REQUEST:
#                 if verbose:
#                     console.print(f"\n[yellow]🔧 Calling tool: {event.data['tool_name']}[/yellow]")

#                 # Store tool call info for assistant message
#                 assistant_tool_calls.append(FunctionCall(
#                     id=event.data["call_id"],
#                     name=event.data["tool_name"],
#                     parameters=event.data["parameters"]
#                 ))

#                 # Execute tool
#                 tool = tool_registry.get_tool(event.data["tool_name"])
#                 if tool:
#                     try:
#                         result = await tool.execute(event.data["parameters"])
#                         there_is_a_tool_response = True
#                         chat_session.add_function_response(
#                             event.data["call_id"],
#                             result.content,
#                             error=result.error
#                         )
#                         if verbose:
#                             console.print(f"[green]✓ Tool completed[/green]")
#                     except Exception as e:
#                         chat_session.add_function_response(
#                             event.data["call_id"],
#                             "",
#                             error=str(e)
#                         )
#                         if verbose:
#                             console.print(f"[red]✗ Tool failed: {e}[/red]")

#             elif event.type == EventType.FINISHED:
#                 # Add assistant message if any content or tool calls
#                 if assistant_content or assistant_tool_calls:
#                     chat_session.add_assistant_message(assistant_content, assistant_tool_calls)
#                 break

#             # Check if we should continue the agentic loop
#             # if event.type == EventType.FINISHED:
#             #     # Check if the last message was a tool response
#             #     # If so, continue the loop to let LLM interpret the tool response
#             #     if chat_session.messages and chat_session.messages[-1].role == "tool":
#             #         continue  # Continue agentic loop with tool response
#             #     else:
#             #         break  # LLM finished naturally, exit agentic loop
#             if there_is_a_tool_response:
#                 continue
#             else:
#                 break

#     console.print("\n")


async def _interactive_chat(
    chat_client: OpenAIChatClient,
    tool_registry: ToolRegistry,
    generation_config: GenerationConfig,
    config: Config,
    verbose: bool
) -> None:
    """Run interactive chat session."""
    console.print(Panel(
        "[bold blue]Policy CLI[/bold blue]\n"
        "Improve your gpt-oss-safeguard policies iteratively\n\n"
        "Commands:\n"
        "  /help - Show help\n"
        "  /tools - List available tools\n"
        "  /config - Show configuration\n"
        "  /quit - Exit",
        title="Welcome"
    ))

    chat_session = ChatSession(chat_client, generation_config)

    while True:
        try:
            # Get user input
            user_input = Prompt.ask("[bold blue]You[/bold blue]").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                if user_input == "/quit" or user_input == "/exit":
                    break
                elif user_input == "/help":
                    _show_help()
                    continue
                elif user_input == "/tools":
                    _show_tools(tool_registry)
                    continue
                elif user_input == "/config":
                    config.show_config()
                    continue
                else:
                    console.print(f"[red]Unknown command: {user_input}[/red]")
                    continue

            # Process user message
            console.print("\n[bold green]Assistant[/bold green]")

            # Add user message to conversation
            user_message = Message(role="user", content=user_input)
            chat_session.messages.append(user_message)

            # Agentic loop: Keep generating until LLM finishes answering the current user question
            while True:
                assistant_content = ""
                # assistant_tool_calls = []

                async for event in chat_session.client.generate_stream(chat_session.messages, chat_session.config):
                    there_is_a_tool_response = False

                    if event.type == EventType.CONTENT_CHUNK:
                        console.print(event.data["text"], end="")
                        assistant_content += event.data["text"]

                    elif event.type == EventType.TOOL_CALL_REQUEST:
                        if verbose:
                            console.print(f"\n[yellow]🔧 {event.data['tool_name']}[/yellow]", end="")

                        # Store tool call info for assistant message
                        my_tool_call = FunctionCall(
                            id=event.data["call_id"],
                            name=event.data["tool_name"],
                            parameters=event.data["parameters"]
                        )
                        chat_session.add_assistant_message(assistant_content, [my_tool_call])
                        assistant_content = ""
                        logger.info(f"🔧🔧🔧🔧🔧 Tool call: {event.data['tool_name']} - {event.data['parameters']}")


                        # Execute tool
                        tool = tool_registry.get_tool(event.data["tool_name"])
                        if tool:
                            try:
                                result = await tool.execute(event.data["parameters"])
                                there_is_a_tool_response = True
                                chat_session.add_function_response(
                                    event.data["call_id"],
                                    result.content,
                                    error=result.error
                                )
                            except Exception as e:
                                chat_session.add_function_response(
                                    event.data["call_id"],
                                    "",
                                    error=str(e)
                                )

                    elif event.type == EventType.FINISHED:
                        # Only break the outer loop when truly finished (stop/length)
                        # Add assistant message if any content or tool calls
                        # if assistant_content or assistant_tool_calls:
                        #     chat_session.add_assistant_message(assistant_content, assistant_tool_calls)
                        if assistant_content:
                            chat_session.add_assistant_message(assistant_content)
                        break

                # Check if we should continue the agentic loop
                # If we had tool calls, we need to continue to get LLM's interpretation
                # If we finished with stop/length, break the outer loop
                # if event.type == EventType.FINISHED: ## marshall: no need to check for this;
                #           the only time you don't exit out of this 2nd while-true loop is when there is a tool-response.
                #            In that case, you have to call the LLM again.
                    # # Check if the last message was a tool response
                    # # If so, continue the loop to let LLM interpret the tool response
                    # if chat_session.messages and chat_session.messages[-1].role == "tool":
                    #     continue  # Continue agentic loop with tool response
                    # else:
                    #     break  # LLM finished naturally, exit agentic loop
                if there_is_a_tool_response:
                    continue
                else:
                    break

            console.print("\n")

        except KeyboardInterrupt:
            console.print("\n[yellow]Use /quit to exit[/yellow]")
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")


def _show_help() -> None:
    """Show help information."""
    help_text = """
[bold blue]Policy CLI Help[/bold blue]

[yellow]Commands:[/yellow]
  /help     - Show this help
  /tools    - List available tools
  /config   - Show configuration
  /quit     - Exit the CLI

[yellow]Chat Features:[/yellow]
  - Ask questions about code, files, or general topics
  - Request file operations (read, write, edit)
  - Run shell commands
  - Search the web for information
  - Use MCP servers for extended functionality

[yellow]Examples:[/yellow]
  "Read the README.md file"
  "List all Python files in this directory"
  "Search for TODO comments in the codebase"
  "Create a new Python script that..."
"""
    console.print(Panel(help_text, title="Help"))


def _show_tools(tool_registry: ToolRegistry) -> None:
    """Show available tools."""
    stats = tool_registry.get_stats()

    console.print(f"\n[bold blue]Available Tools ({stats['total_tools']})[/bold blue]\n")

    for source in ["builtin", "mcp", "discovered"]:
        tools = tool_registry.list_tools(source=source)
        if tools:
            console.print(f"[yellow]{source.title()} Tools ({len(tools)}):[/yellow]")
            for tool_name in tools:
                tool = tool_registry.get_tool(tool_name)
                if tool:
                    console.print(f"  [green]{tool_name}[/green] - {tool.description}")
            console.print()


@cli.group()
def config() -> None:
    """Manage configuration."""
    pass


@config.command("show")
@click.pass_context
def config_show(ctx: click.Context) -> None:
    """Show current configuration."""
    config: Config = ctx.obj["config"]
    config.show_config()


@config.command("set")
@click.argument("key")
@click.argument("value")
@click.pass_context
def config_set(ctx: click.Context, key: str, value: str) -> None:
    """Set a configuration value."""
    config: Config = ctx.obj["config"]

    # Try to parse value as JSON for complex types
    try:
        import json
        parsed_value = json.loads(value)
    except json.JSONDecodeError:
        parsed_value = value

    config.set(key, parsed_value)
    console.print(f"[green]Set {key} = {parsed_value}[/green]")


@config.command("setup")
@click.pass_context
def config_setup(ctx: click.Context) -> None:
    """Run first-time setup."""
    config: Config = ctx.obj["config"]
    config.setup_first_time()


@cli.command()
@click.argument("query")
@click.option("--agent", "-a", help="Agent name to use")
@click.option("--max-turns", type=int, help="Maximum agent turns")
@click.option("--timeout", type=int, help="Timeout in seconds")
@click.pass_context
def agent(
    ctx: click.Context,
    query: str,
    agent: Optional[str],
    max_turns: Optional[int],
    timeout: Optional[int]
) -> None:
    """Run an agent to solve a complex task."""
    asyncio.run(_agent_command(ctx, query, agent, max_turns, timeout))

## marshall: the diff between "agent" and "chat" is that agent does the gather-action-reflect loop (?)
async def _agent_command(
    ctx: click.Context,
    query: str,
    agent: Optional[str],
    max_turns: Optional[int],
    timeout: Optional[int]
) -> None:
    """Async implementation of agent command."""
    config: Config = ctx.obj["config"]
    verbose: bool = ctx.obj["verbose"]

    # Check for API key
    api_key = config.get_api_key()
    if not api_key:
        console.print("[red]Error: No API key found.[/red]")
        sys.exit(1)

    # Initialize components
    chat_client = OpenAIChatClient(api_key)
    tool_registry = ToolRegistry()
    mcp_manager = MCPManager()

    try:
        # Register tools
        register_builtin_tools(tool_registry, config.get_enabled_tools())

        # Setup agent config
        agent_config_dict = config.get_agent_config()
        agent_config = AgentConfig(
            name=agent or "default",
            description=f"Solve this task: {query}",
            model=config.get("api.model", CLI_MODEL),
            max_turns=max_turns or agent_config_dict["max_turns"],
            timeout_seconds=timeout or agent_config_dict["timeout_seconds"]
        )

        # Create agent executor
        execution_context = AgentExecutionContext(
            config=agent_config,
            tool_registry=tool_registry,
            chat_client=chat_client
        )

        executor = AgentExecutor(execution_context)

        console.print(Panel(f"[blue]Agent Task:[/blue] {query}", title="Agent Execution"))

        if verbose:
            console.print(f"[yellow]Agent: {agent_config.name}[/yellow]")
            console.print(f"[yellow]Model: {agent_config.model}[/yellow]")
            console.print(f"[yellow]Max turns: {agent_config.max_turns}[/yellow]")

        # Execute agent
        result = await executor.execute(query)

        # Display results
        console.print(f"\n[bold green]Result:[/bold green]")
        console.print(result.result)

        console.print(f"\n[dim]Completed in {result.turns_used} turns")
        console.print(f"Tools used: {', '.join(result.tools_called) if result.tools_called else 'None'}[/dim]")

    finally:
        await mcp_manager.shutdown()


def main() -> None:
    """Main entry point."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Goodbye![/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()