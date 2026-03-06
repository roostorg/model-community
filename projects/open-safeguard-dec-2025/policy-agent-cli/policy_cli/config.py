"""Configuration management for Policy CLI."""

from __future__ import annotations

import os
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

from policy_cli.constants import CLI_MODEL, DEFAULT_TEMPERATURE

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for Policy CLI."""

    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir) if config_dir else Path.home() / ".policy-cli"
        self.config_file = self.config_dir / "config.json"
        self._config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from file."""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    self._config = json.load(f)
            else:
                self._config = self._get_default_config()
                self._save_config()
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
            self._config = self._get_default_config()

    def _save_config(self) -> None:
        """Save configuration to file."""
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self._config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "api": {
                "provider": "openai",
                "model": CLI_MODEL,
                "temperature": DEFAULT_TEMPERATURE,
                "max_tokens": None,
                "api_key": None
            },
            "tools": {
                "enabled": [
                    "read_file",
                    "write_file",
                    "list_directory",
                    "glob_files",
                    "edit_file",
                    "shell",
                    "grep",
                    "web_fetch",
                    "run_policy",
                    "modify_policy"
                ],
                "safe_tools": [
                    "read_file",
                    "list_directory",
                    "glob_files",
                    "grep",
                    "web_fetch",
                    "analyze_url",
                    "run_policy",
                    "modify_policy"
                ]
            },
            "agents": {
                "max_turns": 50,
                "timeout_seconds": 300,
                "default_model": CLI_MODEL
            },
            "mcp_servers": {},
            "ui": {
                "color": True,
                "verbose": False,
                "show_tokens": False
            },
            "workspace": {
                "auto_detect": True,
                "trusted_folders": []
            }
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation."""
        keys = key.split('.')
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value
        self._save_config()

    def get_api_key(self) -> Optional[str]:
        """Get API key from config or environment."""
        # First check config
        api_key = self.get("api.api_key")
        if api_key:
            return api_key

        # Then check environment variables
        for env_var in ["OPENAI_API_KEY"]:
            api_key = os.getenv(env_var)
            if api_key:
                return api_key

        return None

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "provider": self.get("api.provider", "openai"),
            "model": self.get("api.model", CLI_MODEL),
            "temperature": self.get("api.temperature", DEFAULT_TEMPERATURE),
            "max_tokens": self.get("api.max_tokens"),
            "api_key": self.get_api_key()
        }

    def get_enabled_tools(self) -> List[str]:
        """Get list of enabled tools."""
        return self.get("tools.enabled", [])

    def get_safe_tools(self) -> List[str]:
        """Get list of tools that don't require confirmation."""
        return self.get("tools.safe_tools", [])

    def get_mcp_servers(self) -> Dict[str, Any]:
        """Get MCP server configurations."""
        return self.get("mcp_servers", {})

    def is_workspace_trusted(self, path: str) -> bool:
        """Check if a workspace path is trusted."""
        trusted_folders = self.get("workspace.trusted_folders", [])
        abs_path = os.path.abspath(path)

        for trusted in trusted_folders:
            trusted_abs = os.path.abspath(trusted)
            if abs_path.startswith(trusted_abs):
                return True

        return False

    def add_trusted_folder(self, path: str) -> None:
        """Add a folder to trusted list."""
        trusted_folders = self.get("workspace.trusted_folders", [])
        abs_path = os.path.abspath(path)

        if abs_path not in trusted_folders:
            trusted_folders.append(abs_path)
            self.set("workspace.trusted_folders", trusted_folders)

    def get_agent_config(self) -> Dict[str, Any]:
        """Get agent configuration."""
        return {
            "max_turns": self.get("agents.max_turns", 50),
            "timeout_seconds": self.get("agents.timeout_seconds", 300),
            "default_model": self.get("agents.default_model", CLI_MODEL)
        }

    def setup_first_time(self) -> bool:
        """Setup configuration for first-time users."""
        print("Welcome to Policy CLI! Let's set up your configuration.")

        # Check for API key
        api_key = self.get_api_key()
        if not api_key:
            print("\nYou need an OpenAI API key to use Policy CLI.")
            print("Get one at: https://platform.openai.com/api-keys")

            while True:
                api_key = input("Enter your API key (or press Enter to set it later): ").strip()
                if api_key:
                    self.set("api.api_key", api_key)
                    break
                elif input("Continue without API key? (y/N): ").lower() == 'y':
                    break

        # Ask about workspace trust
        cwd = os.getcwd()
        if input(f"\nTrust current directory ({cwd}) for tool execution? (Y/n): ").lower() != 'n':
            self.add_trusted_folder(cwd)

        # Ask about verbose mode
        if input("\nEnable verbose output? (y/N): ").lower() == 'y':
            self.set("ui.verbose", True)

        print("\nConfiguration saved! You can modify it later with 'policy config'")
        return True

    def show_config(self) -> None:
        """Display current configuration."""
        print("Current Policy CLI Configuration:")
        print("=" * 40)

        # API settings
        print("\nAPI Settings:")
        print(f"  Provider: {self.get('api.provider')}")
        print(f"  Model: {self.get('api.model')}")
        print(f"  Temperature: {self.get('api.temperature')}")
        print(f"  API Key: {'Set' if self.get_api_key() else 'Not set'}")

        # Tools
        enabled_tools = self.get_enabled_tools()
        print(f"\nEnabled Tools ({len(enabled_tools)}):")
        for tool in enabled_tools:
            print(f"  - {tool}")

        # MCP Servers
        mcp_servers = self.get_mcp_servers()
        print(f"\nMCP Servers ({len(mcp_servers)}):")
        for name, config in mcp_servers.items():
            print(f"  - {name}: {config.get('command', 'Unknown command')}")

        # Trusted folders
        trusted_folders = self.get("workspace.trusted_folders", [])
        print(f"\nTrusted Folders ({len(trusted_folders)}):")
        for folder in trusted_folders:
            print(f"  - {folder}")

    def export_config(self, file_path: str) -> None:
        """Export configuration to a file."""
        try:
            with open(file_path, 'w') as f:
                json.dump(self._config, f, indent=2)
            print(f"Configuration exported to {file_path}")
        except Exception as e:
            print(f"Failed to export configuration: {e}")

    def import_config(self, file_path: str) -> None:
        """Import configuration from a file."""
        try:
            with open(file_path, 'r') as f:
                imported_config = json.load(f)

            # Merge with existing config
            self._config.update(imported_config)
            self._save_config()
            print(f"Configuration imported from {file_path}")
        except Exception as e:
            print(f"Failed to import configuration: {e}")