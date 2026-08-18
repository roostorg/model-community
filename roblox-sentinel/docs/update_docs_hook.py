#!/usr/bin/env python
# Copyright 2025 Roblox Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A pre-commit script for updating documentation from docstrings.

This script is designed to be run as a pre-commit hook to ensure documentation
is updated when docstrings change.
"""

import os
import sys
from pathlib import Path


def main():
    """Run the documentation generation script and check if any files changed."""
    # Get the repository root directory
    repo_root = os.getcwd()

    # Set up paths
    docs_dir = Path(repo_root) / "docs"
    gen_script = docs_dir / "generate_docs.py"

    # Generate documentation
    print("Updating documentation from docstrings...")
    import subprocess

    result = subprocess.run(
        [sys.executable, str(gen_script)], capture_output=True, text=True, cwd=repo_root
    )

    if result.returncode != 0:
        print("Error generating documentation:")
        print(result.stderr)
        return 1

    # Check if any files were modified
    git_status = subprocess.run(
        ["git", "status", "--porcelain", "docs/source/*.rst"],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )

    modified_files = git_status.stdout.strip()
    if modified_files:
        print("\nDocumentation files were updated:")
        print(modified_files)
        print("\nPlease stage these changes before committing.")

        # Stage the changes automatically to make it easier for users
        subprocess.run(["git", "add", "docs/source/*.rst"], cwd=repo_root)

        print(
            "Changes have been automatically staged. Please verify them before committing."
        )
    else:
        print("Documentation is up to date.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
