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
Script to check if documentation is in sync with the codebase.

This script generates temporary documentation and compares it with existing
documentation to ensure they match. If they don't match, it means docstrings
have been changed but the documentation hasn't been updated.
"""

import sys
import filecmp
import tempfile
import shutil
from pathlib import Path


def check_docs_sync(source_dir, module_path, package_name="sentinel"):
    """
    Check if documentation is in sync with the codebase.

    Args:
        source_dir: Path to the documentation source directory
        module_path: Path to the module to check
        package_name: Name of the package

    Returns:
        bool: True if docs are in sync, False otherwise
    """
    # Create a temporary directory to store generated docs
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Copy existing conf.py and other necessary files
        if (source_dir / "conf.py").exists():
            shutil.copy(source_dir / "conf.py", temp_path / "conf.py")

        # Import the generate_rst_files function from generate_docs.py
        sys.path.insert(0, str(Path(__file__).parent))
        from generate_docs import generate_rst_files

        # Generate documentation in temporary directory
        generate_rst_files(module_path, temp_path, package_name)

        # Check if all rst files exist in the source directory
        all_synced = True
        for rst_file in temp_path.glob("*.rst"):
            source_file = source_dir / rst_file.name

            # If source file doesn't exist, docs are out of sync
            if not source_file.exists():
                print(f"Missing documentation file: {source_file}")
                all_synced = False
                continue

            # Compare the content of the files
            if not filecmp.cmp(rst_file, source_file, shallow=False):
                print(f"Documentation out of sync: {source_file}")
                all_synced = False

        return all_synced


def main():
    """Run the documentation sync check for Sentinel."""
    # Paths
    source_dir = Path("docs/source")
    module_path = Path("src/sentinel")

    # Check if docs are in sync
    is_synced = check_docs_sync(source_dir, module_path)

    if not is_synced:
        print("Documentation is out of sync with the codebase.")
        print("Please run 'python docs/generate_docs.py' to update.")
        sys.exit(1)

    print("Documentation is in sync with the codebase.")
    sys.exit(0)


if __name__ == "__main__":
    main()
