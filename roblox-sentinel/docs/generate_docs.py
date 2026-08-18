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
Script to automatically generate RST files for Sphinx documentation from the sentinel codebase.

This script crawls the sentinel source code structure and generates appropriate RST files
for documenting modules, classes, and functions to be used with Sphinx.
"""

from pathlib import Path


def generate_rst_files(module_path, output_dir, package_name="sentinel"):
    """Generate RST files for the given module path."""
    module_name = (
        package_name
        if module_path.name == "sentinel"
        else f"{package_name}.{module_path.name}"
    )

    # Create the modules.rst file if it doesn't exist
    modules_file = output_dir / "modules.rst"
    if not modules_file.exists():
        with open(modules_file, "w") as f:
            f.write(
                f"""API Reference
============

.. toctree::
   :maxdepth: 4

   {module_name}

"""
            )

    # Create the module.rst file
    with open(output_dir / f"{module_name}.rst", "w") as f:
        f.write(
            f"""{module_name}
{'=' * len(module_name)}

.. automodule:: {module_name}
   :members:
   :undoc-members:
   :show-inheritance:

Submodules
----------


"""  # Note the extra newlines here
        )

        # Find all Python files and generate submodule entries
        submodules = []
        subpackages = []

        for item in module_path.iterdir():
            if item.is_file() and item.suffix == ".py" and item.name != "__init__.py":
                submodule_name = f"{module_name}.{item.stem}"
                submodules.append((submodule_name, item.stem))
            elif item.is_dir() and (item / "__init__.py").exists():
                subpackage_name = f"{module_name}.{item.name}"
                subpackages.append((subpackage_name, item.name))
                # Recursively generate RST for subpackages
                generate_rst_files(Path(str(module_path) + "/" + item.name), output_dir)

        # Add submodule sections
        for submodule_name, stem in sorted(submodules):
            f.write(
                f"""{module_name}.{stem}
{'-' * len(f"{module_name}.{stem}")}

.. automodule:: {submodule_name}
   :members:
   :undoc-members:
   :show-inheritance:

"""
            )

        # Add subpackage references
        if subpackages:
            f.write(
                """Subpackages
----------

.. toctree::
   :maxdepth: 4

"""
            )
            for _, subpackage in sorted(subpackages):
                f.write(f"   {module_name}.{subpackage}\n")


def main():
    """Run the documentation generation process for sentinel.

    This function sets up paths and calls the RST file generator for the project.
    """
    # Paths
    source_dir = Path("docs/source")
    module_path = Path("src/sentinel")

    print("Generating RST files for documentation...")

    # Generate RST files
    generate_rst_files(module_path, source_dir, package_name="sentinel")

    print(f"Documentation RST files successfully generated in {source_dir}")
    print("To build the HTML documentation, run: cd docs && make html")


if __name__ == "__main__":
    main()
