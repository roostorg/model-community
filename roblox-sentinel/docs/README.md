# Sentinel Documentation

This directory contains the source files for Sentinel's documentation.

## Building Documentation Locally

To build the documentation locally:

1. Make sure you have installed the project with documentation dependencies (run from the **project root**):

```bash
# From the project root
poetry install --with docs
```

2. Generate the RST files for the modules (run from the **project root**):

```bash
# From the project root
poetry run python docs/generate_docs.py
```

3. Build the HTML documentation (run from the **docs directory**):

```bash
# From the docs directory
poetry run make html
```

4. Open `docs/build/html/index.html` in your web browser to view the documentation.

## Simplified Workflow

You can now use the Makefile to do both steps at once. Run from the **docs directory**:

```bash
# From the docs directory
poetry run make update  # Generates RST files and builds HTML in one step
```

## Verifying Documentation

The documentation is kept in sync with the codebase automatically via the pre-commit hook.
This ensures that documentation is always updated when code changes are committed.

For continuous integration, GitHub Actions will verify that documentation is up to date
on pull requests and deploy the updated HTML documentation when changes are merged to main.

## Automatic Documentation Updates

A pre-commit hook has been added that automatically updates documentation when docstrings change.
This hook will run automatically from any directory when you commit changes.

When you commit changes to Python files, the hook will:

1. Check if docstrings have changed
2. Update the RST files if necessary
3. Stage the updated documentation files

## Documentation Structure

- `source/conf.py`: Sphinx configuration file
- `source/index.rst`: Main entry point for the documentation
- `source/*.rst`: Auto-generated RST files for modules and submodules
- `build/`: Directory where the built documentation is stored
- `generate_docs.py`: Script to generate RST files from source code
- `update_docs_hook.py`: Pre-commit hook script to update docs automatically

## Command Reference

Below is a quick reference for all documentation-related commands:

| Command | Run From | Description |
|---------|----------|-------------|
| `poetry install --with docs` | Project Root | Install documentation dependencies |
| `poetry run python docs/generate_docs.py` | Project Root | Generate RST files from docstrings |
| `poetry run make html` | Docs Directory | Build HTML documentation |
| `poetry run make update` | Docs Directory | Generate RST files AND build HTML docs |
| `poetry run make clean html` | Docs Directory | Clean and rebuild documentation from scratch |
