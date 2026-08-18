# Contributing to sentinel

Thank you for your interest in contributing to sentinel! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with the following information:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Environment information (OS, Python version, package version)
- Any additional context or screenshots

### Suggesting Enhancements

We welcome suggestions for improvements! Please include:

- A clear, descriptive title
- A detailed description of the proposed enhancement
- Any potential implementation ideas you have
- Why this enhancement would be useful to users

### Pull Requests

1. Fork the repository
2. Create a new branch for your feature or bugfix
3. Make your changes
4. Run tests and ensure they pass
5. Add or update relevant tests
6. Add or update documentation
7. Submit a pull request

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/sentinel.git
   cd sentinel
   ```

2. Set up Poetry:
   ```bash
   poetry install
   poetry install --extras=sbert
   poetry env activate
   ```

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install development dependencies:
   ```bash
   pip install -e ".[sbert]"
   pip install -e ".[dev]"
   ```

4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Coding Standards

- Follow PEP 8 style guidelines
- Write docstrings for all functions, classes, and modules
- Add type hints to function signatures
- Write unit tests for new functionality

## Testing

- Run tests with pytest:
  ```bash
  pytest
  ```

- Check code coverage:
  ```bash
  pytest --cov=sentinel tests/
  ```

## Documentation

- Update documentation for any changes to the API
- Document new features with examples
- Keep the README updated with any major changes
- Always update docstrings when changing function behavior or parameters

### Automatic Documentation Updates

The project has automated tools to keep documentation in sync with code:

1. A pre-commit hook will automatically update RST files when docstrings change
2. You can manually update docs by running (from the project root):
   ```bash
   # From the project root
   python docs/generate_docs.py
   ```

3. To build the HTML documentation (from the docs directory):
   ```bash
   # From the docs directory
   make html
   ```

4. To both generate and build documentation in one step (from the docs directory):
   ```bash
   # From the docs directory
   make update
   ```

The CI pipeline will verify that documentation is up-to-date when pull requests are opened or updated. The pre-commit hook ensures that documentation stays in sync with your code changes automatically.

## Release Process

The maintainers will handle releases following semantic versioning.

## Questions?

If you have questions about contributing, please open an issue or reach out to the maintainers.

Thank you for contributing to Sentinel!
