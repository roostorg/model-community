# Testing Sentinel

This document provides guidance on testing the Sentinel library.

## Running Tests

Sentinel uses pytest for testing. To run the tests, use:

```bash
# Run all tests
poetry run pytest

# Run tests with coverage report
poetry run pytest --cov=sentinel

# Run a specific test file
poetry run pytest tests/test_sentinel_local_index.py

# Run tests matching a specific pattern
poetry run pytest -k "test_calculate"
```

## Test Categories

The tests are organized into several categories:

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test multiple components working together (run by default)
3. **S3 Tests**: Test S3 functionality (disabled by default)

To run just the integration tests:

```bash
poetry run pytest -m integration
```

To run S3 tests (requires valid AWS credentials):

```bash
S3_TEST_ENABLED=1 poetry run pytest -m s3
```

## Test Coverage

To generate a detailed coverage report:

```bash
poetry run pytest --cov=sentinel --cov-report=html
```

This will create an HTML coverage report in the `htmlcov` directory.

## Writing New Tests

When adding new features or fixing bugs, please add appropriate tests:

1. **Unit Tests**: Place in `tests/test_*.py` files
2. **Integration Tests**: Mark with `@pytest.mark.integration`
3. **S3 Tests**: Mark with `@pytest.mark.s3`

Use fixtures from `tests/conftest.py` where possible to maintain consistency.

## Continuous Integration

Tests run automatically on GitHub Actions for all pull requests and pushes to main.
All tests must pass before a PR can be merged.
