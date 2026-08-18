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

"""Shared test fixtures and configurations for Sentinel tests."""

import os
import sys
import pathlib
import pytest
import numpy as np
import torch
import logging
from unittest.mock import MagicMock


# Ensure the package under src/ is importable without installation
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_PATH = _REPO_ROOT / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))


# Set up logging for tests
@pytest.fixture(autouse=True)
def setup_logging():
    """Set up logging for all tests."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    yield
    # Reset logging after test
    logging.getLogger().handlers = []


@pytest.fixture
def random_embeddings():
    """Generate random embeddings for testing."""

    def _generate(n_samples, dim=384, seed=42):
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        embeddings = rng.standard_normal((n_samples, dim))
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        return torch.tensor(embeddings, dtype=torch.float32)

    return _generate


@pytest.fixture
def mock_sentence_transformer():
    """Create a mock SentenceTransformer for testing."""
    mock = MagicMock()

    # Set up the encode method to return predictable embeddings
    def mock_encode(texts, **kwargs):
        # Create deterministic embeddings for testing
        # This makes texts with similar words have similar embeddings
        unique_words = set()
        for text in texts:
            unique_words.update(text.lower().split())
        word_to_idx = {word: i for i, word in enumerate(sorted(unique_words))}

        embeddings = []
        embedding_size = 4  # Small embedding for testing

        for text in texts:
            # Create a simple embedding based on word presence
            embedding = np.zeros(embedding_size)
            words = text.lower().split()
            for i, word in enumerate(words):
                idx = word_to_idx[word] % embedding_size
                embedding[idx] += 1.0 / (
                    1 + i
                )  # Words earlier in text have more weight

            # Normalize embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            embeddings.append(embedding)

        return np.array(embeddings)

    mock.encode.side_effect = mock_encode
    return mock


# Skip tests requiring S3 access unless explicitly enabled
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "s3: mark test as requiring S3 access")
    config.addinivalue_line("markers", "integration: mark test as an integration test")


def pytest_runtest_setup(item):
    """Skip tests based on markers unless explicitly enabled."""
    # Skip S3 tests unless S3_TEST_ENABLED environment variable is set
    if "s3" in item.keywords and not os.environ.get("S3_TEST_ENABLED"):
        pytest.skip(
            "S3 tests are disabled. Set S3_TEST_ENABLED environment variable to run them."
        )

    # Integration tests run by default
    # Uncomment to enable skipping integration tests with an environment variable
    # if "integration" in item.keywords and os.environ.get("SKIP_INTEGRATION_TESTS"):
    #     pytest.skip("Integration tests are skipped. Unset SKIP_INTEGRATION_TESTS environment variable to run them.")
