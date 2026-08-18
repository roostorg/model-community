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

"""Tests for the embeddings module."""

import pytest
from unittest.mock import patch, MagicMock

from sentinel.embeddings.sbert import (
    get_sentence_transformer_and_scaling_fn,
    clear_model_cache,
    get_cache_info,
    remove_from_cache,
)


class TestSBERT:
    """Test suite for SBERT embeddings functionality."""

    @patch("sentinel.embeddings.sbert.SentenceTransformer")
    def test_get_sentence_transformer_basic(self, mock_transformer):
        """Test basic functionality of get_sentence_transformer_and_scaling_fn."""
        # Setup mock
        mock_instance = MagicMock()
        mock_transformer.return_value = mock_instance

        # Call the function
        model_name = "all-MiniLM-L6-v2"
        returned_model, scale_fn = get_sentence_transformer_and_scaling_fn(model_name)

        # Verify model was loaded correctly
        mock_transformer.assert_called_once_with(model_name)
        assert returned_model == mock_instance

        # Only E5 models have scaling functions
        assert scale_fn is None

    @patch("sentinel.embeddings.sbert.SentenceTransformer")
    def test_get_sentence_transformer_custom_model(self, mock_transformer):
        """Test get_sentence_transformer_and_scaling_fn with custom model."""
        # Setup mock
        mock_instance = MagicMock()
        mock_transformer.return_value = mock_instance

        # Call the function with a custom model name
        model_name = "custom-model-path"
        model, scale_fn = get_sentence_transformer_and_scaling_fn(model_name)

        # Verify model was loaded correctly
        mock_transformer.assert_called_once_with(model_name)
        assert model == mock_instance

        # For unknown models, no scaling function should be returned
        assert scale_fn is None

    @pytest.mark.parametrize(
        "model_name,expected_scaling",
        [
            ("e5-small", True),  # E5 model with scaling
            ("e5-base", True),  # E5 model with scaling
            ("all-MiniLM-L6-v2", False),  # Non-E5 model without scaling
            ("custom-model", False),  # Unknown model without scaling
        ],
    )
    @patch("sentinel.embeddings.sbert.SentenceTransformer")
    def test_model_specific_scaling(
        self, mock_transformer, model_name, expected_scaling
    ):
        """Test scaling functions for different models."""
        # Setup mock
        mock_instance = MagicMock()
        mock_transformer.return_value = mock_instance

        # Call the function
        model, scale_fn = get_sentence_transformer_and_scaling_fn(model_name)

        # Check if scaling function is returned as expected
        if expected_scaling:
            assert scale_fn is not None
            # Test with various similarity values
            for sim in [0.1, 0.5, 0.9]:
                scaled = scale_fn(sim)
                assert 0.0 <= scaled <= 1.0
        else:
            assert scale_fn is None

    @patch("sentinel.embeddings.sbert.SentenceTransformer")
    def test_model_caching(self, mock_transformer):
        """Test that model caching works correctly."""
        # Setup mock
        mock_instance = MagicMock()
        mock_transformer.return_value = mock_instance
        
        # Clear cache to start fresh
        clear_model_cache()
        
        model_name = "test-model"
        
        # First call should create the model
        model1, scale_fn1 = get_sentence_transformer_and_scaling_fn(model_name, use_cache=True)
        assert mock_transformer.call_count == 1
        
        # Second call should use cached model
        model2, scale_fn2 = get_sentence_transformer_and_scaling_fn(model_name, use_cache=True)
        assert mock_transformer.call_count == 1  # Still 1, no new creation
        assert model1 is model2  # Same model instance
        
        # Test cache info
        cache_info = get_cache_info()
        assert model_name in cache_info["cached_models"]
        assert cache_info["cache_size"] == 1
        
        # Test removing from cache
        assert remove_from_cache(model_name) is True
        assert remove_from_cache(model_name) is False  # Already removed
        
        # Test with caching disabled
        model3, scale_fn3 = get_sentence_transformer_and_scaling_fn(model_name, use_cache=False)
        assert mock_transformer.call_count == 2  # New creation
        
        # Test clearing cache
        get_sentence_transformer_and_scaling_fn("another-model", use_cache=True)
        clear_model_cache()
        cache_info_after_clear = get_cache_info()
        assert cache_info_after_clear["cache_size"] == 0
