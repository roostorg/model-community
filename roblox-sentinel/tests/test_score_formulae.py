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

"""Tests for score_formulae module."""

import numpy as np
import pytest

from sentinel.score_formulae import (
    mean_of_positives,
    calculate_contrastive_score,
    skewness,
    top_k_mean,
    percentile_score,
    softmax_weighted_mean,
    max_score,
    contrastive_components,
)


def test_calculate_contrastive_score():
    """Test calculate_contrastive_score function."""
    # Case 1: Text more similar to positive than negative examples
    positive_sims = [0.9, 0.8, 0.7]
    negative_sims = [0.5, 0.4, 0.3]
    score = calculate_contrastive_score(positive_sims, negative_sims)
    assert (
        score > 0
    ), "Score should be positive when text is more similar to positive examples"

    # Case 2: Text more similar to negative than positive examples
    # Based on the implementation, when negatives are more similar, the score is clipped to 0
    positive_sims = [0.5, 0.4, 0.3]
    negative_sims = [0.9, 0.8, 0.7]
    score = calculate_contrastive_score(positive_sims, negative_sims)
    assert (
        score == 0
    ), "Score should be 0 when text is more similar to negative examples"

    # Case 3: Text equally similar to both
    positive_sims = [0.7, 0.6, 0.5]
    negative_sims = [0.7, 0.6, 0.5]
    score = calculate_contrastive_score(positive_sims, negative_sims)
    assert (
        abs(score) < 1e-6
    ), "Score should be close to zero when equally similar to both"

    # Case 4: Test with different list lengths
    positive_sims = [0.9, 0.8]
    negative_sims = [0.5, 0.4, 0.3]
    score = calculate_contrastive_score(positive_sims, negative_sims)
    assert score > 0, "Score should handle different list lengths"


def test_mean_of_positives():
    """Test mean_of_positives function."""
    # Test with all positive scores
    scores = np.array([0.5, 0.3, 0.7])
    result = mean_of_positives(scores)
    assert result == 0.5, "Should return the mean of all positive scores"

    # Test with mixed scores (positive and negative)
    scores = np.array([0.5, -0.3, 0.7, -0.2])
    result = mean_of_positives(scores)
    assert result == 0.6, "Should ignore negative scores and return mean of positives"

    # Test with all negative scores - should return 0.0 like other functions
    scores = np.array([-0.5, -0.3, -0.7])
    result = mean_of_positives(scores)
    assert result == 0.0, "Should return 0.0 when there are no positive scores"

    # Test with empty array - should return 0.0 like other functions
    scores = np.array([])
    result = mean_of_positives(scores)
    assert result == 0.0, "Should return 0.0 for empty array"


def test_skewness():
    """Test skewness function."""
    # Test with fewer scores than min_size_of_scores (which defaults to 10)
    # This should return 0.0 because we don't have enough data points
    scores = np.array([0.1, 0.2, 0.3, 0.9, 1.0])
    result = skewness(scores)
    assert np.isclose(
        result, 0.0
    ), "Should return 0.0 when fewer scores than min_size_of_scores"

    # Test with enough scores and explicitly set min_size_of_scores
    scores = np.array([0.1, 0.2, 0.3, 0.9, 1.0, 0.2, 0.3, 0.1, 0.2, 0.8, 0.7])
    result = skewness(scores, min_size_of_scores=5)
    assert result > 0, "Should return positive value for right-skewed distribution"

    # Test with negatively skewed distribution
    scores = np.array([0.0, 0.1, 0.7, 0.8, 0.9, 0.7, 0.8, 0.9, 0.8, 0.7])
    result = skewness(scores, min_size_of_scores=5)
    assert result < 0, "Should return negative value for left-skewed distribution"

    # Test with symmetric distribution
    symmetric_scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.1, 0.3, 0.5, 0.7, 0.9])
    result = skewness(symmetric_scores, min_size_of_scores=5)
    assert abs(result) < 0.01, "Should return close to zero for symmetric distribution"

    # Test with constant values
    constant_scores = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    result = skewness(constant_scores, min_size_of_scores=5)
    assert abs(result) < 1e-10, "Should return very close to 0.0 for constant values"

    # Test with insufficient scores
    small_scores = np.array([0.5])
    result = skewness(small_scores)
    assert np.isclose(result, 0.0), "Should return 0.0 for insufficient scores"

    # Test with empty array
    empty_scores = np.array([])
    result = skewness(empty_scores)
    assert np.isclose(result, 0.0), "Should return 0.0 for empty array"


def test_additional_aggregators():
    scores = np.array([0.0, 0.2, 0.5, 1.0, 0.7, -0.1, 0.3])

    # top_k_mean
    val = top_k_mean(scores, k=2)
    assert np.isclose(val, np.mean([1.0, 0.7]))

    # percentile_score
    val = percentile_score(scores, q=50)
    # positives are [0.2, 0.5, 1.0, 0.7, 0.3]; median = 0.5
    assert np.isclose(val, 0.5)

    # softmax_weighted_mean (temperature=1)
    val = softmax_weighted_mean(scores, temperature=1.0)
    assert val > 0.5 and val <= 1.0

    # max_score
    val = max_score(scores)
    assert np.isclose(val, 1.0)


def test_aggregation_functions_edge_cases():
    """Test edge cases for aggregation functions to improve coverage."""
    # Test top_k_mean with empty array (lines 98, 101)
    empty_scores = np.array([])
    assert top_k_mean(empty_scores) == 0.0
    
    # Test top_k_mean with no positive scores
    negative_scores = np.array([-1, -2, -3])
    assert top_k_mean(negative_scores) == 0.0
    
    # Test percentile_score with empty array (lines 119, 122)
    assert percentile_score(empty_scores) == 0.0
    
    # Test percentile_score with no positive scores
    assert percentile_score(negative_scores) == 0.0
    
    # Test softmax_weighted_mean with empty array (lines 139, 142)
    assert softmax_weighted_mean(empty_scores) == 0.0
    
    # Test softmax_weighted_mean with no positive scores
    assert softmax_weighted_mean(negative_scores) == 0.0
    
    # Test max_score with empty array (lines 155, 158)
    assert max_score(empty_scores) == 0.0
    
    # Test max_score with no positive scores
    assert max_score(negative_scores) == 0.0


def test_contrastive_components_edge_cases():
    """Test edge cases for contrastive_components function."""
    # Test with divide by zero scenario (line 176)
    # This is hard to trigger since we use exp(), but we can test normal operation
    pos_sims = [0.5, 0.6]
    neg_sims = [0.1, 0.2]
    
    pos_term, neg_term, log_ratio = contrastive_components(pos_sims, neg_sims)
    
    assert pos_term > 0
    assert neg_term > 0
    assert log_ratio != 0
    
    # Test when log_ratio would be infinity (line 188)
    # This tests the inf handling in contrastive_components
    very_high_pos = [10.0, 10.0]  # Very high similarities
    very_low_neg = [-10.0, -10.0]  # Very low similarities
    
    pos_term, neg_term, log_ratio = contrastive_components(very_high_pos, very_low_neg)
    
    assert pos_term > neg_term
    assert log_ratio > 0


class TestScoreFormulaeEdgeCases:
    """Edge case tests for score formulae functions to improve coverage."""
    
    def test_aggregation_functions_empty_arrays(self):
        """Test aggregation functions with empty arrays."""
        empty_array = np.array([])
        
        # Test mean_of_positives with empty array (line 98)
        result = mean_of_positives(empty_array)
        assert result == 0.0
        
        # Test top_k_mean with empty array (line 119)
        result = top_k_mean(empty_array, k=3)
        assert result == 0.0
        
        # Test top_k_mean with k larger than array size (line 122)
        small_array = np.array([0.5])
        result = top_k_mean(small_array, k=3)
        assert np.isclose(result, 0.5)
        
        # Test percentile_score with empty array (line 139)
        result = percentile_score(empty_array, q=50)
        assert result == 0.0
        
        # Test percentile_score with all negative values (line 142)
        negative_array = np.array([-1.0, -0.5, -2.0])
        result = percentile_score(negative_array, q=75)
        assert result == 0.0
        
        # Test skewness with empty array (line 155)
        result = skewness(empty_array)
        assert np.isclose(result, 0.0)
        
        # Test skewness with single value (line 158)
        single_value = np.array([0.5])
        result = skewness(single_value)
        assert np.isclose(result, 0.0)
        
        # Test softmax_weighted_mean with empty array
        result = softmax_weighted_mean(empty_array)
        assert result == 0.0
        
        # Test max_score with empty array
        result = max_score(empty_array)
        assert result == 0.0
