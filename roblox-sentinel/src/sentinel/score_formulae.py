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

"""Score calculation functions for Sentinel index.

This module contains per-observation scoring utilities (contrastive scoring)
and aggregation functions to combine multiple observation scores into a single
affinity number. In addition to the default skewness, a set of robust
alternatives are provided to fit different deployment preferences (recall vs precision,
stability vs sensitivity, etc.).
"""

import numpy as np
from typing import List, Callable


def mean_of_positives(scores: np.array) -> float:
    """Calculate the mean of positive contrastive scores across multiple observations.

    This function aggregates individual observation scores, focusing only on positive scores
    (observations that were more similar to rare class examples than common class examples).

    Unlike skewness, this aggregation method directly measures the average strength of rare class
    similarity but may be more sensitive to the number of observations. It's useful when you want
    to focus on the magnitude of similarity rather than the pattern across observations.

    Args:
        scores: Array of contrastive scores from multiple observations

    Returns:
        Mean of positive scores, indicating overall affinity to rare class content
    """
    if scores.size == 0:
        return 0.0
    positives = scores[scores > 0]
    if positives.size == 0:
        return 0.0
    return float(np.mean(positives))


def skewness(scores: np.array, min_size_of_scores: int = 10) -> float:
    """Calculate the skewness of contrastive scores to detect patterns of rare class content.

    Skewness measures the asymmetry in the distribution of contrastive scores across multiple observations.
    It is particularly effective for rare class detection because:

    1. It focuses on the pattern of scores rather than their quantity
    2. It's sensitive to occasional spikes in similarity that might indicate rare events
    3. It's robust to varying numbers of observations (e.g., recent chat volumes)
    4. It can work with a relatively small number of recent observations

    As a high-recall metric, a positive skewness suggests that while most observations are neutral/common,
    there are enough rare-class observations to create a right-skewed distribution.
    This makes it ideal for generating candidates for further investigation without being affected
    by the total volume of observations.

    Args:
        scores: Array of contrastive scores from multiple observations
        min_size_of_scores: Minimum number of scores required to calculate meaningful skewness

    Returns:
        Skewness value, where positive values suggest patterns of rare class content
    """
    if len(scores) < min_size_of_scores:
        return 0.0
    mean = np.mean(scores)
    median = np.median(scores)
    std = np.std(scores)
    if std == 0:
        return 0.0
    return (mean - median) / std


def top_k_mean(scores: np.array, k: int = 3) -> float:
    """Mean of the top-k positive scores.

    Focuses on the strongest signals while ignoring noise and negatives.

    Args:
        scores: Array of observation scores.
        k: Number of highest positive scores to average.

    Returns:
        Mean of the top-k positive scores (0.0 if no positive scores).
    """
    if scores.size == 0:
        return 0.0
    positives = scores[scores > 0]
    if positives.size == 0:
        return 0.0
    k = min(k, positives.size)
    # Use partition for efficiency, then mean of the largest k
    idx = np.argpartition(positives, -k)[-k:]
    return float(np.mean(positives[idx]))


def percentile_score(scores: np.array, q: float = 90.0) -> float:
    """Return the q-th percentile among positive scores (robust to outliers).

    Args:
        scores: Array of observation scores.
        q: Percentile in [0, 100].

    Returns:
        q-th percentile of positive scores (0.0 if no positive scores).
    """
    if scores.size == 0:
        return 0.0
    positives = scores[scores > 0]
    if positives.size == 0:
        return 0.0
    return float(np.percentile(positives, q))


def softmax_weighted_mean(scores: np.array, temperature: float = 1.0) -> float:
    """Softmax-weighted mean over positive scores.

    Emphasizes higher scores while keeping some contribution from smaller ones.

    Args:
        scores: Array of observation scores.
        temperature: Softmax temperature (>0). Lower values emphasize peaks more.

    Returns:
        Softmax-weighted average of positive scores (0.0 if no positive scores).
    """
    if scores.size == 0:
        return 0.0
    positives = scores[scores > 0]
    if positives.size == 0:
        return 0.0
    t = max(1e-6, float(temperature))
    x = positives / t
    # Numerical stability
    x = x - np.max(x)
    w = np.exp(x)
    w = w / np.sum(w)
    return float(np.sum(w * positives))


def max_score(scores: np.array) -> float:
    """Maximum positive score (simple, sensitive, and easy to interpret)."""
    if scores.size == 0:
        return 0.0
    positives = scores[scores > 0]
    if positives.size == 0:
        return 0.0
    return float(np.max(positives))


def contrastive_components(
    similarities_topk_pos: List[float],
    similarities_topk_neg: List[float],
    aggregation_fn: Callable[[np.array], float] = np.mean,
):
    """Return contrastive components and final log-ratio for a single observation.

    Computes the positive and negative terms used by the contrastive score and
    the unclipped log ratio. Useful for explainability.

    Returns:
        (positives_term, negatives_term, log_ratio)
    """
    if len(similarities_topk_pos) <= 0 or len(similarities_topk_neg) <= 0:
        raise ValueError(
            "The lists of similarities must have at least one element each."
        )

    similarities_topk_pos = np.array(similarities_topk_pos)
    similarities_topk_neg = np.array(similarities_topk_neg)

    positives_term = aggregation_fn(np.exp(similarities_topk_pos))
    negatives_term = aggregation_fn(np.exp(similarities_topk_neg))

    # Avoid divide-by-zero (shouldn’t happen with exp, but be safe)
    if negatives_term == 0:
        log_ratio = np.inf
    else:
        ratio = positives_term / negatives_term
        log_ratio = np.log(ratio)

    return float(positives_term), float(negatives_term), float(log_ratio)


def calculate_contrastive_score(
    similarities_topk_pos: List[float],
    similarities_topk_neg: List[float],
    aggregation_fn: Callable[[np.array], float] = np.mean,
) -> float:
    """Calculate a contrastive score for a single observation.

    This function uses a contrastive learning approach to determine how similar an observation
    is to examples of the rare class compared to examples of the common class. It computes
    a ratio between similarities to positive (rare class) examples and similarities to
    negative (common class) examples.

    These individual observation scores are later aggregated across multiple observations
    (e.g., messages, posts) using functions like `skewness` to identify patterns indicative
    of the rare class, regardless of the total number of observations.

    Args:
        similarities_topk_pos: List of similarities between the observation and rare class examples
        similarities_topk_neg: List of similarities between the observation and common class examples
        aggregation_fn: Function to aggregate similarity values within each category

    Returns:
        A contrastive score where values > 0 indicate closer similarity to rare class content
    """
    positives_term, negatives_term, log_ratio = contrastive_components(
        similarities_topk_pos, similarities_topk_neg, aggregation_fn
    )
    # Clip to zero to avoid negative scores, since we accumulate this score for all chat lines of a user.
    if log_ratio <= 0.0:
        return 0.0
    return float(log_ratio)
