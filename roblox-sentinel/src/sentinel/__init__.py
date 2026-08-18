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
Sentinel - Semantic Ratio for Improved Recognition And Classification of Hurtful or Antagonistic content.

This library provides tools for semantic scoring of text based on contrastive learning principles.
"""

from sentinel.sentinel_local_index import SentinelLocalIndex
from sentinel.score_formulae import (
    calculate_contrastive_score,
    skewness,
    mean_of_positives,
    top_k_mean,
    percentile_score,
    softmax_weighted_mean,
    max_score,
)
from sentinel.simulation import (
    DEFAULT_AGGREGATORS,
    LabeledGroup,
    GroupObservationScores,
    encode_observations,
    score_groups,
    evaluate_groups,
    compare_aggregators,
    run_grid_search,
)

# Kept in step with the version in pyproject.toml, which is the source of truth for
# packaging. Duplicated here because importing package metadata at runtime fails when
# the library is used straight from a source checkout rather than an installed wheel.
__version__ = "2.0.0"

__all__ = [
    "__version__",
    "SentinelLocalIndex",
    "calculate_contrastive_score",
    "skewness",
    "mean_of_positives",
    "top_k_mean",
    "percentile_score",
    "softmax_weighted_mean",
    "max_score",
    "DEFAULT_AGGREGATORS",
    "LabeledGroup",
    "GroupObservationScores",
    "encode_observations",
    "score_groups",
    "evaluate_groups",
    "compare_aggregators",
    "run_grid_search",
]
