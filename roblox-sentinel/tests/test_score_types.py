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

"""Tests for score_types module."""

from sentinel.score_types import RareClassAffinityResult


def test_rare_class_affinity_result_init():
    """Test initialization of RareClassAffinityResult."""
    # Test with valid data
    score = 0.75
    observation_scores = {"text1": 0.8, "text2": 0.7}
    result = RareClassAffinityResult(
        rare_class_affinity_score=score, observation_scores=observation_scores
    )

    assert result.rare_class_affinity_score == score
    assert result.observation_scores == observation_scores

    # Test with empty observation scores
    result = RareClassAffinityResult(
        rare_class_affinity_score=0.0, observation_scores={}
    )

    assert result.rare_class_affinity_score == 0.0
    assert result.observation_scores == {}
