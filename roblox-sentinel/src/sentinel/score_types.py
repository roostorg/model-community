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

"""Data types for rare class detection and scoring."""

from dataclasses import dataclass
from typing import Dict, Optional, Any


@dataclass
class RareClassAffinityResult:
   """Result of calculating affinity to a rare class of text.

   This class contains both:
   1. The overall rare_class_affinity_score for a collection of texts, which is used to prioritize
      cases for further investigation in a realtime context.
   2. The individual observation_scores for each text, which can be used to identify which specific
      observations contributed most to the overall pattern.

   As a high-recall candidate generator, this result helps identify potential instances of rare
   classes that warrant closer examination, prioritizing not missing true positives even at the
   cost of some false positives.

   Attributes:
      rare_class_affinity_score: The aggregated score indicating overall affinity to the rare class,
         typically calculated using skewness to identify patterns.
      observation_scores: Mapping of input text to its individual similarity score.
      aggregation_name: Optional name of the aggregation function used.
      aggregation_stats: Optional dictionary with aggregation-relevant statistics
         (e.g. top_k, percentile, temperature, num_positives).
      explanations: Optional per-text explainability records describing which neighbors and
         components contributed to each score.
   """

   rare_class_affinity_score: float
   observation_scores: Dict[str, float]
   aggregation_name: Optional[str] = None
   aggregation_stats: Optional[Dict[str, Any]] = None
   explanations: Optional[Dict[str, Any]] = None
