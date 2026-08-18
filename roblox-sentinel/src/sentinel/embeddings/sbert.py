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
Module for Sentence-BERT (SBERT) embedding functionality.

This module provides functions for creating and working with SentenceTransformer models.
"""

from typing import Callable, Optional, Tuple
import logging

from sentence_transformers import SentenceTransformer

LOG = logging.getLogger(__name__)

# Global cache for SentenceTransformer models to avoid redundant loading
_model_cache = {}


def get_sentence_transformer_and_scaling_fn(
    sentence_model_name_or_path: str,
    use_cache: bool = True,
) -> Tuple[SentenceTransformer, Optional[Callable[[float], float]]]:
    """
    Create a SentenceTransformer model instance and return it along with an appropriate scaling function.
    
    Uses caching to avoid loading the same model multiple times, which significantly improves
    performance when the same model is used repeatedly.

    Args:
        sentence_model_name_or_path: Path or name of the sentence model.
        use_cache: Whether to use model caching. Default True for performance.

    Returns:
        A tuple containing:
        - A SentenceTransformer model instance
        - A scaling function for similarity scores if needed (only for E5 family models), or None
    """
    global _model_cache

    # Check cache first if caching is enabled
    if use_cache and sentence_model_name_or_path in _model_cache:
        LOG.debug(f"Loading cached SentenceTransformer model: {sentence_model_name_or_path}")
        model = _model_cache[sentence_model_name_or_path]
    else:
        LOG.debug(f"Creating new SentenceTransformer model: {sentence_model_name_or_path}")
        model = SentenceTransformer(sentence_model_name_or_path)

        # Cache the model if caching is enabled
        if use_cache:
            _model_cache[sentence_model_name_or_path] = model
            LOG.debug(f"Cached SentenceTransformer model: {sentence_model_name_or_path}")

    # Check if the model is from E5 family - they need score scaling
    if "e5-" in sentence_model_name_or_path.lower():
        return model, e5_scaling_function

    return model, None


def clear_model_cache() -> None:
    """
    Clear the global model cache to free up memory.
    
    Useful when switching between different models or when memory usage is a concern.
    """
    global _model_cache
    cache_size = len(_model_cache)
    _model_cache.clear()
    LOG.info(f"Cleared model cache ({cache_size} models removed)")


def get_cache_info() -> dict:
    """
    Get information about the current model cache.
    
    Returns:
        Dictionary containing cache statistics including cached models and memory usage info.
    """
    global _model_cache
    return {
        "cached_models": list(_model_cache.keys()),
        "cache_size": len(_model_cache),
        "memory_info": "Use clear_model_cache() to free memory if needed",
    }


def remove_from_cache(model_name: str) -> bool:
    """
    Remove a specific model from the cache.
    
    Args:
        model_name: Name/path of the model to remove from cache.
        
    Returns:
        True if model was found and removed, False if not in cache.
    """
    global _model_cache
    if model_name in _model_cache:
        del _model_cache[model_name]
        LOG.info(f"Removed model from cache: {model_name}")
        return True
    return False

def e5_scaling_function(score: float) -> float:
    """
    Scale the similarity score for E5 embeddings.

    E5 embeddings typically produce similarity scores in the range [0.7, 1.0]
    rather than [0, 1]. This function scales the E5 similarity score to the
    [0, 1] range for better comparison with other models.

    Args:
        score: The raw similarity score from E5 model (typically in range [0.7, 1.0])

    Returns:
        A scaled score in the range [0, 1]
    """
    # Assuming typical E5 similarity scores range from 0.7 to 1.0
    # Scale to [0, 1] by applying min-max scaling
    min_e5_score = 0.7
    max_e5_score = 1.0

    # Clamp the score to the expected range first
    clamped_score = max(min_e5_score, min(score, max_e5_score))

    # Apply min-max scaling: (x - min) / (max - min)
    scaled_score = (clamped_score - min_e5_score) / (max_e5_score - min_e5_score)

    return scaled_score
