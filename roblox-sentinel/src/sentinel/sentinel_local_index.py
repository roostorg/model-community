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
Module for local Sentinel index implementation.

This module provides the implementation of the SentinelLocalIndex class for local semantic scoring.
"""

import logging
from typing import Optional, List, Mapping, Any, Callable, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search

from sentinel.score_formulae import calculate_contrastive_score, skewness, contrastive_components
from sentinel.io.saved_index_config import SavedIndexConfig
from sentinel.io.index_io import (
    save_index,
    load_index,
    load_corpus,
    create_s3_transport_params,
)
from sentinel.embeddings.sbert import get_sentence_transformer_and_scaling_fn
from sentinel.score_types import RareClassAffinityResult

LOG = logging.getLogger(__name__)

# Encoding options applied unless the caller overrides them. Normalization is not
# optional in practice: the similarity maths assumes unit vectors, and getting it
# wrong produces quietly incorrect scores rather than an error. Defined once so the
# constructor and from_texts() cannot drift apart.
DEFAULT_ENCODING_KWARGS: Mapping[str, Any] = {
    "normalize_embeddings": True,
}


def _validate_texts(name: str, texts: Any) -> None:
    """Reject the two text-argument mistakes that would otherwise fail quietly.

    Args:
        name: Argument name, used in the error message.
        texts: The value supplied by the caller.

    Raises:
        ValueError: If a bare string was passed instead of a list, or the list is empty.
    """
    if isinstance(texts, str):
        raise ValueError(
            f"{name} must be a list of strings, not a single string. A bare string is "
            f"iterable, so it would be encoded one character at a time instead of failing."
        )
    if texts is None or len(texts) == 0:
        raise ValueError(f"{name} must not be empty.")


def _corpus_if_aligned(
    name: str, corpus: Optional[List[str]], embeddings: Optional[torch.Tensor]
) -> Optional[List[str]]:
    """Return the corpus only if it lines up with its embeddings, else None.

    Degrading to row numbers is recoverable; confidently reporting the wrong
    sentence as the reason for a score is not, so a mismatch is discarded.

    Args:
        name: Which corpus is being checked, used in the warning.
        corpus: The corpus texts, or None.
        embeddings: The embeddings the corpus is supposed to describe.

    Returns:
        The corpus unchanged, or None if it is absent or misaligned.
    """
    if corpus is None or embeddings is None:
        return None
    if len(corpus) != embeddings.shape[0]:
        LOG.warning(
            "Discarding %s corpus: %d texts for %d embedding rows. Explanations will "
            "show row numbers instead of naming the wrong text.",
            name,
            len(corpus),
            embeddings.shape[0],
        )
        return None
    return corpus


def _split_generators(
    seed: Optional[int],
) -> Tuple[Optional[torch.Generator], Optional[torch.Generator]]:
    """Derive one independent generator per index side, or (None, None) when unseeded.

    A single shared generator would couple the two sides: selecting rows advances it,
    so whether the positives drew at all would decide which negatives came out. Keeping
    every positive draws nothing, so a grid cell at full positive size would select
    different negatives from one that subsampled, and two cells meant to differ along
    one axis would quietly differ along both.

    Args:
        seed: The caller's seed, or None to leave selection unseeded.

    Returns:
        Tuple of (positive generator, negative generator), both None when seed is None.
    """
    if seed is None:
        return None, None
    # Draw the two seeds from a root generator rather than offsetting the caller's seed
    # by hand, which keeps them independent without inventing arithmetic that could
    # collide across nearby seeds.
    root = torch.Generator().manual_seed(seed)
    positive_seed, negative_seed = torch.randint(
        high=2**62, size=(2,), generator=root
    ).tolist()
    return (
        torch.Generator().manual_seed(positive_seed),
        torch.Generator().manual_seed(negative_seed),
    )


def _choose_indices(
    available: int,
    n_keep: Optional[int],
    generator: Optional[torch.Generator],
    label: str,
) -> Optional[torch.Tensor]:
    """Pick which ``n_keep`` of ``available`` positions to keep, in their original order.

    Only the choice lives here, not what is done with it, because the two callers
    keep different things: :meth:`SentinelLocalIndex.subsample` selects rows of an
    existing embedding tensor, while :meth:`SentinelLocalIndex.from_texts` selects
    raw texts before paying to encode them.

    Args:
        available: How many positions there are to choose from.
        n_keep: How many to keep. None, or at least ``available``, keeps everything.
        generator: Optional seeded generator for a reproducible choice.
        label: "positive" or "negative", used in log messages.

    Returns:
        Sorted positions to keep, or None when everything is kept - which lets
        callers skip the copy entirely rather than rebuild an identical list.
    """
    if n_keep is None or n_keep >= available:
        if n_keep is not None and n_keep > available:
            LOG.info(
                "Requested %d %s examples but only %d are available - keeping all of them.",
                n_keep,
                label,
                available,
            )
        return None

    indices = torch.randperm(available, generator=generator)[:n_keep]
    # Order does not affect semantic_search, but keeping the original relative
    # order makes the result far easier to diff and debug.
    indices = torch.sort(indices).values
    LOG.info("Keeping %d %s examples out of %d", n_keep, label, available)
    return indices


def _select_subset(
    embeddings: torch.Tensor,
    corpus: Optional[List[str]],
    n_keep: Optional[int],
    generator: Optional[torch.Generator],
    label: str,
) -> Tuple[torch.Tensor, Optional[List[str]]]:
    """Randomly keep ``n_keep`` rows of one side of an index, corpus included.

    Args:
        embeddings: The embeddings to select from.
        corpus: Matching texts, or None.
        n_keep: How many rows to keep. None or a value at least as large as the
            available rows keeps everything.
        generator: Optional seeded generator for a reproducible choice.
        label: "positive" or "negative", used in log messages.

    Returns:
        Tuple of (embeddings, corpus) for the kept rows.
    """
    indices = _choose_indices(embeddings.shape[0], n_keep, generator, label)
    if indices is None:
        # Copy the corpus list so callers cannot mutate the original through the copy.
        return embeddings, (list(corpus) if corpus is not None else None)
    return _take_rows(embeddings, corpus, indices)


def _take_rows(
    embeddings: torch.Tensor,
    corpus: Optional[List[str]],
    indices: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[List[str]]]:
    """Select rows from embeddings and the matching corpus texts, keeping them aligned.

    The embeddings and the corpus are two parallel lists: row *i* of one describes
    entry *i* of the other. Any operation that drops rows has to drop the same
    positions from both, or explanations will confidently name unrelated text.

    Args:
        embeddings: Tensor to select rows from.
        corpus: Texts describing those rows, or None if the index has no corpus.
        indices: Row positions to keep.

    Returns:
        Tuple of (selected embeddings, selected corpus or None).
    """
    selected = embeddings[indices]
    if corpus is None:
        return selected, None
    if len(corpus) != embeddings.shape[0]:
        LOG.warning(
            "Corpus length %d does not match embedding rows %d - dropping corpus "
            "rather than risk misaligned explanations.",
            len(corpus),
            embeddings.shape[0],
        )
        return selected, None
    return selected, [corpus[i] for i in indices.tolist()]


class SentinelLocalIndex:
    """Calculate scores for detecting extremely rare classes of text using contrastive learning.

    This class implements a realtime approach specifically designed for detecting rare text patterns
    where traditional classifiers would fail due to extreme class imbalance. The core workflow is:

    1. Collect multiple observations from a single source (e.g., recent messages from a user)
    2. Calculate individual observation scores using contrastive learning
    3. Aggregate these scores using skewness to detect patterns, independent of observation count
    4. Apply optional threshold filtering for decision-making

    As a high-recall candidate generator, this approach prioritizes identifying potential cases for
    further investigation, emphasizing not missing true positives even at the cost of some false positives.

    The contrastive learning approach compares each observation against both rare class examples
    and common class examples, calculating a ratio of similarities. This ratio indicates whether
    the observation is more similar to the rare class than to the common class.

    By default, skewness is used as the aggregation method since it captures the prevalence of
    rare patterns without being affected by the total number of observations, making it ideal
    for scenarios with varying observation counts.

    For optimal results with English text, we recommend using the MiniLM-L6-v2 model with
    approximately 5-20k examples of the rare class.
    """

    def __init__(
        self,
        sentence_model: Optional[SentenceTransformer] = None,
        positive_embeddings: Optional[torch.Tensor] = None,
        negative_embeddings: Optional[torch.Tensor] = None,
        scale_fn: Optional[Callable[[float], float]] = None,
        encoding_additional_kwargs: Mapping[
            str, Any
        ] = {},  # Particularly of interest are prompt (or prompt_name) and precision
        positive_corpus: Optional[List[str]] = None,
        negative_corpus: Optional[List[str]] = None,
        model_card: Optional[
            Mapping[str, Any]
        ] = None,  # The description of where this model positive and negative examples came from, etc.
    ):
        """Initialize the SentinelLocalIndex.

        Args:
            sentence_model: A SentenceTransformer model instance.
            positive_embeddings: Tensor of embeddings for positive (rare class) examples.
            negative_embeddings: Tensor of embeddings for negative (common class) examples.
            scale_fn: Optional callable to scale similarity scores (needed for some models like E5).
            encoding_additional_kwargs: Additional keyword arguments for encoding.
            positive_corpus: List of original positive example texts (for debugging).
            negative_corpus: List of original negative example texts (for debugging).
            model_card: Dictionary with metadata about the model.

        Note:
            For direct initialization, you should get a model and scale_fn by calling:
            `model, scale_fn = get_sentence_transformer_and_scaling_fn(encoder_model_name_or_path)`

            When saving the index, you must provide the exact encoder_model_name_or_path
            as SentenceTransformer doesn't store the original model name.

        Use the class method `load` to load an index from S3 or local storage.
        """
        self.sentence_model: SentenceTransformer = sentence_model
        self.scale_fn: Optional[Callable[[float], float]] = scale_fn

        self.positive_embeddings: torch.Tensor = None
        if positive_embeddings is not None:
            if isinstance(positive_embeddings, torch.Tensor):
                self.positive_embeddings = positive_embeddings
            else:
                self.positive_embeddings = torch.tensor(positive_embeddings)

        self.negative_embeddings: torch.Tensor = None
        if negative_embeddings is not None:
            if isinstance(negative_embeddings, torch.Tensor):
                self.negative_embeddings = negative_embeddings
            else:
                self.negative_embeddings = torch.tensor(negative_embeddings)

        self.encoding_kwargs = dict(DEFAULT_ENCODING_KWARGS)
        self.encoding_kwargs.update(encoding_additional_kwargs)
        self.positive_corpus = positive_corpus
        self.negative_corpus = negative_corpus
        self.model_card = model_card

    def save(
        self,
        path: str,
        encoder_model_name_or_path: str,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ) -> SavedIndexConfig:
        """
        Save the index to a file or S3 path.

        Args:
            path: Path to save the index to (local directory or S3 URI).
            encoder_model_name_or_path: Name or path of the sentence transformer encoder model used.
                This must be the exact name used to create the SentenceTransformer as it cannot be
                reliably extracted from the model instance.
            aws_access_key_id: Optional AWS access key ID for S3 access.
            aws_secret_access_key: Optional AWS secret access key for S3 access.

        Returns:
            The SavedIndexConfig object that was saved. This is returned for informational purposes only,
            as the config has already been written to the specified location and will be automatically
            read by the load method.
        """
        # Create config
        config = SavedIndexConfig(
            encoder_model_name_or_path=encoder_model_name_or_path,
            encoding_kwargs=self.encoding_kwargs,
            model_card=self.model_card,
        )

        # Create transport parameters for S3 if needed
        transport_params = create_s3_transport_params(
            aws_access_key_id, aws_secret_access_key
        )

        # Save the index
        save_index(
            path=path,
            config=config,
            positive_embeddings=self.positive_embeddings,
            negative_embeddings=self.negative_embeddings,
            transport_params=transport_params,
            positive_corpus=self.positive_corpus,
            negative_corpus=self.negative_corpus,
        )

        # Return the config for informational purposes
        return config

    @classmethod
    def from_texts(
        cls,
        positive_texts: List[str],
        negative_texts: List[str],
        *,
        model_name: str = "all-MiniLM-L6-v2",
        neg_to_pos_ratio: Optional[float] = None,
        batch_size: int = 256,
        seed: Optional[int] = None,
        encoding_additional_kwargs: Optional[Mapping[str, Any]] = None,
        model_card: Optional[Mapping[str, Any]] = None,
        show_progress_bar: bool = False,
    ) -> "SentinelLocalIndex":
        """Build an index from raw text in one call.

        Doing this by hand takes eight steps, two of which fail *silently* when
        skipped: omit ``normalize_embeddings=True`` and the similarity maths quietly
        returns wrong numbers, and omit the corpus and you lose explanations. No
        crash, no warning. Performing those steps inside the library, where they are
        tested, means a caller cannot forget a step they never have to write.

        Args:
            positive_texts: Examples of the rare class to detect.
            negative_texts: Examples of ordinary, common-class content.
            model_name: Sentence transformer to encode with.
            neg_to_pos_ratio: Optional negatives-to-positives ratio. None keeps every
                negative given. Surplus negatives are dropped before encoding, so
                passing far more than the ratio needs costs little.
            batch_size: Encoding batch size.
            seed: Optional seed for the negative downsampling, so the resulting index
                is reproducible.
            encoding_additional_kwargs: Extra encoding options, merged over
                :data:`DEFAULT_ENCODING_KWARGS`.
            model_card: Optional metadata describing where the examples came from.
            show_progress_bar: Whether to show the encoder progress bar.

        Returns:
            A ready-to-use SentinelLocalIndex, corpus included.

        Raises:
            ValueError: If either text list is empty, if a bare string is passed where
                a list is expected, or if neg_to_pos_ratio is not positive.
        """
        _validate_texts("positive_texts", positive_texts)
        _validate_texts("negative_texts", negative_texts)
        if neg_to_pos_ratio is not None and neg_to_pos_ratio <= 0:
            raise ValueError(
                f"neg_to_pos_ratio must be positive, got {neg_to_pos_ratio}."
            )

        positive_corpus = list(positive_texts)
        negative_corpus = list(negative_texts)

        # Keep both return values: dropping scale_fn silently changes scores for
        # models like E5, with nothing to indicate anything went wrong.
        sentence_model, scale_fn = get_sentence_transformer_and_scaling_fn(model_name)

        encoding_kwargs = dict(DEFAULT_ENCODING_KWARGS)
        encoding_kwargs.update(encoding_additional_kwargs or {})

        # Drop the surplus negatives before encoding, not after. Encoding is the only
        # expensive step here, and it is per-text, so encoding a sentence and then
        # discarding it is pure waste: at a 1:1 ratio against 1,000 positives, a
        # caller passing 100,000 negatives would have paid to encode 99,000 rows
        # that never reach the index.
        if neg_to_pos_ratio is not None:
            n_keep = max(1, int(len(positive_corpus) * neg_to_pos_ratio))
            generator = (
                torch.Generator().manual_seed(seed) if seed is not None else None
            )
            indices = _choose_indices(
                len(negative_corpus), n_keep, generator, "negative"
            )
            if indices is not None:
                negative_corpus = [negative_corpus[i] for i in indices.tolist()]

        LOG.info(
            "Encoding %d positive and %d negative examples with %s",
            len(positive_corpus),
            len(negative_corpus),
            model_name,
        )
        positive_embeddings = torch.tensor(
            sentence_model.encode(
                positive_corpus,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                **encoding_kwargs,
            )
        )
        negative_embeddings = torch.tensor(
            sentence_model.encode(
                negative_corpus,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                **encoding_kwargs,
            )
        )

        return cls(
            sentence_model=sentence_model,
            positive_embeddings=positive_embeddings,
            negative_embeddings=negative_embeddings,
            scale_fn=scale_fn,
            encoding_additional_kwargs=encoding_kwargs,
            positive_corpus=positive_corpus,
            negative_corpus=negative_corpus,
            model_card=model_card,
        )

    @classmethod
    def load(
        cls,
        path: str,
        *,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        negative_to_positive_ratio: Optional[float] = 5.0,
        cache_model: bool = False,
        seed: Optional[int] = None,
    ) -> "SentinelLocalIndex":
        """
        Load the index from a path and returns a new SentinelLocalIndex instance.

        Args:
            path: Path to load the index from (local directory or S3 URI).
            aws_access_key_id: Optional AWS access key ID for S3 access.
            aws_secret_access_key: Optional AWS secret access key for S3 access.
            negative_to_positive_ratio: Ratio of negative examples to keep relative to positive examples.
                                      If None, preserves the original ratio from the saved index.
                                      If 5.0 (default), uses a 5:1 negative to positive ratio for optimal performance.
                                      If specified, downsamples negative examples to achieve the desired ratio.
            cache_model: Whether to use model caching for faster subsequent loads. Default True.
            seed: Optional seed for the negative downsampling. Without it the surviving
                subset differs on every load, so the same saved index scores slightly
                differently each time. Pass a seed to make loading reproducible.

        Returns:
            A new SentinelLocalIndex instance with the loaded model and embeddings.
        """
        # Create transport parameters for S3 if needed
        transport_params = create_s3_transport_params(
            aws_access_key_id, aws_secret_access_key
        )

        # Load the index
        config, positive_embeddings, negative_embeddings = load_index(
            path=path, transport_params=transport_params
        )

        # Optional; absent for indices saved before corpus support.
        positive_corpus, negative_corpus = load_corpus(
            path=path, transport_params=transport_params
        )
        positive_corpus = _corpus_if_aligned(
            "positive", positive_corpus, positive_embeddings
        )
        negative_corpus = _corpus_if_aligned(
            "negative", negative_corpus, negative_embeddings
        )

        # Create the sentence model and get the scaling function
        model_name = config.encoder_model_name_or_path

        sentence_model, scale_fn = get_sentence_transformer_and_scaling_fn(
            model_name,
            use_cache = cache_model
            )

        # Create a new instance with the loaded model and data
        instance = cls(
            sentence_model=sentence_model,
            scale_fn=scale_fn,
            positive_embeddings=positive_embeddings,
            negative_embeddings=negative_embeddings,
            encoding_additional_kwargs=config.encoding_kwargs,
            positive_corpus=positive_corpus,
            negative_corpus=negative_corpus,
            model_card=config.model_card,
        )

        # Apply negative ratio if needed
        instance._apply_negative_ratio(negative_to_positive_ratio, seed=seed)

        return instance

    def _apply_negative_ratio(
        self, negative_to_positive_ratio: Optional[float], seed: Optional[int] = None
    ):
        """
        Apply the negative_to_positive_ratio to reduce the number of negative (common class) examples.

        Args:
            negative_to_positive_ratio: The ratio of negative samples to keep relative to positive samples.
                                      If None, preserves the original ratio from the saved index.
                                      If 5.0 (default), uses optimized 5:1 ratio for best performance.
            seed: Optional seed making the choice of surviving negatives reproducible.
                Uses a private torch.Generator rather than a global seed, so nothing
                else in the caller's program has its randomness reset.
        """
        # Handle null/invalid inputs - preserve original ratio if any issues occur
        if negative_to_positive_ratio is None:
            LOG.info(
                "Preserving original ratio: %d negative examples to %d positive examples (%.1f:1)",
                self.negative_embeddings.shape[0],
                self.positive_embeddings.shape[0],
                self.negative_embeddings.shape[0] / self.positive_embeddings.shape[0],
            )
            return

        # Check for null embeddings
        if self.positive_embeddings is None or self.negative_embeddings is None:
            LOG.warning("Null embeddings detected - cannot apply ratio adjustment")
            return

        # Check for empty embeddings
        if self.positive_embeddings.shape[0] == 0 or self.negative_embeddings.shape[0] == 0:
            LOG.warning("Empty embeddings detected - cannot apply ratio adjustment")
            return

        # Check for invalid ratio values
        if negative_to_positive_ratio <= 0:
            LOG.warning("Invalid ratio %f - must be positive. Preserving original ratio.", negative_to_positive_ratio)
            return

        # Calculate the number of negative samples to keep
        try:
            num_negative_to_keep = int(
                self.positive_embeddings.shape[0] * negative_to_positive_ratio
            )
        except (ValueError, OverflowError, TypeError) as e:
            LOG.warning("Error calculating negative samples to keep: %s. Preserving original ratio.", str(e))
            return

        # Check if calculation resulted in valid number
        if num_negative_to_keep <= 0:
            LOG.warning("Calculated negative samples to keep is %d - invalid. Preserving original ratio.", num_negative_to_keep)
            return

        if self.negative_embeddings.shape[0] > num_negative_to_keep:
            LOG.info(
                "Keeping %d negative examples out of %d",
                num_negative_to_keep,
                self.negative_embeddings.shape[0],
            )
            # Randomly select a subset of the negative examples with error handling
            try:
                generator = (
                    torch.Generator().manual_seed(seed) if seed is not None else None
                )
                indices = torch.randperm(
                    self.negative_embeddings.shape[0], generator=generator
                )[:num_negative_to_keep]
                # Sorting does not change which rows survive, only their order. Keeping
                # the original relative order makes the result far easier to diff and debug.
                indices = torch.sort(indices).values
                self.negative_embeddings, self.negative_corpus = _take_rows(
                    self.negative_embeddings, self.negative_corpus, indices
                )
            except (RuntimeError, IndexError, TypeError) as e:
                LOG.error("Error during negative embedding downsampling: %s. Preserving original embeddings.", str(e))
                return
        else:
            LOG.info(
                "User requested %d negative examples but the model loaded only has %d",
                num_negative_to_keep,
                self.negative_embeddings.shape[0],
            )

    def subsample(
        self,
        n_positive: Optional[int] = None,
        neg_to_pos_ratio: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> "SentinelLocalIndex":
        """Return a smaller copy of this index, reusing the existing embeddings.

        Encoding a sentence produces the same numbers regardless of which index it
        ends up in, so a small index is just a large one with rows removed. This turns
        "re-encode everything" into "copy the rows you want", which is what makes
        sweeping index sizes affordable.

        Corpus texts are kept aligned with the embeddings they describe, and the
        sentence model is shared with the copy rather than reloaded.

        Args:
            n_positive: How many positive examples to keep. None keeps all of them.
                If larger than the index, everything available is kept.
            neg_to_pos_ratio: Negatives to keep per kept positive. None leaves the
                negatives untouched - note that shrinking the positives alone therefore
                *changes* the effective ratio, which is easy to do by accident.
            seed: Optional seed making the selection reproducible. Uses private
                torch.Generators, so the caller's other randomness is unaffected. Each
                side gets its own, which means the negatives chosen for a given seed and
                count do not change according to whether the positives were subsampled.

        Returns:
            A new SentinelLocalIndex. This instance is never modified.

        Raises:
            ValueError: If the index has no embeddings, or if n_positive or
                neg_to_pos_ratio is not positive. Unlike load()'s ratio handling, which
                warns and carries on, this raises: a grid search that silently ignored a
                bad argument would emit result rows describing an index you did not ask for.
        """
        if self.positive_embeddings is None or self.negative_embeddings is None:
            raise ValueError(
                "Cannot subsample an index without both positive and negative embeddings."
            )
        if n_positive is not None and n_positive <= 0:
            raise ValueError(f"n_positive must be positive, got {n_positive}.")
        if neg_to_pos_ratio is not None and neg_to_pos_ratio <= 0:
            raise ValueError(
                f"neg_to_pos_ratio must be positive, got {neg_to_pos_ratio}."
            )

        # One generator per side, so neither side's selection depends on how many draws
        # the other happened to make. See _split_generators.
        positive_generator, negative_generator = _split_generators(seed)

        # Positives first: the ratio is defined relative to how many positives survive,
        # so that count has to be settled before the negatives can be sized.
        positive_embeddings, positive_corpus = _select_subset(
            self.positive_embeddings,
            self.positive_corpus,
            n_positive,
            positive_generator,
            "positive",
        )

        n_negative: Optional[int] = None
        if neg_to_pos_ratio is not None:
            n_negative = int(positive_embeddings.shape[0] * neg_to_pos_ratio)
            if n_negative <= 0:
                LOG.info(
                    "Ratio %.4f against %d positives rounds to zero negatives - keeping 1, "
                    "since an index with no negatives cannot score anything.",
                    neg_to_pos_ratio,
                    positive_embeddings.shape[0],
                )
                n_negative = 1

        negative_embeddings, negative_corpus = _select_subset(
            self.negative_embeddings,
            self.negative_corpus,
            n_negative,
            negative_generator,
            "negative",
        )

        # scale_fn, encoding_kwargs and model_card all carry over. A dropped scale_fn
        # would silently change scores for models like E5.
        return type(self)(
            sentence_model=self.sentence_model,
            positive_embeddings=positive_embeddings,
            negative_embeddings=negative_embeddings,
            scale_fn=self.scale_fn,
            encoding_additional_kwargs=self.encoding_kwargs,
            positive_corpus=positive_corpus,
            negative_corpus=negative_corpus,
            model_card=self.model_card,
        )

    def calculate_rare_class_affinity(
        self,
        text_samples: List[str],
        sample_embeddings: Optional[np.ndarray] = None,
        *,
        top_k: int = 5,
        similarity_formula: Callable[[List[float], List[float]], float] = calculate_contrastive_score,
        # Function to aggregate individual scores into an overall affinity score
        aggregation_function: Callable[[np.array], float] = skewness,
        # Margin to ignore when text is only slightly more similar to positive than negative.
        min_score_to_consider: float = 0.1,
        # Use when simulating by sampling texts from the same data indexed.
        prevent_exact_match: bool = False,
        encoding_additional_kwargs: Mapping[str, Any] = {},
        show_progress_bar: bool = False,
        explain: bool = True,
        include_neighbors: bool = True,
        neighbors_limit: int = 5,
    ) -> RareClassAffinityResult:
        """Calculate rare class affinity for the given text samples in realtime.

        This method serves as a high-recall candidate generator for identifying potential rare class instances
        that warrant further investigation. It encodes recent observations from a single source and compares
        them to rare class and common class examples, prioritizing not missing true positives.

        For each observation, it calculates an individual score based on similarity to the rare class versus
        the common class. It then aggregates these scores, using an aggregation function like skewness,
        to detect patterns across multiple observations, independent of their total count.

        Args:
            text_samples: List of text strings to evaluate for rare class affinity.
            sample_embeddings: Optional pre-computed embeddings for ``text_samples``, in
                the same order. Supplying them skips the encoding step, which is the only
                expensive part of this method. Embeddings depend on the encoder, not on
                the index contents, so one set can be reused across many indices built
                from the same model - see
                :func:`sentinel.simulation.encode_observations`.
            top_k: Number of closest neighbors to consider when calculating the score.
            similarity_formula: Function to calculate individual similarity scores.
            aggregation_function: Function to aggregate individual scores into an overall score.
            min_score_to_consider: Threshold below which scores are set to 0.
            prevent_exact_match: Whether to skip exact matches when scoring.
            encoding_additional_kwargs: Additional keyword arguments for encoding.
            show_progress_bar: Whether to display a progress bar during encoding.
            explain: Whether to include per-text explainability details.
            include_neighbors: Whether to include top-neighbor records in explainability output.
            neighbors_limit: Maximum number of neighbor records to include per text.

        Returns:
            RareClassAffinityResult containing both the overall affinity score and
            individual observation scores for each text sample.

        Raises:
            ValueError: If ``sample_embeddings`` is supplied with a different number of
                rows than ``text_samples``, which would silently score each observation
                against the wrong embedding.
        """
        if sample_embeddings is None:
            # Merge the default encoding kwargs with any additional ones provided
            effective_encoding_kwargs = self.encoding_kwargs.copy()
            effective_encoding_kwargs["show_progress_bar"] = show_progress_bar
            effective_encoding_kwargs.update(encoding_additional_kwargs)

            # Encode the input samples to get their embeddings.
            # We currently don't support multi-process encoding in this method, because
            # it is meant for online scoring.
            sample_embeddings = self.sentence_model.encode(
                text_samples,
                **effective_encoding_kwargs,
            )
        elif len(sample_embeddings) != len(text_samples):
            raise ValueError(
                f"sample_embeddings has {len(sample_embeddings)} rows but text_samples "
                f"has {len(text_samples)} entries. Scoring them together would pair each "
                f"observation with the wrong embedding, so refusing to continue."
            )

        # If we need to prevent exact matches (e.g., when scoring examples that are in the index),
        # request an additional neighbor so we can skip the exact match later
        additional_neighbors = 1 if prevent_exact_match else 0

        # Perform semantic search to find the most similar positive (rare class) examples
        positive_matches = semantic_search(
            sample_embeddings,
            self.positive_embeddings,
            top_k=top_k + additional_neighbors,
        )

        # Perform semantic search to find the most similar negative (common class) examples
        negative_matches = semantic_search(
            sample_embeddings,
            self.negative_embeddings,
            top_k=top_k + additional_neighbors,
        )

        observation_scores = {}
        explanations = {} if explain else None

        for i, q in enumerate(text_samples):
            LOG.debug("Query: %s", q)

            # Combine and sort both positive and negative matches by similarity score
            # The "+" sign marks positive examples, "-" sign marks negative examples
            matches = sorted(
                [(hit["score"], hit["corpus_id"], "+") for hit in positive_matches[i]]
                + [
                    (hit["score"], hit["corpus_id"], "-") for hit in negative_matches[i]
                ],
                key=lambda x: x[0],
                reverse=True,
            )

            # Initialize lists to collect similarity scores for the top matches
            similarities_topk_positive = []
            similarities_topk_negative = []
            max_h = top_k  # Number of examples to consider
            neighbor_records = [] if include_neighbors else None

            # Process each match in order of similarity (highest first)
            for h, (score, corpus_id, sign) in enumerate(matches):
                # Stop once we've collected enough examples
                if h == max_h:
                    break

                # Skip exact matches if requested (when score is almost exactly 1.0)
                if prevent_exact_match and h == 0 and abs(score - 1.0) < 1e-3:
                    max_h += 1  # Compensate for skipping this match
                    continue

                # Apply scaling to the similarity score if a scaling function is available
                if self.scale_fn:
                    scaled_score = self.scale_fn(score)
                else:
                    scaled_score = score

                # Add the score to the appropriate list based on whether it's positive or negative
                if sign == "+":
                    similarities_topk_positive.append(scaled_score)
                    neighbor = (
                        self.positive_corpus[corpus_id]
                        if self.positive_corpus
                        else corpus_id
                    )
                else:
                    if sign != "-":
                        raise AssertionError(f"Unexpected sign: {sign}")
                    similarities_topk_negative.append(scaled_score)
                    neighbor = (
                        self.negative_corpus[corpus_id]
                        if self.negative_corpus
                        else corpus_id
                    )

                if LOG.level <= logging.DEBUG:
                    LOG.debug(
                        f"[{sign}] {neighbor} (Score: {score:.4f}, Scaled Score: {scaled_score:.4f})"
                    )

                if include_neighbors and len(neighbor_records) < neighbors_limit:
                    # Keep a compact neighbor record for explainability
                    try:
                        corpus_id_int = int(corpus_id)
                    except Exception:
                        corpus_id_int = int(corpus_id) if isinstance(corpus_id, (int, np.integer)) else 0
                    neighbor_records.append(
                        {
                            "sign": "+" if sign == "+" else "-",
                            "raw_score": float(score),
                            "scaled_score": float(scaled_score),
                            "neighbor": neighbor,
                            "corpus_id": corpus_id_int,
                        }
                    )

            # Ensure we have at least one similarity value for each category
            # If we didn't find any of a particular category in the top matches,
            # use the first match from the original search
            if not similarities_topk_positive:
                similarities_topk_positive = [positive_matches[i][0]["score"]]
            if not similarities_topk_negative:
                similarities_topk_negative = [negative_matches[i][0]["score"]]

            if LOG.level <= logging.DEBUG:
                LOG.debug(
                    f"Top {top_k} similarities for '{q}': "
                    f"[+] {similarities_topk_positive}, [-] {similarities_topk_negative}"
                )

            # Calculate the final score using the provided formula (default: calculate_contrastive_score)
            # This compares how close the observation is to positive examples vs. negative examples
            score = similarity_formula(
                similarities_topk_pos=similarities_topk_positive,
                similarities_topk_neg=similarities_topk_negative,
            )

            # Apply threshold to filter out borderline cases
            # If the score is below the minimum threshold, set it to zero
            if score < min_score_to_consider:
                observation_scores[q] = 0.0
            else:
                observation_scores[q] = score

            # Per-text explainability
            if explain:
                pos_term, neg_term, log_ratio = contrastive_components(
                    similarities_topk_pos=similarities_topk_positive,
                    similarities_topk_neg=similarities_topk_negative,
                )
                explanations[q] = {
                    "topk_positive": [float(x) for x in similarities_topk_positive],
                    "topk_negative": [float(x) for x in similarities_topk_negative],
                    "contrastive": {
                        "positive_term": pos_term,
                        "negative_term": neg_term,
                        "log_ratio_unclipped": log_ratio,
                    },
                    "neighbors": neighbor_records[:neighbors_limit]
                    if include_neighbors and neighbor_records is not None
                    else None,
                }

        # Calculate the overall rare class affinity score by aggregating individual scores
        # If there are no scores, default to 0.0
        if not observation_scores:
            rare_class_score = 0.0
        else:
            rare_class_score = aggregation_function(
                np.array(list(observation_scores.values()))
            )

        # Aggregation metadata for explainability
        agg_name = getattr(aggregation_function, "__name__", str(aggregation_function))
        agg_stats = {
            "num_texts": len(text_samples),
            "num_positive_scores": int(
                np.sum(np.array(list(observation_scores.values())) > 0)
            ),
            "top_k_per_observation": top_k,
            "min_score_to_consider": float(min_score_to_consider),
        }

        return RareClassAffinityResult(
            rare_class_affinity_score=rare_class_score,
            observation_scores=observation_scores,
            aggregation_name=agg_name,
            aggregation_stats=agg_stats,
            explanations=explanations if explain else None,
        )