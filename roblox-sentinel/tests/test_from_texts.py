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

"""Tests for SentinelLocalIndex.from_texts()."""

import tempfile
import numpy as np
import pytest
import torch

from sentinel.sentinel_local_index import SentinelLocalIndex
from sentinel.score_types import RareClassAffinityResult


class _SpyModel:
    """Stands in for a sentence transformer, recording what it was asked to encode.

    Encoding is the only expensive step in from_texts, so what reaches this stub is
    exactly what a real run would pay for.
    """

    def __init__(self):
        self.encoded = []

    def encode(self, texts, **kwargs):
        self.encoded.append(list(texts))
        return np.zeros((len(texts), 4), dtype=np.float32)


class TestFromTexts:
    """Building an index in one call."""

    POSITIVE = ["unsafe content detected", "harmful behavior observed", "dangerous activity"]
    NEGATIVE = [
        "normal behavior detected",
        "regular activity observed",
        "safe content identified",
        "standard procedure followed",
        "ordinary events occurred",
        "the meeting went well",
    ]

    @pytest.mark.integration
    def test_builds_a_usable_index(self):
        """One call produces an index that scores text correctly."""
        index = SentinelLocalIndex.from_texts(
            positive_texts=self.POSITIVE,
            negative_texts=self.NEGATIVE,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )

        assert index.positive_embeddings.shape[0] == len(self.POSITIVE)
        assert index.negative_embeddings.shape[0] == len(self.NEGATIVE)
        assert index.sentence_model is not None

        result = index.calculate_rare_class_affinity(
            ["harmful unsafe behavior", "normal regular activity"]
        )
        assert isinstance(result, RareClassAffinityResult)

    @pytest.mark.integration
    def test_corpus_is_always_kept(self):
        """The corpus comes along automatically - half the point of the method.

        Forgetting it in the manual recipe costs you explanations, silently.
        """
        index = SentinelLocalIndex.from_texts(
            positive_texts=self.POSITIVE,
            negative_texts=self.NEGATIVE,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )

        assert index.positive_corpus == self.POSITIVE
        assert index.negative_corpus == self.NEGATIVE

    @pytest.mark.integration
    def test_normalization_is_applied_by_default(self):
        """Embeddings come out unit-length without the caller asking.

        Omitting normalize_embeddings by hand does not error; it just makes the
        similarity maths wrong. Asserting the norms catches a silent regression.
        """
        index = SentinelLocalIndex.from_texts(
            positive_texts=self.POSITIVE,
            negative_texts=self.NEGATIVE,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )

        norms = index.positive_embeddings.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
        assert index.encoding_kwargs["normalize_embeddings"] is True

    @pytest.mark.integration
    def test_ratio_downsamples_and_keeps_alignment(self):
        """The ratio is applied, and surviving negatives keep their own text."""
        index = SentinelLocalIndex.from_texts(
            positive_texts=self.POSITIVE,
            negative_texts=self.NEGATIVE,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            neg_to_pos_ratio=1.0,
            seed=42,
        )

        assert index.negative_embeddings.shape[0] == 3  # 3 positives * 1.0
        assert len(index.negative_corpus) == 3
        assert set(index.negative_corpus) <= set(self.NEGATIVE)

    @pytest.mark.integration
    def test_seeded_ratio_is_reproducible(self):
        """Same seed, same index."""
        kwargs = dict(
            positive_texts=self.POSITIVE,
            negative_texts=self.NEGATIVE,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            neg_to_pos_ratio=1.0,
        )
        a = SentinelLocalIndex.from_texts(seed=7, **kwargs)
        b = SentinelLocalIndex.from_texts(seed=7, **kwargs)

        assert torch.equal(a.negative_embeddings, b.negative_embeddings)
        assert a.negative_corpus == b.negative_corpus

    @pytest.mark.integration
    def test_round_trip_through_save_and_load(self):
        """An index built this way saves and reloads with explanations intact."""
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        index = SentinelLocalIndex.from_texts(
            positive_texts=self.POSITIVE,
            negative_texts=self.NEGATIVE,
            model_name=model_name,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            index.save(path=temp_dir, encoder_model_name_or_path=model_name)
            reloaded = SentinelLocalIndex.load(
                path=temp_dir, negative_to_positive_ratio=None, seed=1
            )

        assert reloaded.positive_corpus == self.POSITIVE
        assert reloaded.negative_corpus == self.NEGATIVE

    def _spy_model(self, monkeypatch):
        """Swap the real encoder for a spy, and hand the spy back."""
        spy = _SpyModel()
        monkeypatch.setattr(
            "sentinel.sentinel_local_index.get_sentence_transformer_and_scaling_fn",
            lambda *args, **kwargs: (spy, None),
        )
        return spy

    def test_surplus_negatives_are_never_encoded(self, monkeypatch):
        """The ratio must be applied before encoding, not after.

        Encoding is per-text and is the only expensive step, so encoding a negative
        and then discarding it is pure waste. At a 1:1 ratio against 3 positives,
        only 3 of the 50 negatives belong in the index, so only 3 should ever reach
        the encoder.
        """
        spy = self._spy_model(monkeypatch)
        positives = ["p0", "p1", "p2"]
        negatives = [f"n{i}" for i in range(50)]

        index = SentinelLocalIndex.from_texts(
            positive_texts=positives,
            negative_texts=negatives,
            neg_to_pos_ratio=1.0,
            seed=42,
        )

        assert [len(call) for call in spy.encoded] == [3, 3]
        assert spy.encoded[0] == positives
        # Everything encoded ended up in the index, and vice versa: nothing was
        # paid for and thrown away.
        assert spy.encoded[1] == index.negative_corpus

    def test_without_a_ratio_every_negative_is_encoded(self, monkeypatch):
        """Reordering the downsample must not change the no-ratio path."""
        spy = self._spy_model(monkeypatch)
        negatives = [f"n{i}" for i in range(20)]

        index = SentinelLocalIndex.from_texts(
            positive_texts=["p0", "p1"],
            negative_texts=negatives,
        )

        assert spy.encoded[1] == negatives
        assert index.negative_corpus == negatives

    @pytest.mark.integration
    def test_precomputed_embeddings_match_encoding_inline(self):
        """Passing embeddings must score identically to letting the index encode.

        This is what lets a sweep encode once and reuse the result: the embeddings
        depend on the encoder, not on the index they are scored against.
        """
        index = SentinelLocalIndex.from_texts(
            positive_texts=self.POSITIVE,
            negative_texts=self.NEGATIVE,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )
        texts = ["harmful unsafe behavior", "normal regular activity"]

        inline = index.calculate_rare_class_affinity(texts)
        precomputed = index.calculate_rare_class_affinity(
            texts,
            index.sentence_model.encode(texts, **index.encoding_kwargs),
        )

        assert precomputed.observation_scores == inline.observation_scores

    @pytest.mark.integration
    def test_mismatched_embeddings_raise(self):
        """Too few embeddings would pair observations with the wrong vectors."""
        index = SentinelLocalIndex.from_texts(
            positive_texts=self.POSITIVE,
            negative_texts=self.NEGATIVE,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )
        texts = ["one", "two", "three"]
        too_few = index.sentence_model.encode(texts[:2], **index.encoding_kwargs)

        with pytest.raises(ValueError, match="sample_embeddings has 2 rows"):
            index.calculate_rare_class_affinity(texts, too_few)

    @pytest.mark.parametrize(
        "kwargs,message",
        [
            ({"positive_texts": "a bare string"}, "positive_texts must be a list"),
            ({"negative_texts": "a bare string"}, "negative_texts must be a list"),
            ({"positive_texts": []}, "positive_texts must not be empty"),
            ({"negative_texts": []}, "negative_texts must not be empty"),
            ({"neg_to_pos_ratio": 0}, "neg_to_pos_ratio must be positive"),
            ({"neg_to_pos_ratio": -1.0}, "neg_to_pos_ratio must be positive"),
        ],
    )
    def test_input_validation(self, kwargs, message):
        """Bad input is rejected up front, before any expensive encoding happens.

        A bare string is iterable, so without this check it would be encoded one
        character at a time - confusing, slow, and entirely silent.
        """
        call = {"positive_texts": self.POSITIVE, "negative_texts": self.NEGATIVE}
        call.update(kwargs)
        with pytest.raises(ValueError, match=message):
            SentinelLocalIndex.from_texts(**call)
