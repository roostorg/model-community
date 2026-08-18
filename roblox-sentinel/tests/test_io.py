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

"""Tests for IO modules in Sentinel."""

import os
import tempfile
import pytest
import torch
import json

from sentinel.io.saved_index_config import SavedIndexConfig
from sentinel.io.index_io import (
    save_index,
    load_index,
    load_corpus,
    create_s3_transport_params,
    CONFIG_FILE_NAME,
    EMBEDDINGS_FILE_NAME,
    CORPUS_FILE_NAME,
)


def test_saved_index_config():
    """Test SavedIndexConfig initialization and serialization."""
    # Test basic initialization
    model_name = "all-MiniLM-L6-v2"
    encoding_kwargs = {"normalize_embeddings": True}
    model_card = {"version": "1.0", "description": "Test model"}

    config = SavedIndexConfig(
        encoder_model_name_or_path=model_name,
        encoding_kwargs=encoding_kwargs,
        model_card=model_card,
    )

    assert config.encoder_model_name_or_path == model_name
    assert config.encoding_kwargs == encoding_kwargs
    assert config.model_card == model_card

    # Test to_dict and from_dict
    config_dict = config.to_dict()
    assert config_dict["encoder_model_name_or_path"] == model_name
    assert config_dict["encoding_kwargs"] == json.dumps(encoding_kwargs)
    assert config_dict["model_card"] == model_card

    # Test without model_card (initialized to empty dict by default)
    config_no_card = SavedIndexConfig(
        encoder_model_name_or_path=model_name,
        encoding_kwargs=encoding_kwargs,
    )
    assert config_no_card.model_card == {}
    config_no_card_dict = config_no_card.to_dict()
    assert "model_card" in config_no_card_dict
    assert config_no_card_dict["model_card"] == {}


@pytest.mark.parametrize(
    "positive_shape,negative_shape",
    [
        ((100, 384), (200, 384)),  # Standard case
        ((10, 384), (10, 384)),  # Equal sizes
        ((1, 384), (5, 384)),  # Single positive example
    ],
)
def test_index_io_local(positive_shape, negative_shape):
    """Test saving and loading index to/from local storage."""
    # Create test data
    positive_embeddings = torch.rand(positive_shape)
    negative_embeddings = torch.rand(negative_shape)

    config = SavedIndexConfig(
        encoder_model_name_or_path="all-MiniLM-L6-v2",
        encoding_kwargs={"normalize_embeddings": True},
        model_card={"version": "1.0", "description": "Test model"},
    )

    # Create a temp directory for saving
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save the index
        save_index(
            path=temp_dir,
            config=config,
            positive_embeddings=positive_embeddings,
            negative_embeddings=negative_embeddings,
        )

        # Check if files were created
        assert os.path.exists(os.path.join(temp_dir, CONFIG_FILE_NAME))
        assert os.path.exists(os.path.join(temp_dir, EMBEDDINGS_FILE_NAME))

        # Load the index
        loaded_config, loaded_positive, loaded_negative = load_index(path=temp_dir)

        # Verify loaded data
        assert (
            loaded_config.encoder_model_name_or_path
            == config.encoder_model_name_or_path
        )
        assert loaded_config.encoding_kwargs == config.encoding_kwargs
        assert loaded_config.model_card == config.model_card

        # Check tensor shapes and values
        assert loaded_positive.shape == positive_embeddings.shape
        assert loaded_negative.shape == negative_embeddings.shape
        assert torch.allclose(loaded_positive, positive_embeddings)
        assert torch.allclose(loaded_negative, negative_embeddings)


def _config():
    """Build a throwaway config for corpus tests."""
    return SavedIndexConfig(
        encoder_model_name_or_path="all-MiniLM-L6-v2",
        encoding_kwargs={"normalize_embeddings": True},
    )


def test_corpus_round_trip():
    """A saved corpus comes back identical, so explanations survive a reload."""
    positive_corpus = [f"pos {i}" for i in range(6)]
    negative_corpus = [f"neg {i}" for i in range(9)]

    with tempfile.TemporaryDirectory() as temp_dir:
        save_index(
            path=temp_dir,
            config=_config(),
            positive_embeddings=torch.rand(6, 8),
            negative_embeddings=torch.rand(9, 8),
            positive_corpus=positive_corpus,
            negative_corpus=negative_corpus,
        )

        assert os.path.exists(os.path.join(temp_dir, CORPUS_FILE_NAME))

        loaded_positive, loaded_negative = load_corpus(path=temp_dir)
        assert loaded_positive == positive_corpus
        assert loaded_negative == negative_corpus


def test_index_without_corpus_still_loads():
    """Saving with no corpus writes explicit nulls, and loading reports no corpus.

    The file is written even with nothing to put in it so that it always describes
    the embeddings beside it; see test_saving_without_corpus_clears_stale_corpus.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        save_index(
            path=temp_dir,
            config=_config(),
            positive_embeddings=torch.rand(4, 8),
            negative_embeddings=torch.rand(4, 8),
        )

        corpus_file = os.path.join(temp_dir, CORPUS_FILE_NAME)
        assert os.path.exists(corpus_file)
        with open(corpus_file) as f:
            assert json.load(f) == {"positive_corpus": None, "negative_corpus": None}

        # Both the existing loader and the new one behave.
        config, positive, negative = load_index(path=temp_dir)
        assert positive.shape == (4, 8)
        assert load_corpus(path=temp_dir) == (None, None)


def test_saving_without_corpus_clears_stale_corpus():
    """Re-saving without a corpus must not leave the previous corpus behind.

    The row counts here are deliberately identical, which is the case that slips
    past every alignment check: a corpus of 3 texts lines up with 3 new embedding
    rows, so nothing downstream can tell it describes the wrong 3 rows.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        save_index(
            path=temp_dir,
            config=_config(),
            positive_embeddings=torch.rand(3, 8),
            negative_embeddings=torch.rand(3, 8),
            positive_corpus=["first", "second", "third"],
            negative_corpus=["one", "two", "three"],
        )
        assert load_corpus(path=temp_dir) == (
            ["first", "second", "third"],
            ["one", "two", "three"],
        )

        # Same shape, different content, and this time no corpus to go with it.
        save_index(
            path=temp_dir,
            config=_config(),
            positive_embeddings=torch.rand(3, 8),
            negative_embeddings=torch.rand(3, 8),
        )

        assert load_corpus(path=temp_dir) == (None, None)


def test_deleting_corpus_file_degrades_gracefully():
    """An index with no corpus.json at all leaves the rest usable.

    This is the backward-compatibility guarantee: indices saved before corpus
    support lack the file entirely, and a missing file has to mean "no corpus"
    rather than an error.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        save_index(
            path=temp_dir,
            config=_config(),
            positive_embeddings=torch.rand(3, 8),
            negative_embeddings=torch.rand(3, 8),
            positive_corpus=["a", "b", "c"],
            negative_corpus=["x", "y", "z"],
        )
        os.remove(os.path.join(temp_dir, CORPUS_FILE_NAME))

        assert load_corpus(path=temp_dir) == (None, None)
        # load_index is untouched by any of this.
        config, positive, negative = load_index(path=temp_dir)
        assert positive.shape == (3, 8)


@pytest.mark.parametrize("which", ["positive", "negative"])
def test_save_refuses_misaligned_corpus(which):
    """A corpus that does not match its embeddings is refused at save time.

    Failing here is deliberate: writing it would bake the misalignment into the
    artifact, where it would later surface as confidently wrong explanations.
    """
    kwargs = {
        "positive_embeddings": torch.rand(5, 8),
        "negative_embeddings": torch.rand(5, 8),
        f"{which}_corpus": ["only", "three", "texts"],
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(ValueError, match=f"{which}_corpus has 3 entries"):
            save_index(path=temp_dir, config=_config(), **kwargs)

        # Nothing partial was left behind.
        assert not os.path.exists(os.path.join(temp_dir, CORPUS_FILE_NAME))


def test_malformed_corpus_json_propagates():
    """Corrupt JSON is a real problem and is not silently swallowed."""
    with tempfile.TemporaryDirectory() as temp_dir:
        save_index(
            path=temp_dir,
            config=_config(),
            positive_embeddings=torch.rand(2, 8),
            negative_embeddings=torch.rand(2, 8),
            positive_corpus=["a", "b"],
            negative_corpus=["c", "d"],
        )
        with open(os.path.join(temp_dir, CORPUS_FILE_NAME), "w") as f:
            f.write("{not valid json")

        with pytest.raises(json.JSONDecodeError):
            load_corpus(path=temp_dir)


def test_create_s3_transport_params():
    """Test creating S3 transport parameters."""
    # Test with credentials
    access_key = "test_access_key"
    secret_key = "test_secret_key"

    params = create_s3_transport_params(access_key, secret_key)
    assert params is not None
    assert "s3" in params
    assert "client_kwargs" in params["s3"]
    assert params["s3"]["client_kwargs"]["aws_access_key_id"] == access_key
    assert params["s3"]["client_kwargs"]["aws_secret_access_key"] == secret_key

    # Test with no credentials
    params = create_s3_transport_params()
    assert params == {}
