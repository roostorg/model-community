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

"""Module for handling index configuration storage and retrieval."""

import json
from typing import Dict, Any, Optional


class SavedIndexConfig:
    """Encapsulate the configuration of a Sentinel index."""

    def __init__(
        self,
        encoder_model_name_or_path: str,
        encoding_kwargs: Dict[str, Any],
        model_card: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the SavedIndexConfig.

        Args:
            encoder_model_name_or_path: Path or name of the sentence transformer encoder model.
                This must be the exact name used to create the SentenceTransformer as it cannot be
                automatically extracted from the model instance.
            encoding_kwargs: Encoding keyword arguments.
            model_card: Metadata about the model.
        """
        self.encoder_model_name_or_path = encoder_model_name_or_path
        self.encoding_kwargs = encoding_kwargs
        self.model_card = model_card or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary."""
        return {
            "encoder_model_name_or_path": self.encoder_model_name_or_path,
            "encoding_kwargs": json.dumps(self.encoding_kwargs),
            "model_card": self.model_card,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SavedIndexConfig":
        """
        Create a SavedIndexConfig from a dictionary.

        Args:
            config_dict: Dictionary containing configuration data.

        Returns:
            A new SavedIndexConfig instance.
        """
        return cls(
            encoder_model_name_or_path=config_dict["encoder_model_name_or_path"],
            encoding_kwargs=json.loads(config_dict["encoding_kwargs"]),
            model_card=config_dict["model_card"],
        )
