import json
import logging
import os
from dataclasses import dataclass, field


@dataclass
class Config:
    input_base_folder: str
    output_base_folder: str
    features: list[str]
    concat_features: list[list[str]]
    vocabs: dict[str, str]
    feature_must_divide_by: dict[str, int] = field(
        default_factory=lambda: {
            "events": 8,
        }
    )

    def __post_init__(self):
        if "events" not in self.vocabs:
            self.vocabs["events"] = os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "supporting_files",
                "musicbert_fairseq_vocab.txt",
            )
            logging.info(f"Using default events vocab from {self.vocabs['events']}")

        self.input_base_folder = os.path.expanduser(self.input_base_folder)
        self.output_base_folder = os.path.expanduser(self.output_base_folder)

        for key in self.vocabs:
            self.vocabs[key] = os.path.expanduser(self.vocabs[key])

    @property
    def train_input_folder(self):
        return os.path.join(self.input_base_folder, "train")

    @property
    def test_input_folder(self):
        return os.path.join(self.input_base_folder, "test")

    @property
    def valid_input_folder(self):
        return os.path.join(self.input_base_folder, "valid")

    @property
    def train_output_folder(self):
        return os.path.join(self.output_base_folder, "train")

    @property
    def test_output_folder(self):
        return os.path.join(self.output_base_folder, "test")

    @property
    def valid_output_folder(self):
        return os.path.join(self.output_base_folder, "valid")

    @property
    def concatted_features(self):
        return ["_".join(concat) for concat in self.concat_features]


def load_config_from_json(json_path: str) -> Config:
    with open(json_path, "r") as f:
        return Config(**json.load(f))
