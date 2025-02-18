"""
This script takes a JSON config file (see the example in `data_configs/local_rn_config.json`).

The config file can have the following fields:

- `input_base_folder` (required): the base folder for the input data. This folder is
    expected to contain three subfolders: `train`, `valid`, and `test`. Each of these
    subfolders, in turn, should contain a number of `.csv` files. See below for a
    description of the expected csv format.
- `output_base_folder` (required): the base folder for the output data
- `features` (required): a list of features to process
- `vocabs`: a dictionary mapping feature names to paths to vocab files. If a feature is
  not listed in `vocabs`, the script will look for a file in the `vocab_dir` directory
  with the basename `[feature_name].txt`. Note that it is not necessary to specify a
  vocab for the `events` feature, as it is assumed to be the octuple-encoded input and
  there is a default vocab file included with the package.
- `vocab_dir`: the directory to look for vocab files in if they are not listed in the
  `vocabs` dictionary.
- `concat_features`: a list of lists of features to concatenate into a single feature.
  For example, you could combine "key_pc" and "mode" into a single feature "key_pc_mode"

There is an additional field `feature_must_divide_by` only used to validate the octuple
input, which shouldn't generally be modified.

# Expected CSV format

Each row represents a sequence (e.g., a piece of music). The only required column is
"events", which should contain a sequence of space-separated OctupleMIDI tokens. (We
expect this sequence to begin with 8 start tokens. It can also end with 8 stop tokens;
if it ends with fewer than 8 such tokens, the end will be padded with stop tokens.)
Other columns represent space separated sequences of tokens of features that you want to
predict. For example, `key` or `chord_quality`.

"""

import argparse
import logging
import pdb
import sys
import traceback

from musicbert_hf.script_helpers.data_preprocessing_helpers import (
    Config,
    load_config_from_json,
)
from musicbert_hf.script_helpers.get_vocab import handle_vocab
from musicbert_hf.script_helpers.tokens_h5 import to_tokens_h5


def custom_excepthook(exc_type, exc_value, exc_traceback):
    if exc_type is not KeyboardInterrupt:
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
        pdb.post_mortem(exc_traceback)


sys.excepthook = custom_excepthook


def main(config: Config):
    all_itos = {}
    all_stoi = {}
    for feature in config.features:
        itos, stoi = handle_vocab(
            csv_folder=config.train_input_folder,
            feature=feature,
            path=config.vocabs.get(feature, None),
        )

        all_itos[feature] = itos
        all_stoi[feature] = stoi

    for features_to_concat in config.concat_features:
        concatted_feature = "_".join(features_to_concat)
        if concatted_feature not in config.vocabs:
            raise NotImplementedError(
                f"Concatenated feature {concatted_feature} not in vocabs"
            )
        itos, stoi = handle_vocab(
            path=config.vocabs[concatted_feature],
        )
        all_itos[concatted_feature] = itos
        all_stoi[concatted_feature] = stoi

    for split in ["train", "test", "valid"]:
        logging.info(f"Processing {split} split")
        to_tokens_h5(
            input_csv_folder=getattr(config, f"{split}_input_folder"),
            output_folder=getattr(config, f"{split}_output_folder"),
            features=config.features,
            stoi=all_stoi,
            concat_features=config.concat_features,
            feature_must_divide_by=config.feature_must_divide_by,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    config = load_config_from_json(args.config)
    main(config)
