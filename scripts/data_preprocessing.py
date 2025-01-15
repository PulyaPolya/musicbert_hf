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
