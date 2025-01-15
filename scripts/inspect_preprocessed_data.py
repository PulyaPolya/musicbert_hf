import os
import sys

import h5py
import numpy as np
from lovely_numpy import lo, plot
from omegaconf import OmegaConf

from musicbert_hf.script_helpers.data_preprocessing_helpers import Config


def inspect_preprocessed_data(config: Config):
    tokens = {}
    for split in ["train", "test", "valid"]:
        output_folder = getattr(config, f"{split}_output_folder")
        tokens[split] = {}
        for feature in config.features + config.concatted_features:
            these_tokens = []
            h5_file = h5py.File(os.path.join(output_folder, f"{feature}.h5"), "r")
            for i in range(len(h5_file)):
                these_tokens.append(h5_file[f"{i}"][()])
            tokens[split][feature] = np.concatenate(these_tokens)
            print(f"{split} {feature}")
            print(lo(tokens[split][feature]))
            plot(tokens[split][feature])


if __name__ == "__main__":
    conf = OmegaConf.from_cli(sys.argv[1:])
    config = Config(**conf)  # type:ignore
    inspect_preprocessed_data(config)
