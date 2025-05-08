import glob
import json
import logging
import os

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from musicbert_hf.script_helpers.get_vocab import handle_vocab


def to_tokens_h5(
    input_csv_folder,
    output_folder,
    features,
    stoi: dict[str, dict[str, int]],
    concat_features: list[list[str]] | None = None,
    max_rows=None,
    feature_must_divide_by: dict[str, int] | None = None,
):
    #print("in the function")
    csv_files = glob.glob(os.path.join(input_csv_folder, "*.csv"))
    logging.info(f"Found {len(csv_files)} csv files in {input_csv_folder}")
    #print(f"Found {len(csv_files)} csv files in {input_csv_folder}")
    os.makedirs(output_folder, exist_ok=True)
    if concat_features is None:
        concat_features = []
    output_files = {
        feature: h5py.File(os.path.join(output_folder, f"{feature}.h5"), "w")
        for feature in features
    } | {
        "_".join(concat): h5py.File(
            os.path.join(output_folder, f"{'_'.join(concat)}.h5"), "w"
        )
        for concat in concat_features
    }
    #print(output_files)
    for feature_name, this_stoi in stoi.items():
        vocab_size = len(this_stoi)
        #print(feature_name)
        #print(output_files)
        if feature_name in output_files:
            output_files[feature_name].create_dataset("vocab_size", data=vocab_size)
            string_dt = h5py.special_dtype(vlen=str)
            output_files[feature_name].create_dataset(
                "vocab", data=json.dumps(this_stoi), dtype=string_dt
            )
            output_files[feature_name].create_dataset(
                "name", data=feature_name, dtype=string_dt
            )

    row_count = 0
    for csv_file in tqdm(csv_files):
        df = pd.read_csv(csv_file)

        for feature in features:
            if feature == "events":
                seven_stop_tokens = "".join(
                    [" </s>"] * (feature_must_divide_by[feature] - 1)
                )
                assert df.iloc[0][feature].endswith(seven_stop_tokens)
                assert not df.iloc[0][feature][: -len(seven_stop_tokens)].endswith(
                    "</s>"
                )
            else:
                assert not df.iloc[0][feature].endswith("</s>")
        for _, row in df.iterrows():
            for feature in features:
                this_stoi = stoi[feature]
                tokens = row[feature].split()
                if feature == "events":
                    # FairSEQ preprocessing appends a single </s> token to the end of
                    # each sequence. To accommodate that, my data ends with 7 (not 8)
                    # stop tokens in the case of events and no stop tokens otherwise.
                    # Either way, we check for a missing stop token and append it.
                    missing_stop_token_count = 0
                    while tokens[-8 + missing_stop_token_count] != "</s>":
                        missing_stop_token_count += 1
                        if missing_stop_token_count == 8:
                            break
                    tokens.extend(["</s>"] * missing_stop_token_count)
                else:
                    if tokens[-1] != "</s>":
                        tokens.append("</s>")

                data = np.array(
                    [this_stoi.get(token, this_stoi["<unk>"]) for token in tokens]
                )
                output_files[feature].create_dataset(f"{row_count}", data=data)
            for concat in concat_features:
                concatted_feature = "_".join(concat)
                this_stoi = stoi[concatted_feature]
                separate_tokens = [row[feature].split() for feature in concat]
                merged_tokens = ["".join(tokens) for tokens in zip(*separate_tokens)]
                merged_tokens.append("</s>")
                data = np.array(
                    [
                        this_stoi.get(token, this_stoi["<unk>"])
                        for token in merged_tokens
                    ]
                )
                output_files[concatted_feature].create_dataset(
                    f"{row_count}", data=data
                )
            row_count += 1
            if max_rows is not None and row_count >= max_rows:
                # TODO: (Malcolm 2025-01-13) maybe split into multiple files if input is
                # very large?
                break
    logging.info(f"Wrote {row_count} rows to {output_folder}")
    for output_file in output_files.values():
        logging.info(f"Wrote {output_file.filename}")
        output_file.create_dataset("num_seqs", data=row_count)
        output_file.close()


def read_tokens(h5_path):
    out = []
    with h5py.File(h5_path, "r") as f:
        for i in range(len(f)):
            out.append(f[str(i)][()])
    return out


if __name__ == "__main__":
    _, events_stoi = handle_vocab("/Users/malcolm/tmp/foo/data/test/", "events")
    _, degree_stoi = handle_vocab(
        path="/Volumes/Zarebski/musicbert/saved_predictions/45951812_cond_on_39958320/test/primary_alteration_primary_degree_secondary_alteration_secondary_degree_dictionary.txt"
    )
    stoi = {
        "events": events_stoi,
        "primary_alteration_primary_degree_secondary_alteration_secondary_degree": degree_stoi,
    }
    to_tokens_h5(
        "/Users/malcolm/tmp/foo/data/test/",
        "/Users/malcolm/tmp/foo_tokens/data/test/",
        ["events"],
        concat_features=[
            [
                "primary_alteration",
                "primary_degree",
                "secondary_alteration",
                "secondary_degree",
            ]
        ],
        feature_must_divide_by={"events": 8},
        stoi=stoi,
    )

    tokens = read_tokens("/Users/malcolm/tmp/foo_tokens/data/test/events.h5")
    tokens2 = read_tokens(
        "/Users/malcolm/tmp/foo_tokens/data/test/primary_alteration_primary_degree_secondary_alteration_secondary_degree.h5"
    )
    breakpoint()
