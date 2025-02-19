"""
This module contains functions used by the `data_preprocessing.py` script for handling
vocabularies.
"""

import glob
import json
import logging
import os
from typing import Iterable, Literal

import pandas as pd
from tqdm import tqdm


def get_vocab(
    csv_folder=None,
    feature=None,
    path: str | None = None,
    sort: Literal["lexical", "frequency", "none"] = "lexical",
    specials: Iterable[str] = ("<unk>", "<pad>", "<s>", "</s>"),
):
    if path is not None and not os.path.exists(path):
        raise FileNotFoundError(f"Vocab file {path} does not exist")

    if path is not None:
        if path.endswith(".json"):
            logging.info(f"Loading JSON vocab from {path}")
            with open(path, "r") as f:
                return json.load(f)
        elif path.endswith(".txt"):
            logging.info(f"Loading FairSEQ formatted vocab from {path}")
            # Expect FairSEQ formatted vocab text file like this:
            # [token] [count]
            # [token] [count]
            # ...
            # Some files, for whatever reason, seem to have only
            # [token]
            # ...
            # So we need to handle both cases
            with open(path, "r") as f:
                return [
                    "<unk>",
                    "<pad>",
                    "<s>",
                    "</s>",
                ] + [
                    line.split()[0].strip()
                    for line in f.readlines()
                    if not line.startswith("madeupword")
                ]
        else:
            logging.info(f"Loading plaintext vocab from {path}")
            with open(path, "r") as f:
                return [line.strip() for line in f.readlines()]

    assert feature is not None and csv_folder is not None

    logging.info(f"Inferring {feature} vocab from {csv_folder}")
    csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))
    unique_tokens = set(specials)
    # for csv_file in csv_files:
    for csv_file in tqdm(csv_files, total=len(csv_files)):
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            unique_tokens.update(row[feature].split())

    # Remove specials so we can put them first after sorting
    unique_tokens = list(unique_tokens - set(specials))

    # TODO: (Malcolm 2025-01-13) This won't work for MusicBERT input because
    #    <0-100> comes before <0-25>
    if sort == "lexical":
        unique_tokens = sorted(unique_tokens)
    elif sort == "frequency":
        unique_tokens = sorted(
            unique_tokens, key=lambda x: df[feature].str.count(x).sum()
        )

    vocab = list(specials) + unique_tokens
    if path is not None:
        with open(path, "w") as f:
            if path.endswith(".json"):
                json.dump(vocab, f)
            else:
                for token in vocab:
                    f.write(token + "\n")

    return vocab


def handle_vocab(csv_folder=None, feature=None, path=None):
    itos = get_vocab(csv_folder, feature, path)
    stoi = {token: i for i, token in enumerate(itos)}

    # pad is -100 in huggingface, 1 in musicbert
    # For itos, we want to support both (therefore we need a dict rather
    #     than the simpler list implementation)
    # for stoi, we use -100 to be compatible with huggingface
    itos = {i: token for i, token in enumerate(itos)} | {-100: "<pad>"}
    stoi["<pad>"] = -100

    return itos, stoi
