import ast
import re
from typing import Any

import pandas as pd
import torch


def get_degree(concat_degree):
    if concat_degree == "<unk>":
        return "x", ""
    m = re.match(
        r"""
        (?P<primary_degree>[#b_][^#b_]+)
        (?P<secondary_degree>[#b_][^#b_]+)
        """,
        concat_degree,
        flags=re.VERBOSE,
    )
    assert m
    primary_degree = m.group("primary_degree").replace("_", "")
    secondary_degree = m.group("secondary_degree").replace("_", "")
    if secondary_degree == "I":
        secondary_degree = ""
    else:
        secondary_degree = "/" + secondary_degree
    return primary_degree, secondary_degree


def get_tokens(logits: dict[str, torch.Tensor], itos: dict[int, str], feature: str):
    these_logits = logits.get(feature)
    assert these_logits is not None
    preds = these_logits.argmax(axis=-1)
    return [itos[feature][p.item()] for p in preds]


def get_quality(quality):
    # (Malcolm 2024-04-18) possibly we want to do further processing, e.g.
    #   - remove 6 from augmented 6 chords "aug6" quality and otherwise simplify
    #   - only display the quality when it contradicts the expected value for the
    #       scale (this of course would require a lot more coding)

    return quality.replace("7", "")


TRIAD_INVERSIONS = {0: "", 1: "6", 2: "64"}
SEVENTH_CHORD_INVERSIONS = {0: "7", 1: "65", 2: "43", 3: "42"}


def get_inversion(raw_inversion, quality):
    # If the chord is a 7th or augmented 6th, we use 7th chord inversions. (Since
    #   we only have integers to indicate 1st, 2nd inversion etc., we can't distinguish
    #   German and Italian 6th chords.)
    raw_inversion = int(float(raw_inversion))
    if "7" in quality or quality == "aug6":
        return SEVENTH_CHORD_INVERSIONS.get(raw_inversion, "?")
    # If the quality is unknown we ignore the inversion
    elif quality == "x":
        return ""
    # Otherwise, assume to be a triad
    return TRIAD_INVERSIONS.get(raw_inversion, "?")


def get_rn_annotations(
    logits: dict[str, torch.Tensor],
    itos: dict[int, str],
    degree_feature_name: str = "degree",
):
    tokens = {}
    raw_inversions = None
    raw_qualities = None
    raw_primary_degrees = None

    for feature in (degree_feature_name, "inversion", "quality"):
        raw_tokens = get_tokens(logits, itos, feature)

        if feature == degree_feature_name:
            raw_primary_degrees, tokens["secondary_degree"] = zip(
                *map(get_degree, raw_tokens)
            )
        elif feature == "quality":
            raw_qualities = raw_tokens
            tokens["quality"] = list(map(get_quality, raw_qualities))
        elif feature == "inversion":
            raw_inversions = raw_tokens

    assert raw_inversions is not None
    assert raw_qualities is not None
    assert raw_primary_degrees is not None
    # We need to update the inversion based on whether it's a triad or 7th chord
    # We need the raw qualities here because we remove "7" from the processed qualities
    tokens["inversion"] = list(map(get_inversion, raw_inversions, raw_qualities))

    # Remove the "x" degree for augmented 6th chords
    tokens["primary_degree"] = [
        d if q != "Aug6" else "" for d, q in zip(raw_primary_degrees, raw_qualities)
    ]
    rn_annots = pd.Series(
        "".join(t)
        for t in zip(
            tokens["primary_degree"],
            tokens["quality"],
            tokens["inversion"],
            tokens["secondary_degree"],
        )
    )
    return rn_annots


MAJOR_KEYS = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")
MINOR_KEYS = ("c", "c#", "d", "eb", "e", "f", "f#", "g", "g#", "a", "bb", "b")


def get_key(key_pc_mode):
    assert len(key_pc_mode) > 1
    key_pc = key_pc_mode[:-1]
    mode = key_pc_mode[-1]

    # Using ast here seems like overkill but maybe I used it for some good reason
    #   that I've forgotten.
    key_pc = int(ast.literal_eval(key_pc))
    if mode == "M":
        # I was using ":" as the separator character but it is a special
        #   value in humdrum even when escaped.
        return MAJOR_KEYS[key_pc] + "."
    return MINOR_KEYS[key_pc] + "."


def keep_new_elements_only(series: pd.Series, fill_element=""):
    """
    >>> s = pd.Series(list("aaabbcddde"))
    >>> keep_new_elements_only(s)  # doctest: +NORMALIZE_WHITESPACE
    0    a
    1
    2
    3    b
    4
    5    c
    6    d
    7
    8
    9    e
    dtype: object
    """
    mask = series != series.shift(1)
    out = series.copy()
    out[~mask] = fill_element
    return out


def get_key_annotations(key_output: dict[str, Any]):
    decoded_keys = key_output["decoded_keys"]
    expanded_decoded_keys = [
        decoded_keys[slice_i] for slice_i in key_output["slice_ids"]
    ]
    human_readable_keys = [get_key(k) for k in expanded_decoded_keys]
    return keep_new_elements_only(pd.Series(human_readable_keys))
