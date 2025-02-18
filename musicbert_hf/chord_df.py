import math
import os
from itertools import count

import pandas as pd
from music_df.add_feature import add_default_time_sig, infer_barlines, make_bar_explicit
from music_df.keys import keys_to_key_change_ints


def get_chord_df(music_df: pd.DataFrame):
    orig_idx = None

    time_sig_mask = music_df.type == "time_signature"
    has_initial_time_sig = time_sig_mask.any() and (
        music_df[time_sig_mask].index[0]
        <= music_df[music_df.type.isin({"note", "bar"})].index[0]
    )

    if not has_initial_time_sig:
        # warnings.warn(f"{csv_path=} has no time signature")
        orig_idx = music_df.index.copy()
        music_df = add_default_time_sig(music_df, keep_old_index=True)
    if "bar" not in music_df["type"].values:
        if orig_idx is None:
            orig_idx = music_df.index.copy()

        music_df = infer_barlines(music_df, keep_old_index=True)

    music_df = make_bar_explicit(music_df)

    if orig_idx is not None:
        # remove the added rows (barlines, time sigs) so that the chord list
        #   and music_df will align
        music_df = music_df.set_index("index").loc[orig_idx]

    chord_df = music_df[music_df["harmonic_analysis"].ne("")]

    def split_analysis(t):
        if "." in t:
            return pd.Series(t.split(".", maxsplit=1))
        else:
            return pd.Series(["", t])

    split_df = chord_df["harmonic_analysis"].apply(split_analysis)
    split_df = split_df.rename({0: "key", 1: "rn"}, axis=1)
    chord_df = pd.concat([chord_df, split_df], axis=1)

    chord_df = chord_df[["key", "rn", "onset", "bar_number"]].reset_index(drop=True)
    chord_df["release"] = chord_df["onset"].shift(-1)

    # If the last row is a bar or other with NAN release, iterate backwards until we
    #   find a note etc.
    for i in count(start=-1, step=-1):
        if not math.isnan(music_df["release"].iloc[i]):
            chord_df.loc[len(chord_df) - 1, "release"] = music_df["release"].iloc[i]
            break

    key_changes, key_mask = keys_to_key_change_ints(chord_df["key"])
    for column_name in key_changes:
        chord_df[column_name] = ""
        chord_df.loc[key_mask, column_name] = key_changes[column_name]

    return chord_df
