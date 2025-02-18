"""
Predict keys and Roman-numeral annotations from a MIDI file.

Usage:

```bash
python scripts/predict.py --config /path/to/config.json \
    [--input_path /path/to/input.mid] \
    [--output_folder /path/to/output_folder]
```

The config file should has the following requiredfields:

- `key_checkpoint_path`: path to the key-prediction checkpoint
- `rn_checkpoint_path`: path to the Roman-numeral-prediction checkpoint. If this model
    does not predict "harmony_onset", then `harmony_onset_checkpoint_path` is required.

The config file may have the following optional fields:
- `input_path`: path to the input MIDI file. This field is required if it is not provided
    as a command line argument with the `--input_path` flag.
- `output_folder`: path to the folder where the output will be saved. This field is
    required if it is not provided as a command line argument with the `--output_folder`
    flag.
- `harmony_onset_checkpoint_path`: path to the harmony-onset-prediction checkpoint. If
    the Roman numeral model does not predict "harmony_onset", then this field is
    required.
- `make_pdf`: whether to make a PDF of the output. In order to work, this functionality
    has a number of external dependencies that must be available in the PATH (see the
    README). In addition, it may fail on certain types of complex rhythmic input.

For the remaining fields, see the `Config` class below.
"""

import argparse
import json
import logging
import os
import pdb
import shutil
import sys
import traceback
from dataclasses import dataclass
from functools import partial
from typing import Any, Sequence

import pandas as pd
import torch
from music_df.add_feature import add_default_time_sig, infer_barlines
from music_df.humdrum_export.pdf import df_to_pdf
from reprs.oct import OctupleEncodingSettings
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from musicbert_hf.chord_df import get_chord_df
from musicbert_hf.constants import INPUT_PAD, TARGET_PAD
from musicbert_hf.decoding_helpers import (
    get_key_annotations,
    get_rn_annotations,
    keep_new_elements_only,
)
from musicbert_hf.musicbert_class import (
    MusicBertMultiTaskTokenClassConditioned,
    MusicBertTokenClassification,
)
from musicbert_hf.script_helpers.get_vocab import handle_vocab
from musicbert_hf.utils.collate import collate_logits, collate_slice_ids
from musicbert_hf.utils.read import read_symbolic_score
from musicbert_hf.utils.sticky_viterbi import sticky_viterbi
from musicbert_hf.utils.sync_slices import sync_slices

# TODO: (Malcolm 2025-02-11) set to False
DEBUG = True

if DEBUG:

    def custom_excepthook(exc_type, exc_value, exc_traceback):
        if exc_type is not KeyboardInterrupt:
            traceback.print_exception(
                exc_type, exc_value, exc_traceback, file=sys.stdout
            )
        pdb.post_mortem(exc_traceback)

    sys.excepthook = custom_excepthook


@dataclass
class Config:
    input_path: str
    output_folder: str
    key_checkpoint_path: str
    rn_checkpoint_path: str
    harmony_onset_checkpoint_path: str | None = None
    make_pdf: bool = False
    batch_size: int = 4
    degree_feature_name: str = (
        "primary_alteration_primary_degree_secondary_alteration_secondary_degree"
    )

    # window_size should be the same as the model was trained with
    window_size: int = 1000

    # hop_size needs to be <= window_size
    hop_size: int = 250

    viterbi_alpha: float = 7.0

    harmony_onset_threshold: float = 0.3


def collate_for_musicbert_fn(batch, compound_ratio: int = 8):
    max_input_length = max(len(item["input_ids"]) for item in batch)
    # max_attention_length = max(len(item["attention_mask"]) for item in batch)
    # max_slice_ids_length = max(len(item["slice_ids"]) for item in batch)
    # assert (
    #     max_slice_ids_length
    #     == max_attention_length
    #     == max_input_length / compound_ratio
    # )
    max_attention_length = max_slice_ids_length = max_input_length // compound_ratio

    input_ids = torch.full((len(batch), max_input_length), INPUT_PAD, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_attention_length), dtype=torch.long)
    slice_ids = torch.full(
        (len(batch), max_slice_ids_length), TARGET_PAD, dtype=torch.long
    )
    for i, item in enumerate(batch):
        seq_len = len(item["input_ids"])
        input_ids[i, :seq_len] = item["input_ids"]
        attn_len = len(item["attention_mask"])
        attention_mask[i, :attn_len] = item["attention_mask"]
        slice_ids[i, : len(item["slice_ids"])] = item["slice_ids"]

    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "slice_ids": slice_ids,
    }
    if "conditioning_ids" in batch[0]:
        max_conditioning_length = max(len(item["conditioning_ids"]) for item in batch)
        assert max_conditioning_length == max_attention_length
        conditioning = torch.full(
            (len(batch), max_conditioning_length), INPUT_PAD, dtype=torch.long
        )
        for i, item in enumerate(batch):
            conditioning[i, : len(item["conditioning_ids"])] = item["conditioning_ids"]
        out["conditioning_ids"] = conditioning
    return out


class MusicBertInput(Dataset):
    def __init__(
        self,
        input_path: str,
        settings: OctupleEncodingSettings | None = None,
        dtype=None,
        device=None,
        compound_ratio: int = 8,
        window_size: int = 1000,
        hop_size: int = 250,
    ):
        super().__init__()
        df = read_symbolic_score(input_path)

        if settings is None:
            settings = OctupleEncodingSettings()
        encoded = settings.encode_f(df, feature_names=["distinct_slice_id"])

        # TODO: (Malcolm 2025-02-10)
        # Subtract 2 for the start and end tokens. Not totally sure that's
        #  the same way I'm doing it elsewhere. I should double check in
        #  the fine-tuning code.
        segments = encoded.segment(window_len=window_size - 2, hop=hop_size, start_i=0)

        itos, stoi = handle_vocab(
            path=os.path.join(
                os.path.dirname(__file__),
                "..",
                "supporting_files",
                "musicbert_fairseq_vocab.txt",
            ),
        )
        self.stoi = stoi
        self.itos = itos

        self.settings = settings

        self.input_ids = []
        self.segment_onsets = []
        self.df_indices = []
        self.slice_ids = []

        self.dtype = dtype
        self.device = device

        self.compound_ratio = compound_ratio

        for segment in segments:
            self.segment_onsets.append(segment["segment_onset"])
            self.df_indices.append(segment["df_indices"])

            slice_ids = segment["distinct_slice_id"]
            # A bit hacky, but the first item in slice_ids is "<s>" and
            #   we want integers throughout. We also add -100 for the stop token.
            slice_ids[0] = TARGET_PAD
            self.slice_ids.append(
                torch.tensor(
                    slice_ids + [TARGET_PAD], dtype=torch.long, device=self.device
                )
            )

            # Fairseq appends </s> to the end of the segment, we actually need 8
            # so the encoder adds 7 and we need to add 1 more here to get up
            # to the full octuple. This is a bit of a hack.
            segment_ids = torch.tensor(
                [stoi[i] for i in segment["input"]] + [stoi["</s>"]]
            )
            if self.dtype is not None:
                segment_ids = segment_ids.to(self.dtype)
            if self.device is not None:
                segment_ids = segment_ids.to(self.device)
            assert len(segment_ids) % self.compound_ratio == 0
            self.input_ids.append(segment_ids)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        input_ids = self.input_ids[idx]
        attention_mask = torch.ones(
            (len(input_ids) // self.compound_ratio,),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        slice_ids = self.slice_ids[idx]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "slice_ids": slice_ids,
        }


class KeyConditionedMusicBertInput(MusicBertInput):
    """
    Attributes:
        raw_key_indices: torch.Tensor of shape (n_slices,)
    """

    def __init__(
        self,
        *args,
        keys: Sequence[str],
        keys_stoi: dict[str, int],
        **kwargs,
    ):
        """
        Args:
            keys: list of keys of length n_slices
            keys_stoi: dict mapping keys to indices
        """
        super().__init__(*args, **kwargs)

        self.conditioning = []
        if DEBUG:
            max_slice_id = max(slice_id.max() for slice_id in self.slice_ids)
            assert max_slice_id == len(keys) - 1

        self.raw_key_indices = torch.tensor(
            [keys_stoi[key] for key in keys], dtype=torch.long, device=self.device
        )
        max_slice_id = max(slice_id.max() for slice_id in self.slice_ids)

        for slice_id in self.slice_ids:
            self.conditioning.append(
                torch.tensor(
                    [
                        self.raw_key_indices[i] if i >= 0 else INPUT_PAD
                        for i in slice_id
                    ],
                    dtype=torch.long,
                    device=self.device,
                )
            )

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        out["conditioning_ids"] = self.conditioning[idx]
        return out


def predict_keys(config: Config):
    """
    Returns:
        dict with keys "decoded_keys", "stoi", "per_slice_logits", "slice_ids":
            "decoded_keys": list of decoded keys of length n_slices
            "stoi": dict mapping keys to indices
            "per_slice_logits": torch.Tensor of shape (..., n_slices, n_keys)
            "slice_ids": torch.Tensor of shape (..., n_notes,)
    """
    dataset = MusicBertInput(
        input_path=config.input_path,
        settings=OctupleEncodingSettings(),
        window_size=config.window_size,
        hop_size=config.hop_size,
    )
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        drop_last=False,
        batch_size=4,
        collate_fn=collate_for_musicbert_fn,
    )
    key_model = MusicBertTokenClassification.from_pretrained(config.key_checkpoint_path)
    key_model.eval()
    with torch.no_grad():
        key_logits = []
        attention_masks = []
        slice_ids = []

        for batch in tqdm(dataloader, desc="Key model"):
            outputs = key_model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )
            key_logits.extend(outputs.logits)
            attention_masks.extend(batch["attention_mask"])
            slice_ids.extend(batch["slice_ids"])

        # We don't trim the start and end tokens because it complicates the calculation
        # of the correct overlap size. It doesn't matter much because the start and end
        # tokens will be near zero because of the interpolation for intermediate
        # segments, and we can remove the first and last start/end tokens respectively
        # below.

        collated_key_logits = collate_logits(
            key_logits,
            overlap_size=config.window_size - config.hop_size,
            attention_masks=attention_masks,
            trim_start=False,
            trim_end=False,
        )
        collated_key_logits = collated_key_logits[..., 1:-1, :]

        collated_slice_ids = collate_slice_ids(
            slice_ids,
            # Subtract 2
            overlap_size=config.window_size - config.hop_size - 2,
            check_overlap=True,
        )
        per_slice_key_logits = sync_slices(
            collated_key_logits,
            slice_ids=collated_slice_ids,
            return_per_slice=True,
        )
        per_slice_key_logits, key_stoi = drop_specials(
            per_slice_key_logits, key_model.config.label2id
        )
        per_slice_key_probs = torch.nn.functional.softmax(per_slice_key_logits, dim=-1)
        decoded_key_indices = sticky_viterbi(
            per_slice_key_probs, alpha=config.viterbi_alpha
        )
        key_itos = {v: k for k, v in key_stoi.items()}
        decoded_keys = [key_itos[i.item()] for i in decoded_key_indices]

    out = {
        "decoded_keys": decoded_keys,
        "stoi": key_stoi,
        "per_slice_logits": per_slice_key_logits,
        "slice_ids": collated_slice_ids,
    }
    return out


def drop_specials(logits: torch.Tensor, stoi: dict[str, int]):
    to_keep = {i: k for (k, i) in stoi.items() if not k.startswith("<")}
    logits = logits[..., list(to_keep.keys())]
    trimmed_stoi = {k: i for i, k in enumerate(to_keep.values())}
    return logits, trimmed_stoi


def handle_harmony_onset_logits(
    config: Config,
    onset_logits: list[torch.Tensor],
    attention_masks: list[torch.Tensor],
    slice_ids: list[torch.Tensor],
    harmony_onset_stoi: dict[str, int],
):
    """
    Returns:
        dict with keys "slice_ids", "onset_logits", "stoi":
            "slice_ids": torch.Tensor of shape (..., n_notes,)
            "onset_logits": torch.Tensor of shape (..., n_slices, 2)
            "stoi": dict mapping labels to indices
    """
    # We don't trim the start and end tokens because it complicates the calculation
    # of the correct overlap size. It doesn't matter much because the start and end
    # tokens will be near zero because of the interpolation for intermediate
    # segments, and we can remove the first and last start/end tokens respectively
    # below.
    collated_onset_logits = collate_logits(
        onset_logits,
        overlap_size=config.window_size - config.hop_size,
        attention_masks=attention_masks,
        trim_start=False,
        trim_end=False,
    )
    collated_onset_logits = collated_onset_logits[..., 1:-1, :]
    collated_slice_ids = collate_slice_ids(
        slice_ids,
        # Subtract 2
        overlap_size=config.window_size - config.hop_size - 2,
        check_overlap=True,
    )
    n_slices = collated_slice_ids[-1] + 1

    synced_onset_logits = sync_slices(
        collated_onset_logits,
        slice_ids=collated_slice_ids,
        return_per_slice=True,
    )
    synced_onset_logits, harmony_onset_stoi = drop_specials(
        synced_onset_logits, harmony_onset_stoi
    )
    assert synced_onset_logits.shape[-2] == n_slices

    out = {
        "slice_ids": collated_slice_ids,
        "onset_logits": synced_onset_logits,
        "stoi": harmony_onset_stoi,
    }
    return out


def remap_onset_slice_ids(
    onset_slice_ids: torch.Tensor, onset_predictions: torch.Tensor
) -> torch.Tensor:
    """
    Args:
        onset_slice_ids: torch.Tensor of shape (n_notes,)
        onset_predictions: torch.Tensor of shape (n_slices)

    Returns:
        torch.Tensor of shape (n_notes,)

    >>> onset_predictions = torch.tensor([True, False, True, False])
    >>> onset_slice_ids = torch.tensor([0, 0, 1, 2, 2, 2, 3])
    >>> remap_onset_slice_ids(onset_slice_ids, onset_predictions)
    tensor([0, 0, 0, 1, 1, 1, 1])
    """
    assert (onset_predictions[0]).all()
    # Create mapping array: for each index i, how many True values are there in onset_predictions up to i
    mapping = (
        torch.cumsum(onset_predictions, dim=-1) - 1
    )  # -1 because we want 0-based indices

    # Use mapping to convert values in x
    return mapping[onset_slice_ids]


def harmony_threshold_logits(
    config: Config,
    logits: list[torch.Tensor],
    attention_masks: list[torch.Tensor],
    harmony_onset_predictions: torch.Tensor,
    harmony_onset_slice_ids: torch.Tensor,
):
    """
    Args:
        config: Config
        logits: list of torch.Tensor of shape (..., segment_length, vocab_size)
        attention_masks: list of torch.Tensor of shape (..., segment_length,)
        harmony_onset_predictions: torch.Tensor of shape (..., n_slices)
        harmony_onset_slice_ids: torch.Tensor of shape (..., n_notes,)

    Returns:
        torch.Tensor of shape (..., n_notes, vocab_size)
    """
    # We don't trim the start and end tokens because it complicates the calculation
    # of the correct overlap size. It doesn't matter much because the start and end
    # tokens will be near zero because of the interpolation for intermediate
    # segments, and we can remove the first and last start/end tokens respectively
    # below.

    collated_logits = collate_logits(
        logits,
        overlap_size=config.window_size - config.hop_size,
        attention_masks=attention_masks,
        trim_start=False,
        trim_end=False,
    )
    collated_logits = collated_logits[..., 1:-1, :]
    assert collated_logits.shape[-2] == harmony_onset_slice_ids.shape[-1]

    remapped_slice_ids = remap_onset_slice_ids(
        harmony_onset_slice_ids, harmony_onset_predictions
    )

    thresholded_logits = sync_slices(
        collated_logits,
        slice_ids=remapped_slice_ids,
        return_per_slice=False,
    )
    return thresholded_logits


def predict_harmony_onset(config: Config):
    """Called when using a separate harmony onset model."""
    dataset = MusicBertInput(
        input_path=config.input_path,
        settings=OctupleEncodingSettings(),
        window_size=config.window_size,
        hop_size=config.hop_size,
    )
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        drop_last=False,
        batch_size=4,
        collate_fn=collate_for_musicbert_fn,
    )
    harmony_onset_model = MusicBertTokenClassification.from_pretrained(
        config.harmony_onset_checkpoint_path
    )
    harmony_onset_model.eval()
    with torch.no_grad():
        harmony_onset_logits = []
        attention_masks = []
        slice_ids = []

        for batch in tqdm(dataloader, desc="Harmony onset model"):
            outputs = harmony_onset_model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )
            harmony_onset_logits.extend(outputs.logits)
            attention_masks.extend(batch["attention_mask"])
            slice_ids.extend(batch["slice_ids"])

        harmony_onset_output = handle_harmony_onset_logits(
            config,
            harmony_onset_logits,
            attention_masks,
            slice_ids,
            harmony_onset_model.config.label2id,
        )
    return harmony_onset_output


def get_onset_mask(
    onset_logits: torch.Tensor,
    dataset: KeyConditionedMusicBertInput,
    harmony_onset_stoi: dict[str, int],
):
    """
    Args:
        onset_logits: torch.Tensor of shape (..., n_slices, 2)
        dataset: KeyConditionedMusicBertInput
        harmony_onset_stoi: dict mapping labels to indices

    Returns:
        torch.Tensor of shape (..., n_slices)
    """
    assert onset_logits.shape[-1] == len(harmony_onset_stoi) == 2
    onset_probs = torch.nn.functional.softmax(onset_logits, dim=-1)
    onset_predictions = (
        onset_probs[..., harmony_onset_stoi["yes"]] > config.harmony_onset_threshold
    )

    key_changes = dataset.raw_key_indices != torch.roll(dataset.raw_key_indices, 1)

    onset_predictions = onset_predictions | key_changes

    onset_predictions[0] = True  # The first slice is always an onset

    return onset_predictions


def predict_rn(
    config: Config,
    keys: Sequence[str],
    harmony_onset_output: dict[str, Any] | None = None,
):
    rn_model = MusicBertMultiTaskTokenClassConditioned.from_pretrained(
        config.rn_checkpoint_path
    )
    if harmony_onset_output is None:
        assert "harmony_onset" in rn_model.config.targets

    dataset = KeyConditionedMusicBertInput(
        input_path=config.input_path,
        settings=OctupleEncodingSettings(),
        window_size=config.window_size,
        hop_size=config.hop_size,
        keys=keys,
        keys_stoi=rn_model.config.multitask_label2id["key_pc_mode"],
    )
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        drop_last=False,
        batch_size=4,
        collate_fn=collate_for_musicbert_fn,
    )
    rn_model.eval()
    with torch.no_grad():
        rn_logits = {target: [] for target in rn_model.config.targets}
        attention_masks = []
        slice_ids = []
        for batch in tqdm(dataloader, desc="RN model"):
            outputs = rn_model(
                input_ids=batch["input_ids"],
                conditioning_ids=batch["conditioning_ids"],
                attention_mask=batch["attention_mask"],
            )
            for target, logits in zip(rn_model.config.targets, outputs.logits):
                rn_logits[target].extend(logits)
            attention_masks.extend(batch["attention_mask"])
            slice_ids.extend(batch["slice_ids"])

        if harmony_onset_output is None:
            harmony_onset_output = handle_harmony_onset_logits(
                config,
                rn_logits["harmony_onset"],
                attention_masks,
                slice_ids,
                rn_model.config.multitask_label2id["harmony_onset"],
            )

        harmony_onset_predictions = get_onset_mask(
            harmony_onset_output["onset_logits"], dataset, harmony_onset_output["stoi"]
        )

        trimmed_logits = {}
        trimmed_itos = {}
        trimmed_stoi = {}

        for target, logits in rn_logits.items():
            if target == "harmony_onset":
                continue
            thresholded_logits = harmony_threshold_logits(
                config,
                logits,
                attention_masks,
                harmony_onset_predictions,
                harmony_onset_output["slice_ids"],
            )
            trimmed_logits[target], trimmed_stoi[target] = drop_specials(
                thresholded_logits, rn_model.config.multitask_label2id[target]
            )

            trimmed_itos[target] = {v: k for k, v in trimmed_stoi[target].items()}
    out = {
        "rn_logits": trimmed_logits,
        "stoi": trimmed_stoi,
        "itos": trimmed_itos,
        "slice_ids": harmony_onset_output["slice_ids"],
    }
    return out


def get_annotations(key_output: dict[str, Any], rn_output: dict[str, Any]):
    rn_annots = get_rn_annotations(
        rn_output["rn_logits"],
        rn_output["itos"],
        degree_feature_name=config.degree_feature_name,
    )
    key_annots = get_key_annotations(key_output)
    assert len(key_annots) == len(rn_annots)

    # Consider the case where we have C.I immediately followed by F.I; in that case,
    #   `keep_new_elements_only` will drop the second I. So we need to

    rn_annots_at_key_changes = rn_annots[key_annots != ""]
    rn_annots = keep_new_elements_only(rn_annots)
    rn_annots[rn_annots_at_key_changes.index] = rn_annots_at_key_changes
    annots = key_annots + rn_annots
    return annots


def apply_annotations(config: Config, annotations: pd.Series):
    music_df = read_symbolic_score(config.input_path)
    music_df["harmonic_analysis"] = ""
    music_df.loc[music_df.type == "note", "harmonic_analysis"] = annotations.values
    return music_df


def save_tensor(tensor: torch.Tensor, path: str):
    with open(path, "wb") as f:
        torch.save(tensor, f)
    logging.info(f"Saved tensor to {path}")


def save_to_json(data: Any, path: str):
    with open(path, "w") as f:
        json.dump(data, f)
    logging.info(f"Saved JSON to {path}")


def save_to_pandas(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)
    logging.info(f"Saved pandas to {path}")


def save_to_pdf(df: pd.DataFrame, path: str):
    # Check if external requirements are in the path
    missing_humdrum_commands = []

    for cmd in ["timebase", "rid", "minrhy"]:
        if not shutil.which(cmd):
            missing_humdrum_commands.append(cmd)
    missing_other_commands = []

    for cmd in ["rsvg-convert", "img2pdf", "verovio"]:
        if not shutil.which(cmd):
            missing_other_commands.append(cmd)

    error = ""
    if missing_humdrum_commands:
        error += f"Required humdrum-tools not found in PATH: {', '.join(missing_humdrum_commands)}\n"
    if missing_other_commands:
        error += f"Other required commands not found in PATH: {', '.join(missing_other_commands)}\n"
    if error:
        raise RuntimeError(error)

    if "bar" not in df["type"].values:
        df = add_default_time_sig(df)
        df = infer_barlines(df)
    df_to_pdf(df, path, label_col="harmonic_analysis", capture_output=True)
    logging.info(f"Saved PDF to {path}")


def get_output_path(basename: str, output_folder: str):
    return os.path.join(output_folder, basename)


def save_output(
    config: Config,
    key_output: dict[str, Any],
    rn_output: dict[str, Any],
    annotated_music_df: pd.DataFrame,
    chord_df: pd.DataFrame,
    onset_output: dict[str, Any] | None = None,
):
    os.makedirs(config.output_folder, exist_ok=True)
    get_path = partial(get_output_path, output_folder=config.output_folder)
    save_tensor(key_output["per_slice_logits"], get_path("key_per_slice_logits.pt"))
    save_tensor(key_output["slice_ids"], get_path("key_slice_ids.pt"))
    save_to_json(key_output["stoi"], get_path("key_stoi.json"))

    for target, logits in rn_output["rn_logits"].items():
        save_tensor(logits, get_path(f"{target}_logits.pt"))
    for target, stoi in rn_output["stoi"].items():
        save_to_json(stoi, get_path(f"{target}_stoi.json"))

    save_tensor(rn_output["slice_ids"], get_path("rn_slice_ids.pt"))

    if onset_output is not None:
        save_tensor(onset_output["onset_logits"], get_path("onset_logits.pt"))
        save_to_json(onset_output["stoi"], get_path("onset_stoi.json"))

    save_to_pandas(annotated_music_df, get_path("annotated_music_df.csv"))
    save_to_pandas(chord_df, get_path("chord_df.csv"))

    if config.make_pdf:
        try:
            save_to_pdf(annotated_music_df, get_path("annotated.pdf"))
        except RuntimeError as e:
            logging.warning(f"Error making PDF: {e}")


def main(config: Config):
    key_output = predict_keys(config)

    if config.harmony_onset_checkpoint_path:
        onset_output = predict_harmony_onset(config)
        rn_output = predict_rn(
            config,
            key_output["decoded_keys"],
            harmony_onset_output=onset_output,
        )
    else:
        rn_output = predict_rn(config, key_output["decoded_keys"])

    annotations = get_annotations(key_output, rn_output)
    annotated_music_df = apply_annotations(config, annotations)
    chord_df = get_chord_df(annotated_music_df)
    save_output(
        config=config,
        key_output=key_output,
        rn_output=rn_output,
        annotated_music_df=annotated_music_df,
        chord_df=chord_df,
        onset_output=onset_output,
    )


def load_config_from_json(
    json_path: str, input_path: str | None = None, output_folder: str | None = None
) -> Config:
    with open(json_path, "r") as f:
        config_dict = json.load(f)
    if input_path is not None:
        config_dict["input_path"] = input_path
    if output_folder is not None:
        config_dict["output_folder"] = output_folder
    return Config(**config_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--input_path",
        type=str,
        required=False,
        help="Path to the input MIDI file. If not provided, then read from the config file.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=False,
        help="Path to the output folder. If not provided, then read from the config file.",
    )

    args = parser.parse_args()
    config = load_config_from_json(args.config, args.input_path, args.output_folder)
    logging.basicConfig(level=logging.INFO)
    main(config)
