import json

import h5py
import torch
from torch.utils.data import Dataset

from musicbert_hf.constants import INPUT_PAD, TARGET_PAD


def collate_for_musicbert_fn(batch, multitask=False):
    """
    Expects a batch of dicts with keys "input_ids", "attention_mask", and "labels".
    Possibly also "conditioning_ids".
    """
    has_conditioning = "conditioning_ids" in batch[0]

    # Get max length in this batch
    max_input_length = max(len(item["input_ids"]) for item in batch)
    max_attention_length = max(len(item["attention_mask"]) for item in batch)

    if multitask:
        max_label_length = max(len(item["labels"][0]) for item in batch)
        num_tasks = len(batch[0]["labels"])
    else:
        max_label_length = max(len(item["labels"]) for item in batch)
        num_tasks = None
    # Initialize tensors
    input_ids = torch.full((len(batch), max_input_length), INPUT_PAD, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_attention_length), dtype=torch.long)

    if multitask:
        labels = [
            torch.full((len(batch), max_label_length), TARGET_PAD, dtype=torch.long)
            for _ in range(num_tasks)
        ]
    else:
        labels = torch.full(
            (len(batch), max_label_length), TARGET_PAD, dtype=torch.long
        )

    # Fill tensors
    for i, item in enumerate(batch):
        seq_len = len(item["input_ids"])
        input_ids[i, :seq_len] = item["input_ids"]
        attn_len = len(item["attention_mask"])
        attention_mask[i, :attn_len] = item["attention_mask"]
        if multitask:
            for j, label in enumerate(item["labels"]):
                labels[j][i, : len(label)] = label
        else:
            labels[i, : len(item["labels"])] = item["labels"]

    if has_conditioning:
        max_conditioning_length = max(len(item["conditioning_ids"]) for item in batch)
        conditioning_ids = torch.full(
            (len(batch), max_conditioning_length), INPUT_PAD, dtype=torch.long
        )
        for i, item in enumerate(batch):
            conditioning_ids[i, : len(item["conditioning_ids"])] = item[
                "conditioning_ids"
            ]
    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    if has_conditioning:
        out["conditioning_ids"] = conditioning_ids
    return out


class HDF5Dataset(Dataset):
    def __init__(
        self,
        inputs_path: str,
        targets_path: str | list[str],
        conditioning_path: str | None = None,
        dtype=None,
        device=None,
        compound_ratio: int = 8,
    ):
        super().__init__()
        self.inputs = h5py.File(inputs_path, "r")
        self.num_seqs = self.inputs["num_seqs"][()]
        self.conditioning = (
            h5py.File(conditioning_path, "r") if conditioning_path else None
        )
        if isinstance(targets_path, str):
            targets_path = [targets_path]
        self.multitask = len(targets_path) > 1
        self.targets = [h5py.File(target_path, "r") for target_path in targets_path]
        self.dtype = dtype
        self.device = device
        self.compound_ratio = compound_ratio

        # We need to cast to int because numpy int types aren't JSON serializable and
        #   HuggingFace will serialize the parameters.
        self.vocab_sizes = [int(target["vocab_size"][()]) for target in self.targets]
        target_names = [target["name"][()].decode() for target in self.targets]
        self.stois = {
            name: json.loads(target["vocab"][()].decode())
            for name, target in zip(target_names, self.targets)
        }
        if self.conditioning is not None:
            self.conditioning_vocab_size = int(self.conditioning["vocab_size"][()])
            conditioning_name = self.conditioning["name"][()].decode()
            self.stois[conditioning_name] = json.loads(
                self.conditioning["vocab"][()].decode()
            )
        else:
            self.conditioning_vocab_size = None

    def __len__(self):
        return self.num_seqs

    @property
    def num_tasks(self):
        return len(self.targets)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.inputs[f"{idx}"][()])
        targets = [torch.tensor(target[f"{idx}"][()]) for target in self.targets]
        if self.dtype is not None:
            input_ids = input_ids.to(self.dtype)
            targets = [target.to(self.dtype) for target in targets]
        if self.device is not None:
            input_ids = input_ids.to(self.device)
            targets = [target.to(self.device) for target in targets]
        assert len(input_ids) % self.compound_ratio == 0
        attention_mask = torch.ones(
            (len(input_ids) // self.compound_ratio,),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        target_lengths = {len(target) for target in targets}
        assert len(target_lengths) == 1
        target_length = target_lengths.pop()
        assert target_length * self.compound_ratio == input_ids.shape[0]

        if not self.multitask:
            targets = targets[0]
        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": targets,
        }
        if self.conditioning is not None:
            conditioning = torch.tensor(self.conditioning[f"{idx}"][()])
            out["conditioning_ids"] = conditioning
        return out
