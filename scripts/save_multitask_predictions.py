import argparse
import json
import logging
import os
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional
from transformers import BertConfig
from typing import Dict, List, Union
from functools import partial
from transformers import Trainer, TrainingArguments
from typing import Literal, Sequence


import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from pathlib import Path
import optuna

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helpers import load_baseline_params, create_hyperparams_dict, get_dataset, load_model, LimitedDataset
from musicbert_hf.data import HDF5Dataset, collate_for_musicbert_fn
from musicbert_hf.metrics import compute_metrics, compute_metrics_multitask
from musicbert_hf.models import freeze_layers, MusicBertTokenClassification, MusicBertMultiTaskTokenClassification, MusicBertMultiTaskTokenClassConditioned
from config import load_config

# @dataclass
# class Config:
#     data_dir: str  # assumed to end in '_bin' and have an equivalent '_raw' with 'metadata_test.txt'
#     checkpoint: str  # path to model checkpoint
#     output_folder: str  # where predictions will be stored
#     targets: str | list[str]
#     ref_dir: Optional[str] = None  # contains target_names.json and label[x]/dict.txt
    

#     # Dataset and batching
#     dataset: str = "test"  # choices: test, valid, train
#     batch_size: int = 4
#     max_examples: Optional[int] = None
    

#     # Model/task parameters
#     compound_token_ratio: int = 8
#     ignore_specials: int = 4
#     task: str = "musicbert_multitask_sequence_tagging"
#     head: str = "sequence_multitask_tagging_head"

#     # Misc
#     msdebug: bool = False
#     overwrite: bool = False
#     seed: int = 42
#     conditioning: str | None = None
#     data_dir_for_metadata: str = None
#     baseline: bool = False

#     @property
#     def train_dir(self) -> str:
#         return os.path.join(self.data_dir, "train")

#     @property
#     def valid_dir(self) -> str:
#         return os.path.join(self.data_dir, "valid")

#     @property
#     def test_dir(self) -> str:
#         return os.path.join(self.data_dir, "test")
    
#     @property
#     def output_dir(self) -> str:
#         return os.path.join(self.output_dir_base, self.job_id)

#     def target_paths(self, split: Literal["train", "valid", "test"]) -> list[str]:
#         return [
#             os.path.join(self.data_dir, split, f"{target}.h5")
#             for target in self.targets
#         ]

#     def conditioning_path(self, split: Literal["train", "valid", "test"]) -> str | None:
#         return (
#             None
#             if not self.conditioning
#             else os.path.join(self.data_dir, split, f"{self.conditioning}.h5")
#         )

#     @property
#     def multitask(self) -> bool:
#         return len(self.targets) > 1



# def _load_yaml(path: Path) -> dict:
#     raw = path.read_text()
#     if path.suffix.lower() in (".yml", ".yaml"):
#         data = yaml.safe_load(raw) or {}
#     return data

# def load_config(path: str | os.PathLike) -> Config:
#     p = Path(path)
#     if not p.exists():
#         raise FileNotFoundError(f"Config file not found: {p}")
#     data = _load_yaml(p)
#     # allow hyphenated keys in file
#     data = {k.replace("-", "_"): v for k, v in data.items()}
#     # validate keys
#     valid = set(Config.__annotations__.keys())
#     unknown = set(data) - valid
#     if unknown:
#         raise ValueError(f"Unknown config keys: {sorted(unknown)}")
#     return Config(**data)



def _ensure_dir_clean(path: str, overwrite: bool):
    if os.path.exists(path):
        if overwrite:
            shutil.rmtree(path)
        else:
            raise ValueError(f"Output folder {path} already exists. Use overwrite=True.")
    os.makedirs(path, exist_ok=False)

def _decode_ids_to_tokens(ids: np.ndarray, id2label: Dict[int, str]) -> List[str]:
    # Map each int id -> label string; unknown ints -> str(id)
    return [id2label.get(int(i), str(int(i))) for i in ids.tolist()]

def _write_dictionary_dump(path: str, id2label: Dict[int, str]):
    # Match fairseq's "dict.txt" vibe (label and dummy count per line)
    with open(path, "w", encoding="utf-8") as f:
        for idx in sorted(id2label):
            f.write(f"{id2label[idx]} 1\n")

@torch.no_grad()
def predict_and_save_hf_multitask(
    trainer,                       # Hugging Face Trainer
    model,                         # Your MusicBertMultiTaskTokenClassification model
    dataset,                       # test_dataset (HF Dataset)
    config_params,                 # has .targets, .batch_size, etc.
    output_folder: str,            # base folder (like --output-folder)
    dataset_name: str = "test",
    data_dir_for_metadata: str = None,  # path that contains *_raw or *_bin to copy metadata from
    overwrite: bool = False,
    ignore_specials: int = 4,
    compound_token_ratio: int = 8,
    trim_bos_eos: bool = True,
):
    """
    Saves:
      <output_folder>/<dataset_name>/predictions/<target>.txt
      <output_folder>/<dataset_name>/predictions/<target>.h5
      <output_folder>/<dataset_name>/<target>_dictionary.txt
      <output_folder>/<dataset_name>/num_ignored_specials.txt
      <output_folder>/<dataset_name>/metadata_<dataset_name>.txt  (if data_dir_for_metadata provided)
    """

    # 1) Run prediction (no metrics)
    
    outputs = trainer.predict(dataset, metric_key_prefix="test")
    for k, v in outputs.metrics.items():
        print(f"{k}: {v:.4f}")

    logits_all = outputs.predictions  # could be np.ndarray or dict[target] -> np.ndarray
    # Also collect attention masks to compute valid lengths
    # Trainer doesn't return inputs, so pull from dataset directly
    # We'll build a list of per-example valid lengths from attention_mask if present.
    target_names = list(config_params.targets)

    def _to_numpy(x):
        import numpy as np, torch
        if isinstance(x, np.ndarray):
            return x
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        # handles lists/tuples of numbers
        return np.asarray(x)

    is_multitask = getattr(config_params, "multitask", False)

    if is_multitask:
        # HF may return: dict[target]->ndarray  OR list/tuple aligned with targets
        if isinstance(logits_all, dict):
            task_to_logits = {t: _to_numpy(logits_all[t]) for t in target_names}
        elif isinstance(logits_all, (list, tuple)):
            if len(logits_all) != len(target_names):
                raise ValueError(
                    f"Multitask predictions length {len(logits_all)} "
                    f"!= number of targets {len(target_names)}"
                )
            task_to_logits = {t: _to_numpy(x) for t, x in zip(target_names, logits_all)}
        else:
            # Some models still return a single array; treat as single-task by mistake
            raise TypeError(
                f"Expected dict/list of per-target logits for multitask, got {type(logits_all)}"
            )
    else:
        # Single-task: must be one array/tensor
        task_to_logits = {target_names[0]: _to_numpy(logits_all)}
# ---------------------------------------------------------------
    has_attn = hasattr(dataset[0], "keys") and ("attention_mask" in dataset[0] or "mask" in dataset[0])

    # 2) Prepare folders
    dataset_folder = os.path.join(output_folder, dataset_name)
    preds_folder = os.path.join(dataset_folder, "predictions")
    _ensure_dir_clean(dataset_folder, overwrite=overwrite)
    os.makedirs(preds_folder, exist_ok=True)

    # 3) Gather task names and id2label dicts
    is_multitask = hasattr(model.config, "multitask_id2label") and isinstance(model.config.multitask_id2label, dict)
    if is_multitask:
        target_names = list(config_params.targets)
        id2label_map: Dict[str, Dict[int, str]] = model.config.multitask_id2label
    else:
        # single task fallback
        target_names = [getattr(model.config, "task_name", "task0")]
        id2label_map = {target_names[0]: model.config.id2label}

    # 4) Compute valid token lengths per example, reduced to "target length" by compound_token_ratio
    #    We create a vector of n_examples with per-example n_tokens.
    N = len(dataset)
    if has_attn:
        n_valid_tokens = []
        for i in range(N):
            ex = dataset[i]
            attn = ex.get("attention_mask", ex.get("mask", None))
            if attn is None:
                n_valid_tokens.append(len(ex["input_ids"]))
            else:
                n_valid_tokens.append(int(np.asarray(attn).sum()))
        n_target_tokens = np.array(n_valid_tokens)#np.array(n_valid_tokens) // compound_token_ratio
    else:
        # Fallback: use full sequence length from logits (minus specials assumed later by ignore_specials)
        if isinstance(logits_all, dict):
            # choose any target to get T
            any_logits = next(iter(logits_all.values()))
            T = any_logits.shape[1]
        else:
            T = logits_all.shape[1]
        n_target_tokens = np.full(N, T, dtype=int)  # will trim via ignore_specials below

    # 5) Helper to write one task
    def handle_one_task(target: str, task_logits: np.ndarray):
        """
        Improved: handles BOS/EOS trimming consistently with fairseq.
        """
        txt_path = os.path.join(preds_folder, f"{target}.txt")
        h5_path = os.path.join(preds_folder, f"{target}.h5")
        dict_path = os.path.join(dataset_folder, f"{target}_dictionary.txt")

        _write_dictionary_dump(dict_path, id2label_map[target])

        h5f = h5py.File(h5_path, "w")
        try:
            pred_ids = task_logits.argmax(axis=-1)  # [N, T]

            with open(txt_path, "w", encoding="utf-8") as txt_out:
                for idx in range(N):
                    n_tok = int(n_target_tokens[idx])
                    
                    # IMPROVEMENT 1: Apply BOS/EOS trimming if specified
                    if trim_bos_eos:
                        # Remove first and last token: [1:-1]
                        seq_start, seq_end = 1, n_tok - 1 if n_tok > 1 else 1
                    else:
                        seq_start, seq_end = 0, n_tok
                    
                    ids_seq = pred_ids[idx, seq_start:seq_end]
                    
                    # IMPROVEMENT 2: Trim logits for argmax (exact fairseq parity)
                    # fairseq: data[:, ignore_specials:] on the trimmed logits
                    trimmed_logits = task_logits[idx, seq_start:seq_end, ignore_specials:]
                    trimmed_ids = trimmed_logits.argmax(axis=-1) + ignore_specials
                    decoded = _decode_ids_to_tokens(trimmed_ids, id2label_map[target])

                    txt_out.write(" ".join(decoded) + "\n")

                    # IMPROVEMENT 3: Save HDF5 with same trimming as txt
                    # Match fairseq exactly: [1:-1] + [:, ignore_specials:]
                    data = task_logits[idx, seq_start:seq_end, :]
                    if ignore_specials:
                        data = data[:, ignore_specials:]
                    
                    h5f.create_dataset(
                        f"logits_{idx}",
                        data=data.astype(np.float32),
                        #compression="gzip",  # BONUS: compress to save disk space
                        #compression_opts=4
                    )
        finally:
            h5f.close()

    # 6) Dispatch per task
    # if isinstance(logits_all, dict):
    #     # Multitask: logits_all[target] -> [N, T, V]
    #     for t in target_names:
    #         handle_one_task(t, logits_all[t])
    # else:
    #     # Single task
    #     handle_one_task(target_names[0], logits_all)
    for t, task_logits in task_to_logits.items():
        handle_one_task(t, task_logits)

    # 7) Write num_ignored_specials.txt
    with open(os.path.join(dataset_folder, "num_ignored_specials.txt"), "w") as f:
        f.write(str(ignore_specials))

    # 8) Copy metadata_<dataset_name>.txt if a data dir is provided (mirrors fairseq behavior)
    if data_dir_for_metadata:
        meta_name = f"metadata_{dataset_name}.txt"
        # Try <data_dir>/meta, else try switching *_bin -> *_raw like your original script
        meta_src = os.path.join(data_dir_for_metadata, meta_name)
        if not os.path.exists(meta_src):
            # try sibling *_raw
            base = data_dir_for_metadata.rstrip(os.path.sep)
            if base.endswith("_bin"):
                alt = base[:-4] + "_raw"
                meta_src = os.path.join(alt, meta_name)
        if os.path.exists(meta_src):
            shutil.copy(meta_src, os.path.join(dataset_folder, meta_name))

def main(args):
    data_dir, ref_dir, checkpoint, output_folder_base = (
        args.data_dir,
        args.ref_dir,
        args.checkpoint_path,
        args.output_dir_base,
    )
    if ref_dir is None:
        ref_dir = data_dir
    
    output_folder = os.path.join(output_folder_base, args.dataset)

    if os.path.exists(output_folder):
        if args.overwrite:
            shutil.rmtree(output_folder)
        else:
            raise ValueError(f"Output folder {output_folder} already exists")

    #assert data_dir.rstrip(os.path.sep).endswith("_bin")

   
    target_names = args.targets


    
    
    # if args.baseline:
    #     print("loading baseline parameters")
    #     hyperparams_dict = load_baseline_params(args.targets)
    # else:
        # print("evaluating the model from hpo")
        # study = optuna.load_study(study_name="nas_layers_extended_new",                
        #                         storage = "sqlite:///optuna_nas.db")
        # best_trials = study.best_trials
        # params = best_trials[3].params    # 0 is trial 35, 3 is 58
        # hyperparams_dict = create_hyperparams_dict(args.targets, params)
    
    # path = Path(args.checkpoint_path)
    # config = BertConfig.from_pretrained(path, force_download=True) # this ensures that the most
    #                                                                             # up-to-date model is loaded (polina)
    # config.hyperparams =hyperparams_dict
    model, config = load_model(args)
    # if args.conditioning:
    #     model = MusicBertMultiTaskTokenClassConditioned.from_pretrained(pretrained_model_name_or_path =path, config=config) 
    # else:
    #     model = MusicBertMultiTaskTokenClassification.from_pretrained(pretrained_model_name_or_path =path, config=config) 

    test_dataset = get_dataset(args, "test")
    model.config.targets = list(args.targets)
    train_dataset = get_dataset(args, "train")
    #test_dataset = LimitedDataset(test_dataset, limit=  10)
    model.config.multitask_lgitabel2id = train_dataset.stois
    model.config.multitask_id2label = {
        target: {v: k for k, v in train_dataset.stois[target].items()}
        for target in train_dataset.stois
    }
    
    
    test_args = TrainingArguments(
    #output_dir=best_model_dir,
    per_device_eval_batch_size=args.batch_size,
    report_to=None,
    do_train=False,
    do_eval=True,
    )
    
    # compute_metrics_fn = partial(
    # compute_metrics_multitask, task_names=args.targets
    # )
    if getattr(args, "multitask", False):
        compute_metrics_fn = partial(
            compute_metrics_multitask,
            task_names=list(args.targets),
            #multitask_id2label=model.config.multitask_id2label,
        )
    else:
        compute_metrics_fn = compute_metrics
    test_trainer = Trainer(
        model=model,
        args=test_args,
        data_collator=partial(collate_for_musicbert_fn, multitask=True),
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics_fn,
    )

    print("Evaluating model on test set...")
    #test_trainer.evaluate()
    predict_and_save_hf_multitask(
    trainer=test_trainer,
    model=model,
    dataset=test_dataset,
    config_params=args,
    output_folder=args.output_dir_base,             # like --output-folder
    dataset_name="test",                        # "test" | "valid" | "train"
    data_dir_for_metadata=args.data_dir_for_metadata,  # so we can copy metadata_<dataset>.txt
    overwrite=True,
    ignore_specials=4,
    compound_token_ratio=8,
)

if __name__ == "__main__":
    cfg =load_config("scripts/save_params.yaml")
   # args = parser.parse_args()

    main(cfg)