import sys
import os
from dataclasses import dataclass
from functools import partial
from typing import Literal, Sequence
from transformers import BertConfig
from torchinfo import summary
from pathlib import Path
from omegaconf import OmegaConf
from transformers import Trainer, TrainingArguments
import optuna
import json
import os
import torch
from math import ceil
import numpy as np
import random
import pandas as pd
import yaml
from transformers import EarlyStoppingCallback
from torch.utils.data import Subset
import wandb
from torch.utils.data import DataLoader
import pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from musicbert_hf.checkpoints import (
    load_musicbert_multitask_token_classifier_from_fairseq_checkpoint,
    load_musicbert_multitask_token_classifier_with_conditioning_from_fairseq_checkpoint,
    load_musicbert_token_classifier_from_fairseq_checkpoint,

)
from musicbert_hf.data import HDF5Dataset, collate_for_musicbert_fn
from musicbert_hf.metrics import compute_metrics, compute_metrics_multitask
from musicbert_hf.models import freeze_layers, MusicBertTokenClassification, MusicBertMultiTaskTokenClassification
from helpers import set_seed, load_baseline_params, get_dataset, create_hyperparams_dict, load_model
from config import load_config

# @dataclass
# class Config:
#     # data_dir should have train, valid, and test subdirectories
#     data_dir: str
#     #output_dir_base: str
#     #
#     checkpoint: str
#     targets: str | list[str]
#     conditioning: str | None = None
#     #num_workers: int = 4
#     optuna_name: str | None = None
#     seed : int | None = None
#     optuna_storage: str | None = None
#     baseline: str | None = None
#     batch_size : int = 4
#     multitask: bool = False
#     study_name: str | None = None
#     storage: str | None = None
#     trial_number: int | None = None

#     @property
#     def train_dir(self) -> str:
#         return os.path.join(self.data_dir, "train")

#     @property
#     def valid_dir(self) -> str:
#         return os.path.join(self.data_dir, "valid")

#     @property
#     def test_dir(self) -> str:
#         return os.path.join(self.data_dir, "test")

# def load_config(path: str | os.PathLike) -> Config:
#     p = Path(path)
#     if not p.exists():
#         raise FileNotFoundError(f"Config file not found: {p}")
#     data = _load_yaml_(p)
#     # allow hyphenated keys in file
#     data = {k.replace("-", "_"): v for k, v in data.items()}
#     # validate keys
#     valid = set(Config.__annotations__.keys())
#     unknown = set(data) - valid
#     if unknown:
#         raise ValueError(f"Unknown config keys: {sorted(unknown)}")
#     return Config(**data)

args =load_config("scripts/test_params.yaml")
# if args.baseline:
#         print("loading baseline parameters")
#         hyperparams_dict = load_baseline_params(args.targets)
# else:
#     print("evaluating the model from hpo")
#     study = optuna.load_study(study_name="nas_layers_extended_new",                
#                             storage = "sqlite:///optuna_nas.db")
#     best_trials = study.best_trials
#     params = best_trials[3].params    # 0 is trial 35, 3 is 58
#     hyperparams_dict = create_hyperparams_dict(args.targets, params)
# args.multitask = False if len(args.targets)== 1 else True
# path = Path(args.checkpoint)
# config = BertConfig.from_pretrained(path, force_download=True) # this ensures that the most
#                                                                             # up-to-date model is loaded (polina)
# config.hyperparams =hyperparams_dict

# model = MusicBertMultiTaskTokenClassification.from_pretrained(pretrained_model_name_or_path =path, config=config) 

# model.config.multitask_label2id = train_dataset.stois
# model.config.multitask_id2label = {
#     target: {v: k for k, v in train_dataset.stois[target].items()}
#     for target in train_dataset.stois
# }
# label2id = model.config.multitask_label2id
# id2label = model.config.multitask_id2label
model, config = load_model(args)
model.config.targets = list(config.targets)

test_args = TrainingArguments(
#output_dir=best_model_dir,
per_device_eval_batch_size=args.batch_size,
report_to=None,
do_train=False,
do_eval=True,
)

compute_metrics_fn = partial(
compute_metrics_multitask, task_names=args.targets
) if args.multitask else compute_metrics

test_dataset = get_dataset(args, "test")
test_trainer = Trainer(
    model=model,
    args=test_args,
    data_collator=partial(collate_for_musicbert_fn, multitask=args.multitask),
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics_fn,
)

print("Evaluating best model on test set...")
test_results = test_trainer.evaluate()
for k, v in test_results.items():
    print(f"{k}: {v:.4f}")
