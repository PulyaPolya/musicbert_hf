import os
import sys
import time
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
from transformers import EarlyStoppingCallback, TrainerCallback
#from transformers import T
from torch.utils.data import Subset
import wandb
from torch.utils.data import DataLoader
import pickle
from musicbert_hf.data import HDF5Dataset
from musicbert_hf.models import freeze_layers, MusicBertTokenClassification, MusicBertMultiTaskTokenClassification, MusicBertMultiTaskTokenClassConditioned

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For full reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_best_params_from_dict(best_params_dict, target):   # if we want to run the best model, here we can read all parameters from the file
    hyperparams_dict = {}
    num_layers = best_params_dict[f"num_linear_layers_{target}"] # TODO expand for multiple targets too
    
    target_params = {
    "linear_layers_dim": [],
    "activation_fn": [],
    "pooler_dropout": [],
    "normalisation": []
        }                                           # so far it will only work for one target 
    target_params["num_linear_layers"] = best_params_dict[f"num_linear_layers_{target}"]
    for i in range(num_layers):
        target_params["linear_layers_dim"].append(best_params_dict[f"layer_dim_{target}_{i}"])
        target_params["linear_layers_dim"].sort(reverse = True)
        target_params["activation_fn"].append(best_params_dict[f"activation_fn_{target}_{i}"])
        target_params["pooler_dropout"].append(best_params_dict[f"pooler_dropout_{target}_{i}"])
        target_params["pooler_dropout"].sort(reverse = True)
        target_params["normalisation"].append(best_params_dict[f"normalisation_{target}_{i}"])  
    hyperparams_dict[target] = target_params
    return hyperparams_dict

def create_hyperparams_dict(targets, params):
    hyperparams_dict = {}
    for target in (targets):
        target_params = {}
        # First choose num_linear_layers to use in later loops
        target_params["num_linear_layers"] =  params[f"num_linear_layers_{target}"]
        num_layers = target_params["num_linear_layers"]
        target_params["linear_layers_dim"]  = [
                params[f"layer_dim_{target}_{i}"] for i in range(num_layers)
        ]
        # Activation function per layer
        target_params["activation_fn"] = [ 
             params[f"activation_fn_{target}_{i}"] for i in range(num_layers)
        ]
        target_params["input_dropout"] =0 #params[f"input_dropout_{target}"]
        # Dropout per layer
        target_params["pooler_dropout"] = [
             params[f"pooler_dropout_{target}_{i}"] for i in range(num_layers)
        ]
        target_params["normalisation"] = [
            params[f"normalisation_{target}_{i}"] for i in range(num_layers)
        ]
        hyperparams_dict[target] = target_params
    return hyperparams_dict


def load_baseline_params(targets):

    hyperparams_dict = {}
    for target in (targets):
        target_params = {}
        # First choose num_linear_layers to use in later loops
        target_params["num_linear_layers"] = 1
        target_params["linear_layers_dim"]  = [ 768]*2
        # Activation function per layer
        target_params["activation_fn"] = [ "tanh"]*2
        # Input dropout
        target_params["input_dropout"] = 0
        # Dropout per layer
        target_params["pooler_dropout"] = [0] * 2
        target_params["normalisation"] = ["none"]*2
        hyperparams_dict[target] = target_params
    return hyperparams_dict

class LimitedDataset:
    def __init__(self, base_dataset, limit):
        self.base_dataset = base_dataset
        self.limit = limit

        # Copy over needed attributes
        self.vocab_sizes = base_dataset.vocab_sizes
        self.stois = base_dataset.stois
        self.conditioning_vocab_size = getattr(base_dataset, "conditioning_vocab_size", None)

    def __getitem__(self, index):
        if index >= self.limit:
            raise IndexError("Index out of range for LimitedDataset")
        return self.base_dataset[index]

    def __len__(self):
        return min(self.limit, len(self.base_dataset))


def get_dataset(config, split):
    data_dir = getattr(config, f"{split}_dir")
    device = torch.device("cpu")
    print(f"loading to device {device}")
    dataset = HDF5Dataset(
        os.path.join(data_dir, "events.h5"),
        config.target_paths(split),
        conditioning_path=config.conditioning_path(split),
        device = device
    )
    return dataset

def load_model(args):
    if args.baseline:
        print("loading baseline parameters")
        hyperparams_dict = load_baseline_params(args.targets)
    else:
        print("evaluating the model from hpo")
        study = optuna.load_study(study_name= args.study_name,                
                                storage = args.storage)
        #best_trials = study.best_trials
        trial= study.trials[args.trial_number]
        params = trial.params    # 0 is trial 35, 3 is 58
        hyperparams_dict = create_hyperparams_dict(args.targets, params)
    
    path = Path(args.checkpoint_path)
    config = BertConfig.from_pretrained(path, force_download=True) # this ensures that the most
                                                                                # up-to-date model is loaded (polina)
    config.hyperparams =hyperparams_dict
    if args.conditioning:
        model = MusicBertMultiTaskTokenClassConditioned.from_pretrained(pretrained_model_name_or_path =path, config=config) 
    else:
        model = MusicBertMultiTaskTokenClassification.from_pretrained(pretrained_model_name_or_path =path, config=config) 
    return model, config