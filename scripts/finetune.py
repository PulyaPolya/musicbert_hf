"""
This script can be used to finetune a MusicBERT model on a token classification task.

Config parameters are passed on the command line. They are parsed with OmegaConf and
should have the format `key=value` (e.g., `data_dir=/path/to/data`).

The required parameters are:
- `data_dir`: the directory containing the training data. This directory should have
  `train`, `valid`, and `test` subdirectories, each containing a set of `.h5` files,
  including `events.h5` featuring the octuple-encoded input and one `.h5` file for each
  target and/or conditioning feature. These `.h5` files should have (at least) the
  following contents:
    - `num_seqs`: the number of sequences in the dataset
    - `vocab_size`: the number of tokens in the vocabulary
    - `name`: the name of the feature
    - `vocab`: a JSON-serialized mapping from tokens to integers (e.g.,
      `{"Major": 0, "Minor": 1, "Diminished": 2, ...}`)
    - integer keys between 0 and `num_seqs` - 1: the actual sequences of integer tokens
- `output_dir_base`: the base directory for the output. The final output directory will
  be `output_dir_base/job_id`. If `job_id` is not explicitly set, it is the ID of the
  SLURM job if running on a cluster, or a string of the current time if not.
- `checkpoint_path`: the path to the checkpoint to finetune from.
- `targets`: a target or list of targets to finetune on. We expect each target to have a
  corresponding `.h5` file in the `data_dir` directory. For example, if `targets` is
  `["key", "chord_quality"]`, we expect to find `key.h5` and `chord_quality.h5` in the
  `data_dir/train` directory.

For the full list of config parameters, see the `Config` dataclass below.

"""

import os
import sys
import time
from dataclasses import dataclass
from functools import partial
from typing import Literal, Sequence
from transformers import BertConfig
from torchinfo import summary
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
from transformers import EarlyStoppingCallback
from torch.utils.data import Subset
import wandb
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from musicbert_hf.checkpoints import (
    load_musicbert_multitask_token_classifier_from_fairseq_checkpoint,
    load_musicbert_multitask_token_classifier_with_conditioning_from_fairseq_checkpoint,
    load_musicbert_token_classifier_from_fairseq_checkpoint,
)
from musicbert_hf.data import HDF5Dataset, collate_for_musicbert_fn
from musicbert_hf.metrics import compute_metrics, compute_metrics_multitask
from musicbert_hf.models import freeze_layers, MusicBertTokenClassification, MusicBertMultiTaskTokenClassification
from optuna.pruners import MedianPruner, BasePruner

import signal

class TrialTimeout(Exception):
    pass

def alarm_handler(signum, frame):
    raise TrialTimeout()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For full reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
gpu = torch.cuda.is_available()
@dataclass
class Config:
    # data_dir should have train, valid, and test subdirectories
    data_dir: str
    output_dir_base: str
    checkpoint_path: str
    targets: str | list[str]
    conditioning: str | None = None
    log_dir: str = os.path.expanduser("logs/musicbert_hf_logs")
    # We will always load from a checkpoint so we don't need to specify architecture
    # architecture: Literal["base", "tiny"] = "base"
    num_epochs: int = 0
    batch_size: int = 4
    num_workers: int = 4
    learning_rate: float = 2.5e-4
    warmup_steps: int = 0
    max_steps: int = -1
    wandb_project: str | None = None
    wandb_name: str | None = None
    optuna_name: str | None = None
    # time limit for each trial in the optuna run
    time_limit : str | None = None
    # If None, freeze all layers; if int, freeze all layers up to and including
    #   the specified layer; if sequence of ints, freeze the specified layers
    freeze_layers: int | Sequence[int] | None = None
    # In general, we want to leave job_id as None and set automatically, but for
    #   local testing we can set it manually
    job_id: str | None = None
    hf_repository: str | None = None
    hf_token: str | None = None
    TESTING: bool = True
    RUN_NAS : bool = False
    num_trials: int = 1

    def __post_init__(self):
        assert self.num_epochs is not None or self.max_steps is not None, (
            "Either num_epochs or max_steps must be provided"
        )
        if self.job_id is None:
            self.job_id = os.environ.get("SLURM_JOB_ID", None)
            if self.job_id is None:
                # Use the current time as the job ID if not running on the cluster
                self.job_id = str(int(time.time()))

        if isinstance(self.targets, str):
            self.targets = [self.targets]

    @property
    def train_dir(self) -> str:
        return os.path.join(self.data_dir, "train")

    @property
    def valid_dir(self) -> str:
        return os.path.join(self.data_dir, "valid")

    @property
    def test_dir(self) -> str:
        return os.path.join(self.data_dir, "test")

    @property
    def output_dir(self) -> str:
        return os.path.join(self.output_dir_base, self.job_id)

    def target_paths(self, split: Literal["train", "valid", "test"]) -> list[str]:
        return [
            os.path.join(self.data_dir, split, f"{target}.h5")
            for target in self.targets
        ]

    def conditioning_path(self, split: Literal["train", "valid", "test"]) -> str | None:
        return (
            None
            if not self.conditioning
            else os.path.join(self.data_dir, split, f"{self.conditioning}.h5")
        )

    @property
    def multitask(self) -> bool:
        return len(self.targets) > 1


def get_dataset(config, split):
    data_dir = getattr(config, f"{split}_dir")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading to device {device}")
    dataset = HDF5Dataset(
        os.path.join(data_dir, "events.h5"),
        config.target_paths(split),
        conditioning_path=config.conditioning_path(split),
        device = device
    )
    return dataset


def get_config_and_training_kwargs(config_path=None):
    if config_path:
        file_conf = OmegaConf.load(config_path)
    else:
        file_conf = OmegaConf.create()  
    cli_conf = OmegaConf.from_cli(sys.argv[1:])
    # Merge file config with command-line overrides, with CLI taking precedence
    conf = OmegaConf.merge(file_conf, cli_conf)
    config_fields = set(Config.__dataclass_fields__.keys())
    config_kwargs = {k: v for k, v in conf.items() if k in config_fields}
    training_kwargs = {k: v for k, v in conf.items() if k not in config_fields}
    config = Config(**config_kwargs)  # type:ignore
    return config, training_kwargs

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

def create_dataloader(config, split, batch_size=4, num_workers=4, shuffle=True, dtype=None, device=None):

    dataset = get_dataset(config, split)
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            num_workers=num_workers, 
                            shuffle=shuffle,  
                            pin_memory=True)  
    return dataloader

def make_objective(config, train_dataset, valid_dataset, test_dataset, time_limit):
    def objective(trial):
    # Load original config from JSON
        _, training_kwargs = get_config_and_training_kwargs(config_path= "scripts/finetune_params.json")
        
        if config.wandb_project:
            os.environ["WANDB_PROJECT"] = config.wandb_project
        else:
            os.environ.pop("WANDB_PROJECT", None)
        
        if config.RUN_NAS:
            seed = 42
            hyperparams_dict = {}
            parameters = {"num_linear_layers": [1, 6], "activation_fn": ["tanh", "relu", "gelu"],
                        "pooler_dropout": [0.0, 0.5], "normalisation" :  ["none", "layer"], "linear_layers_dim":[32, 768] }
            for target in (config.targets):
                MIN_LAYERS, MAX_LAYERS, = parameters["num_linear_layers"][0], parameters["num_linear_layers"][1]
                target_params = {}
                # First choose num_linear_layers to use in later loops
                target_params["num_linear_layers"] = trial.suggest_int(
                    f"num_linear_layers_{target}",
                    parameters["num_linear_layers"][0],
                    parameters["num_linear_layers"][1],
                )
                num_layers = target_params["num_linear_layers"]
                #max_dim = max(768, train_dataset.vocab_sizes[config.targets.index(target)] )
                #min_dim = min (32, train_dataset.vocab_sizes[config.targets.index(target)])
                target_params["linear_layers_dim"]  = [
                    trial.suggest_int(f"layer_dim_{target}_{i}", parameters["linear_layers_dim"][0], parameters["linear_layers_dim"][1])
                    for i in range(MAX_LAYERS)
                ][:num_layers]
                # Activation function per layer
                target_params["activation_fn"] = [
                    trial.suggest_categorical(f"activation_fn_{target}_{i}", parameters["activation_fn"])
                    for i in range(MAX_LAYERS)
                ][:num_layers]
                # Dropout per layer
                target_params["pooler_dropout"] = [
                    trial.suggest_float(f"pooler_dropout_{target}_{i}", parameters["pooler_dropout"][0], parameters["pooler_dropout"][1])
                    for i in range(MAX_LAYERS)
                ][:num_layers]
                target_params["normalisation"] = [
                    trial.suggest_categorical(f"normalisation_{target}_{i}", parameters["normalisation"])
                    for i in range(MAX_LAYERS)
                ][:num_layers]
                hyperparams_dict[target] = target_params
            config.freeze_layers = trial.suggest_int(f"freeze_layers", 6, 11)
            config.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log = True)
            config.batch_size =trial.suggest_categorical("batch_size",[4, 8, 16, 32])
        else:
            with open("best_summary.json") as f:
                best_params_dict = json.load(f)
                hyperparams_dict  = get_best_params_from_dict(best_params_dict, "inversion")
                seed = random.randint(0, 2**31-1)
        set_seed(seed)  
        long_degree = "primary_alteration_primary_degree_secondary_alteration_secondary_degree"
        hyperparams_df = pd.DataFrame.from_dict(hyperparams_dict).T
        hyperparams_df.rename(index = {long_degree: "degree"}, inplace=True)
        print("Chosen hyperparameters")
        print(hyperparams_df)
        print(f"number of frozen layers: {config.freeze_layers},  learning_rate: {config.learning_rate}, batch_size {config.batch_size}")
        # Load model
        if not config.checkpoint_path:
            raise ValueError("checkpoint_path must be provided")

        if config.multitask:
            if config.conditioning:
                model = load_musicbert_multitask_token_classifier_with_conditioning_from_fairseq_checkpoint(
                    config.checkpoint_path,
                    checkpoint_type="musicbert",
                    num_labels=train_dataset.vocab_sizes,
                    z_vocab_size=train_dataset.conditioning_vocab_size,
                )
            else:
                model = load_musicbert_multitask_token_classifier_from_fairseq_checkpoint(
                    hyperparams_dict,
                    config.checkpoint_path,
                    checkpoint_type="musicbert",
                    num_labels=train_dataset.vocab_sizes,
                )
            model.config.multitask_label2id = train_dataset.stois
            model.config.multitask_id2label = {
                target: {v: k for k, v in train_dataset.stois[target].items()}
                for target in train_dataset.stois
            }
        else:
            if config.conditioning:
                raise NotImplementedError("Conditioning not supported in single-task mode") 
            model = load_musicbert_token_classifier_from_fairseq_checkpoint(   # end up here when we only have one task(Polina)
                hyperparams_dict,
                config.checkpoint_path,
                checkpoint_type="musicbert",
                num_labels=train_dataset.vocab_sizes[0],
                
            )
            model.config.label2id = list(train_dataset.stois.values())[0]
            model.config.id2label = {v: k for k, v in model.config.label2id.items()}

        model.config.targets = list(config.targets)
        if config.conditioning:
            model.config.conditioning = config.conditioning

        freeze_layers(model, config.freeze_layers)
        summary(model)
        # originally evaluate every 1000 steps, adjust for different batch sizes
        eval_steps = 5 if TESTING else max(4000 // config.batch_size, 200)
        #evals_per_epoch = 25

        # how many updates in one epoch for this batch size?
        #steps_per_epoch = ceil(len(train_dataset) / config.batch_size)

        # then space your evaluations evenly:
        #eval_steps = ceil(steps_per_epoch / evals_per_epoch)
        if TESTING:
            config.max_steps = 10
            config.warmup_steps = 2
            config.batch_size = 4
        push_to_hub = False #False if TESTING else True

        # Update training kwargs with trial-specific parameters
        #max_steps = int(config.max_steps/ (config.batch_size / 4)) # making sure that the number of training steps in total is the same
        
        training_kwargs =(
            dict(
            output_dir= config.output_dir,
            num_train_epochs= config.num_epochs,
            per_device_train_batch_size= config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            learning_rate= config.learning_rate,
            warmup_steps= config.warmup_steps,
            logging_dir= config.log_dir,
            eval_strategy= "steps",
            eval_steps= eval_steps,   
            save_steps = eval_steps, 
            fp16=gpu, 
            max_steps = config.max_steps,
            load_best_model_at_end = True,
            metric_for_best_model= "accuracy",
            greater_is_better= True,
            save_total_limit= 2,
            save_strategy = "steps",
            push_to_hub= push_to_hub,
            hub_model_id = config.hf_repository,
            eval_on_start= False,
            seed = seed
        )| training_kwargs
        )

        training_kwargs["report_to"] = "wandb" if config.wandb_name else None
        training_args = TrainingArguments(**training_kwargs)

        compute_metrics_fn = partial(
            compute_metrics_multitask, task_names=config.targets
        ) if config.multitask else compute_metrics
        if config.wandb_name:
            params_to_log = ["freeze_layers", "batch_size", "learning_rate"]
            hyperparams_dict.update( { key:getattr(config, key)  for key in params_to_log if hasattr(config, key)})
            if config.RUN_NAS:
                name = f"trial_{trial.number}"
                group= config.wandb_name
            else:
                name = config.wandb_name
                group = None
            wandb.init(project="musicbert", name=name, group=group, config=hyperparams_dict, reinit= True)
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=partial(collate_for_musicbert_fn, multitask=config.multitask),
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            compute_loss_func=partial(model.compute_loss),
            compute_metrics=compute_metrics_fn,
            callbacks = [EarlyStoppingCallback(early_stopping_patience =5)]
        )
        print(model.device)
        #print(f"trial{trial.number} before")
        #print("Model hash before training:", hash(tuple(p.data_ptr() for p in model.parameters())))
        signal.signal(signal.SIGALRM, alarm_handler)
        signal.alarm(time_limit*60)  # converting minutes to seconds
        try :
            trainer.train()
        except TrialTimeout:
            raise optuna.TrialPruned()
        finally:
            signal.alarm(0)
            print(f"evaluating the model")
            eval_result = trainer.evaluate()
            if config.wandb_name:
                accuracies = [eval_result[f"eval_{target}_accuracy"] for target in config.targets]
                wandb.log({f"eval_{target}_accuracy": eval_result[f"eval_{target}_accuracy"],
                        "seed":seed
                        })
                wandb.finish()
            return accuracies
    return objective    

if __name__ == "__main__":
    with open("scripts/finetune_params.json") as f:
        config_dict = json.load(f)
    config_params = Config(**config_dict)
    os.environ["HF_TOKEN"] = config_params.hf_token
    global TESTING  
    TESTING  = config_params.TESTING
    test_dataset = get_dataset(config_params, "test") 
    train_dataset = get_dataset(config_params, "train")
    valid_dataset = get_dataset(config_params, "valid")  
    #train_dataloader = create_dataloader(config_params, "train",shuffle=True, batch_size=config_params.batch_size, num_workers =config_params.num_workers)
    #valid_dataloader = create_dataloader(config_params, "valid", shuffle=False,batch_size=config_params.batch_size, num_workers =config_params.num_workers)

    if TESTING:
        #train_dataset = get_dataset(config_params, "train")
        #valid_dataset = get_dataset(config_params, "valid") 
        train_dataset = LimitedDataset(train_dataset, limit=30)
        valid_dataset = LimitedDataset(valid_dataset, limit=20)
        test_dataset = LimitedDataset(test_dataset, limit=20)
    #"""
    if not config_params.RUN_NAS:
            config_params.num_trials = 1
    median_pruner = optuna.pruners.MedianPruner(n_warmup_steps=0)
    sampler = optuna.samplers.TPESampler(seed=42, 
                                         multivariate=True,
                                         warn_independent_sampling=False)
    study = optuna.create_study(study_name=config_params.optuna_name,
                                # in case of 4 classification tasks
                                directions= ["maximize", "maximize","maximize", "maximize"],
                                sampler = sampler,
                                pruner = median_pruner,
                                storage = "sqlite:///optuna.db",
                                load_if_exists=True )
    # adding 2.5 h time limit
    study.optimize(make_objective(config_params, train_dataset, valid_dataset, test_dataset, time_limit=config_params.time_limit), n_trials=config_params.num_trials)
    if config_params.RUN_NAS:
        best_trials = study.best_trials
        best_summaries = []
        for trial in study.best_trials:
            best_summaries.append( {
            "trial_number": trial.number,
            **trial.params,
            "objectives": trial.values

            })
        
        with open ("best_summaries.json", "w") as f:
            json.dump(best_summaries, f, indent = 4)

    """
    print("evaluating the model from hf")
    with open('best_summary.json') as json_file:
        best_trial = json.load(json_file)
    
    
    
    #tsest_dataset = LimitedDataset(test_dataset, limit=10)
    config_dict["hyperparams"] = best_trial
    config = BertConfig.from_pretrained(config_dict["hf_repository"], force_download=True) # thus ensures that the most
                                                                                # up-to-date model is loaded (polina)
    #config.hyperparams = best_trial
    #config_dict.update(best_trial) # best_trial.params
    model = MusicBertTokenClassification.from_pretrained(config_dict["hf_repository"], config= config)   #config_dict["hf_repository"],
    

    model.config.multitask_label2id = train_dataset.stois
    model.config.multitask_id2label = {
        target: {v: k for k, v in train_dataset.stois[target].items()}
        for target in train_dataset.stois
    }
    model.config.targets = list(config_params.targets)
    
    test_args = TrainingArguments(
    #output_dir=best_model_dir,
    per_device_eval_batch_size=config_params.batch_size,
    report_to=None,
    do_train=False,
    do_eval=True,
    )
    
    compute_metrics_fn = partial(
    compute_metrics_multitask, task_names=config_params.targets
    ) if config_params.multitask else compute_metrics
    del train_dataset
    test_trainer = Trainer(
        model=model,
        args=test_args,
        data_collator=partial(collate_for_musicbert_fn, multitask=config_params.multitask),
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics_fn,
    )

    print("Evaluating best model on test set...")
    test_results = test_trainer.evaluate()
    for k, v in test_results.items():
        print(f"{k}: {v:.4f}")
   """