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
from torch.utils.data import Subset
import wandb
from torch.utils.data import DataLoader
import pickle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helpers import  get_best_params_from_dict, set_seed, LimitedDataset, get_dataset

from musicbert_hf.checkpoints import (
    load_musicbert_multitask_token_classifier_from_fairseq_checkpoint,
    load_musicbert_multitask_token_classifier_with_conditioning_from_fairseq_checkpoint,
    load_musicbert_token_classifier_from_fairseq_checkpoint
)
from musicbert_hf.data import HDF5Dataset, collate_for_musicbert_fn
from musicbert_hf.metrics import compute_metrics, compute_metrics_multitask
from musicbert_hf.models import freeze_layers, MusicBertTokenClassification, MusicBertMultiTaskTokenClassification
from optuna.pruners import MedianPruner, BasePruner
from config import load_config
gpu = torch.cuda.is_available()

    
class OptunaTransformersPruningCallback(TrainerCallback):
    def __init__(self, trial, monitor="eval_loss"):
        self.trial = trial
        self.monitor = monitor

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or self.monitor not in metrics:
            return

        value = metrics[self.monitor]
        self.trial.report(value, step=state.global_step)

        if self.trial.should_prune():
            raise optuna.TrialPruned(f"Pruned at step {state.global_step} with {self.monitor}={value}")
            
class SaveSamplerCallback:
    """Saving sampler state after each trial in case we stop before finshing the main loop"""
    def __init__(self, filename):
        self.filename = filename
    
    def __call__(self, study, trial):
        # Called after each trial completes
        print(f"Saving current sampler state")
        with open(self.filename, "wb") as fout:
            pickle.dump(study.sampler, fout)



def create_dataloader(config, split, batch_size=4, num_workers=4, shuffle=True, dtype=None, device=None):

    dataset = get_dataset(config, split)
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            num_workers=num_workers, 
                            shuffle=shuffle,  
                            pin_memory=True)  
    return dataloader

def save_sampler_callback(study, trial):
    save_dir = Path("sampler")  
    save_dir.mkdir(parents=True, exist_ok=True)
    final_path = save_dir / f"sampler_{study.study_name}.pkl"
    tmp_path = final_path.with_suffix(".tmp")

    # atomic write to avoid partial/0-byte files
    with open(tmp_path, "wb") as f:
        pickle.dump(study.sampler, f)
    os.replace(tmp_path, final_path)  # atomic on POSIX

def choose_hyperparameters(trial, config, parameters = None):
    long_degree = "primary_alteration_primary_degree_secondary_alteration_secondary_degree"
    hyperparams_dict = {}
    if not parameters:
        parameters = {"num_linear_layers": [1, 6], "activation_fn": ["tanh", "relu", "gelu"],
                    "pooler_dropout": [0.0, 0.5], "normalisation" :  ["none", "layer"], "linear_layers_dim":[32, 768] }
    for target in (config.targets):
        MIN_LAYERS, MAX_LAYERS, = parameters["num_linear_layers"][0], parameters["num_linear_layers"][1]
        target_params = {}
        # First choose num_linear_layers to use in later loops
        target_params["num_linear_layers"] =  trial.suggest_int(
            f"num_linear_layers_{target}",
            parameters["num_linear_layers"][0],
            parameters["num_linear_layers"][1],
        )
        num_layers = target_params["num_linear_layers"]
        # Layer dimension per layer
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
        target_params["input_dropout"] = trial.suggest_float(f"input_dropout_{target}",  parameters["pooler_dropout"][0], parameters["pooler_dropout"][1])
        target_params["pooler_dropout"] = [
            trial.suggest_float(f"pooler_dropout_{target}_{i}", parameters["pooler_dropout"][0], parameters["pooler_dropout"][1])
            for i in range(MAX_LAYERS)
        ][:num_layers]
        target_params["normalisation"] = [
            trial.suggest_categorical(f"normalisation_{target}_{i}", parameters["normalisation"])
            for i in range(MAX_LAYERS)
        ][:num_layers]
        hyperparams_dict[target] = target_params
    hyperparams_dict["freeze_layers"] = trial.suggest_int(f"freeze_layers", 0, 11)
    hyperparams_dict["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    hyperparams_dict["batch_size"] = trial.suggest_categorical("batch_size",[4, 8, 16, 32])   #!!!!!!!!!
    hyperparams_df = pd.DataFrame.from_dict(hyperparams_dict).T
    hyperparams_df.rename(index = {long_degree: "degree"}, inplace=True)
    print("Chosen hyperparameters")
    print(hyperparams_df)
    return hyperparams_dict
def make_objective(config, train_dataset, valid_dataset):
    def objective(trial):
    # Load original config from JSON
        
        #_, training_kwargs = get_config_and_training_kwargs(config_path= "scripts/finetune_params.json")
        hyperparams_dict = choose_hyperparameters(trial, config)
        if config.wandb_project:
            os.environ["WANDB_PROJECT"] = config.wandb_project
        else:
            os.environ.pop("WANDB_PROJECT", None)
        
        set_seed(config.seed)  
        #print(f"number of frozen layers: {config.freeze_layers},  learning_rate: {config.learning_rate}, batch_size {config.batch_size}")
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
                    hyperparams_config=hyperparams_dict
                )
            else:
                # so far nas works only for this case
                model = load_musicbert_multitask_token_classifier_from_fairseq_checkpoint(
                    
                    config.checkpoint_path,
                    checkpoint_type="musicbert",
                    num_labels=train_dataset.vocab_sizes,
                    hyperparams_config=hyperparams_dict
                )
            model.config.multitask_label2id = train_dataset.stois
            model.config.multitask_id2label = {
                target: {v: k for k, v in train_dataset.stois[target].items()}
                for target in train_dataset.stois
            }
        else:
            if config.conditioning:
                raise NotImplementedError("Conditioning not supported in single-task mode") 
            model = load_musicbert_token_classifier_from_fairseq_checkpoint(   
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

        freeze_layers(model, hyperparams_dict["freeze_layers"])
        summary(model)
        # originally evaluate every 1000 steps, adjust for different batch sizes
        eval_steps = 5 if config.DEBUG else 4000 // hyperparams_dict["batch_size"]
        if config.DEBUG:
            config.max_steps, config.warmup_steps, config.batch_size = 1, 2, 4
        push_to_hub = False #False if TESTING else True

        # Update training kwargs with trial-specific parameters
        #max_steps = int(config.max_steps/ (config.batch_size / 4)) # making sure that the number of training steps in total is the same
        trial_dir = os.path.join(config.output_dir_base, config.optuna_name, f"trial_{trial.number:04d}")
        os.makedirs(trial_dir, exist_ok=True)
        training_kwargs =(
            dict(
            output_dir= trial_dir,
            num_train_epochs= config.num_epochs,
            per_device_train_batch_size= hyperparams_dict["batch_size"],
            per_device_eval_batch_size=hyperparams_dict["batch_size"],
            learning_rate= hyperparams_dict["learning_rate"],
            warmup_steps= config.warmup_steps,
            logging_dir= config.log_dir,
            eval_strategy= "steps",
            eval_steps= eval_steps,   
            save_steps = eval_steps, 
            fp16=gpu, 
            max_steps = config.max_steps,
            load_best_model_at_end = True,
            metric_for_best_model= "eval_accuracy",
            greater_is_better= True,
            save_total_limit= 1,
            save_strategy = "steps",
            push_to_hub= push_to_hub,
            hub_model_id = config.hf_repository,
            eval_on_start= False,
            seed = config.seed,
            dataloader_num_workers=16,
            dataloader_pin_memory = True,
            dataloader_persistent_workers = True
        )#| training_kwargs
        )

        training_kwargs["report_to"] = "wandb" if config.wandb_name else None
        training_args = TrainingArguments(**training_kwargs)

        compute_metrics_fn = partial(
            compute_metrics_multitask, task_names=config.targets, #multitask_id2label = model.config.multitask_id2label
        ) if config.multitask else compute_metrics
        if config.wandb_name:
         
            name = f"trial_{trial.number}"
            group= config.wandb_name
            wandb.init(project="musicbert", name=name, group=group, config=hyperparams_dict, reinit= True)
            wandb.config.update({"seed": config.seed}, allow_val_change=True)
        pruning_callback = OptunaTransformersPruningCallback(
                trial=trial,
                monitor="eval_accuracy",
                )
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=partial(collate_for_musicbert_fn, multitask=config.multitask),
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            compute_loss_func=partial(model.compute_loss),
            compute_metrics=compute_metrics_fn,
            callbacks = [EarlyStoppingCallback(early_stopping_patience =5),pruning_callback]
        )
        print(model.device)
        try:
            trainer.train()
            eval_result = trainer.evaluate()
            accuracies = [eval_result[f"eval_{target}_accuracy"] for target in config.targets]
            eval_acc = eval_result[f"eval_accuracy"]
            log_dict = {f"eval_{target}_accuracy": eval_result[f"eval_{target}_accuracy"] 
            for target in config.targets}
            if config.wandb_name:
                log_dict["seed"] = config.seed
                wandb.log(log_dict)
            return eval_acc
        except optuna.TrialPruned:
            print(f"Trial {trial.number} was pruned")
            raise  
        finally:
            if config.wandb_name:
                wandb.finish()
    return objective    

if __name__ == "__main__":
    config =load_config("scripts/hpo_parameters.yaml")
    #os.environ["HF_TOKEN"] = config.hf_token
    if not config.seed:
        config.seed = random.randint(0, 2**31-1)
    set_seed(config.seed)
        
    test_dataset = get_dataset(config, "test") 
    train_dataset = get_dataset(config, "train")
    valid_dataset = get_dataset(config, "valid")  

    if config.DEBUG:
        train_dataset = LimitedDataset(train_dataset, limit=5)
        valid_dataset = LimitedDataset(valid_dataset, limit=10)
        test_dataset = LimitedDataset(test_dataset, limit=20)

    median_pruner = optuna.pruners.MedianPruner(n_warmup_steps=config.warmup_steps)
   
    if config.sampler_path:
        sampler = pickle.load(open(config.sampler_path,  "rb"))
    else:
         sampler = optuna.samplers.TPESampler(seed=config.seed, 
                                         multivariate=True,
                                         warn_independent_sampling=False)
    study = optuna.create_study(study_name=config.optuna_name,
                                # in case of 4 classification tasks
                                directions= ["maximize"],
                                sampler = sampler,
                                pruner = median_pruner,
                                storage = f"sqlite:///{config.optuna_storage}.db",
                                load_if_exists=True )

    study.optimize(make_objective(config, train_dataset, valid_dataset), n_trials=config.num_trials, callbacks=[SaveSamplerCallback(f"sampler_{config.optuna_name}.pkl")])
    with open(f"sampler_{config.optuna_name}.pkl", "wb") as fout:
        pickle.dump(study.sampler, fout)

