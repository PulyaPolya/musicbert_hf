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
from torchinfo import summary
from omegaconf import OmegaConf
from transformers import Trainer, TrainingArguments
import optuna
import json
import os
from transformers import EarlyStoppingCallback
from torch.utils.data import Subset
import wandb
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#from ray.tune.integration.huggingface import TuneReportCallback
from musicbert_hf.checkpoints import (
    load_musicbert_multitask_token_classifier_from_fairseq_checkpoint,
    load_musicbert_multitask_token_classifier_with_conditioning_from_fairseq_checkpoint,
    load_musicbert_token_classifier_from_fairseq_checkpoint,
)
from musicbert_hf.data import HDF5Dataset, collate_for_musicbert_fn
from musicbert_hf.metrics import compute_metrics, compute_metrics_multitask
from musicbert_hf.models import freeze_layers

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
    learning_rate: float = 2.5e-4
    warmup_steps: int = 0
    max_steps: int = -1
    wandb_project: str | None = None
    # If None, freeze all layers; if int, freeze all layers up to and including
    #   the specified layer; if sequence of ints, freeze the specified layers
    freeze_layers: int | Sequence[int] | None = None
    activation_function: str | None = None
    pooler_dropout : int = 0
    # In general, we want to leave job_id as None and set automatically, but for
    #   local testing we can set it manually
    job_id: str | None = None

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
    dataset = HDF5Dataset(
        os.path.join(data_dir, "events.h5"),
        config.target_paths(split),
        conditioning_path=config.conditioning_path(split),
    )
    return dataset


def get_config_and_training_kwargs(config_path=None):
    if config_path:
        file_conf = OmegaConf.load(config_path)
    else:
        file_conf = OmegaConf.create()  
    #conf = OmegaConf.from_cli(sys.argv[1:])
    cli_conf = OmegaConf.from_cli(sys.argv[1:])
    # Merge file config with command-line overrides, with CLI taking precedence
    conf = OmegaConf.merge(file_conf, cli_conf)
    config_fields = set(Config.__dataclass_fields__.keys())
    config_kwargs = {k: v for k, v in conf.items() if k in config_fields}
    training_kwargs = {k: v for k, v in conf.items() if k not in config_fields}
    config = Config(**config_kwargs)  # type:ignore
    return config, training_kwargs

def model_init(model):
    return model


def objective(trial):
    # Load original config from JSON
    _, training_kwargs = get_config_and_training_kwargs(config_path= "scripts/finetune_params.json")

    with open("scripts/finetune_params.json") as f:
        config_dict = json.load(f)
    # Inject trial-suggested hyperparameters
    #config_dict["batch_size"] = trial.suggest_categorical("batch_size", [8, 16, 32])
    #config_dict["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    #config_dict["max_steps"] = trial.suggest_int("max_steps", 50, 100)
    #config_dict["num_epochs"] = trial.suggest_int("num_epochs", 5, 30)
    config_dict["activation_function"] = trial.suggest_categorical("activation_fn", ["tanh"])
    config_dict["pooler_dropout"] = trial.suggest_int("pooler_dropout", 0, 1)
    # Reload config and training_kwargs
    #config, training_kwargs = get_config_and_training_kwargs(config_dict=config_data)
    config = Config(**config_dict)
    # W&B setup
    if config.wandb_project:
        os.environ["WANDB_PROJECT"] = config.wandb_project
    else:
        os.environ.pop("WANDB_PROJECT", None)

    # Prepare dataset
    train_dataset = get_dataset(config, "train")
    valid_dataset = get_dataset(config, "valid")
    #train_dataset = LimitedDataset(train_dataset, limit=100)
    #valid_dataset = LimitedDataset(valid_dataset, limit=100)
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
                {
            "activation_fn": config.activation_function,
            "pooler_dropout": config.pooler_dropout/10,
            "num_linear_layers": 3
        },
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
        model = load_musicbert_token_classifier_from_fairseq_checkpoint(
            {
            "activation_fn": config.activation_function,
            "pooler_dropout": config.pooler_dropout/10,
            "num_linear_layers": 3
        },
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
    # Update training kwargs with trial-specific parameters
    training_kwargs =(
        dict(
        output_dir= config.output_dir,
        num_train_epochs= config.num_epochs,
        per_device_train_batch_size= config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate= config.learning_rate,
        warmup_steps= config.warmup_steps,
        logging_dir= config.log_dir,
        max_steps= config.max_steps,
        eval_strategy= "steps",
        eval_steps= 1000,
        metric_for_best_model= "accuracy",
        greater_is_better= True,
        save_total_limit= 2,
        report_to = "wandb",
        push_to_hub= False,
        eval_on_start= False,
    )| training_kwargs
    )

    training_kwargs["report_to"] = "wandb" #if config.wandb_project else None
    training_args = TrainingArguments(**training_kwargs)

    compute_metrics_fn = partial(
        compute_metrics_multitask, task_names=config.targets
    ) if config.multitask else compute_metrics
    print(f"starting with the model training")
    print(f"max_steps {config.max_steps}")
    
    wandb.init(project="musicbert", name=f"gpu_trial_number_{trial.number}", config={
            "target": "quality",
            "features" : "key",
            "epochs": 100,
            "batch_size": config.batch_size ,
            "max_steps": config.max_steps,
            "lr": config.learning_rate,
            "augmentation": False,
            })
            
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=partial(collate_for_musicbert_fn, multitask=config.multitask),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_loss_func=partial(model.compute_loss),
        compute_metrics=compute_metrics_fn,
        callbacks = [EarlyStoppingCallback(early_stopping_patience =4)]
    )

    trainer.train()
    print(f"evaluating the model")
    eval_result = trainer.evaluate()
    wandb.log({"eval_accuracy": eval_result["eval_accuracy"]})

    return eval_result["eval_accuracy"] 

if __name__ == "__main__":
    print("start")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1)

    print("Best hyperparameters:", study.best_params)
   