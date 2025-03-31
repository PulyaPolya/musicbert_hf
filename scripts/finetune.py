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
import ray
from ray import tune
#from ray.tune.integration.huggingface import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler
from musicbert_hf.checkpoints import (
    load_musicbert_multitask_token_classifier_from_fairseq_checkpoint,
    load_musicbert_multitask_token_classifier_with_conditioning_from_fairseq_checkpoint,
    load_musicbert_token_classifier_from_fairseq_checkpoint,
)
from musicbert_hf.data import HDF5Dataset, collate_for_musicbert_fn
from musicbert_hf.metrics import compute_metrics, compute_metrics_multitask
from musicbert_hf.models import freeze_layers


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


def hp_space(trial):
    return {
        "inner_dim": tune.choice([128, 256, 512]),
        "pooler_dropout": tune.uniform(0.1, 0.5),
        "num_epochs": tune.choice([3, 5, 10]),
    }

def trainer_factory(training_args, model):
    return Trainer(
        model_init=partial(model_init, model),
        args=training_args,
        data_collator=partial(collate_for_musicbert_fn, multitask=config.multitask),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_loss_func=partial(model.compute_loss),
        compute_metrics=compute_metrics_fn,
    )

def model_init(model):
    return model

if __name__ == "__main__":
    config, training_kwargs = get_config_and_training_kwargs(config_path= "scripts/finetune_params.json")

    if config.wandb_project:
        os.environ["WANDB_PROJECT"] = config.wandb_project
    else:
        os.environ.pop("WANDB_PROJECT", None)
        # os.environ["WANDB_DISABLED"] = "true"

        # Uncomment to turn on model checkpointing (up to 100Gb)
        # os.environ["WANDB_LOG_MODEL"] = "checkpoint"

    train_dataset = get_dataset(config, "train")
    valid_dataset = get_dataset(config, "valid")

    if config.checkpoint_path:
        if config.multitask:
            if config.conditioning:
                model = load_musicbert_multitask_token_classifier_with_conditioning_from_fairseq_checkpoint(   #already here the model has the same structure as rnbert has
                    config.checkpoint_path,
                    checkpoint_type="musicbert",
                    num_labels=train_dataset.vocab_sizes,
                    z_vocab_size=train_dataset.conditioning_vocab_size,
                )
            else:
                model = (
                    load_musicbert_multitask_token_classifier_from_fairseq_checkpoint(
                        config.checkpoint_path,
                        checkpoint_type="musicbert",
                        num_labels=train_dataset.vocab_sizes,
                    )
                )
            model.config.multitask_label2id = train_dataset.stois
            model.config.multitask_id2label = {
                target: {
                    v: k for k, v in model.config.multitask_label2id[target].items()
                }
                for target in train_dataset.stois
            }
        else:
            if config.conditioning:
                raise NotImplementedError(
                    "Conditioning is not yet implemented for single-task training"
                )
            else:
                model = load_musicbert_token_classifier_from_fairseq_checkpoint(
                    config.checkpoint_path,
                    checkpoint_type="musicbert",
                    num_labels=train_dataset.vocab_sizes[0],
                )
            model.config.label2id = list(train_dataset.stois.values())[0]
            model.config.id2label = {v: k for k, v in model.config.label2id.items()}
    else:
        raise ValueError("checkpoint_path must be provided")

    model.config.targets = list(config.targets)
    if config.conditioning:
        model.config.conditioning = config.conditioning

    freeze_layers(model, config.freeze_layers)
    summary(model)
    total_layers = sum ( 1 for _ in model.named_modules())
    print(f"total number of layers {total_layers}")
    training_kwargs = (
        dict(
            output_dir=config.output_dir,
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            warmup_steps=config.warmup_steps,
            logging_dir=config.log_dir,
            max_steps=config.max_steps,
            push_to_hub=False,
            eval_on_start=False,
            eval_strategy="epoch",   # used to be steps
            metric_for_best_model="accuracy",
            greater_is_better=True,
            save_total_limit=2,
            #load_best_model_at_end=True,
        )
        | training_kwargs
    )
    if config.wandb_project:
        training_kwargs["report_to"] = "wandb"
    else:
        training_kwargs["report_to"] = None

    training_args = TrainingArguments(**training_kwargs)

    if config.multitask:
        compute_metrics_fn = partial(
            compute_metrics_multitask, task_names=config.targets
        )
    else:
        compute_metrics_fn = compute_metrics

    trainer = Trainer(
        #model=model,
        args=training_args,
        data_collator=partial(collate_for_musicbert_fn, multitask=config.multitask),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        model_init=partial(model_init, model),
        compute_loss_func=partial(model.compute_loss),
        compute_metrics=compute_metrics_fn,
    )

    trainer.train()

#     trainer.hyperparameter_search(
#     direction="maximize", 
#     backend="ray", 
#     n_trials=2 # number of trials
# )
#     best_run = trainer_factory(training_args, model).hyperparameter_search(
#     direction="maximize",
#     backend="ray",
#     n_trials=2,  # Number of trials
#     hp_space=hp_space,
# )

    # del train_dataset, valid_dataset

    #test_dataset = get_dataset(config, "test")

    # results = trainer.evaluate(test_dataset, metric_key_prefix="test")
    # print(results)

    # print(f"Training complete. Output saved to {config.output_dir}")
