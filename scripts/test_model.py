"""
This script can be used to perform an HPO (Hyperparameter Optimization) on a MusicBERT model for a token classification task.
Config parameters are loaded from a YAML file, which should specify the following parameters:
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
- `checkpoint_path`: the path to the RNBERT checkpoint folder to evaluate.
- `targets`: a target or list of targets to finetune on. We expect each target to have a
  corresponding `.h5` file in the `data_dir` directory. For example, if `targets` is
  `["key", "chord_quality"]`, we expect to find `key.h5` and `chord_quality.h5` in the
  `data_dir/train` directory. To reproduce the results, use ["quality", "inversion", "key_pc_mode", "primary_alteration_primary_degree_secondary_alteration_secondary_degree"]
- `DEBUG`: if true, use a very small number of steps, and a small subset of data for quick testing.
- `baseline`: if true, the baseline mode will be evaluated. Otherwise, the Optuna trial specified by `optuna_name`, `optuna_storage`, and `trial_number` will be evaluated.
- `trial_number`: the number of the Optuna trial to evaluate. 
- `optuna_name`: the name of the Optuna study.
- `optuna_storage`: the path to the Optuna storage (e.g., a SQLite
    database) to use for storing the results of the trials.

"""
import sys
import os
from functools import partial
from transformers import Trainer, TrainingArguments
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from musicbert_hf.data import HDF5Dataset, collate_for_musicbert_fn
from musicbert_hf.metrics import compute_metrics, compute_metrics_multitask
from helpers import set_seed, load_baseline_params, get_dataset, create_hyperparams_dict, load_model, LimitedDataset
from config import load_config
from pathlib import Path

args =load_config("scripts/test_params.yaml")

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
compute_metrics_multitask, task_names=args.targets, entropy = True
) if args.multitask else compute_metrics

test_dataset = get_dataset(args, "test")
if args.DEBUG:
        test_dataset = LimitedDataset(test_dataset, limit=5)
test_trainer = Trainer(
    model=model,
    args=test_args,
    data_collator=partial(collate_for_musicbert_fn, multitask=args.multitask),
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics_fn,
)
print("Evaluating best model on test set...")
test_results = test_trainer.evaluate()
type = "baseline" if args.baseline else f"hpo_{args.optuna_name}_{args.optuna_storage}_trial_{args.trial_number}"
output_file_name = os.path.join("outputs", f"{args.file_prefix}{type}.txt")
path = Path(output_file_name)
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("w") as f:
    f.write(f"size of data: {len(test_dataset)}\n")
    for k, v in test_results.items():
        print(f"{k}: {v:.4f}")
        f.write(f"{k}: {v:.4f}\n")

