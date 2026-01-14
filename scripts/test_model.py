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
prefix = ""
if args.DEBUG:
        test_dataset = LimitedDataset(test_dataset, limit=5)
        prefix = "DEBUG_"
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
output_file_name = os.path.join("outputs", f"{prefix}{type}.txt")
path = Path(output_file_name)
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("w") as f:
    f.write(f"size of data: {len(test_dataset)}\n")
    for k, v in test_results.items():
        print(f"{k}: {v:.4f}")
        f.write(f"{k}: {v:.4f}\n")

