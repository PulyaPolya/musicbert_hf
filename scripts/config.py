from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Literal
import yaml
import os
from pathlib import Path
import time

@dataclass
class Config:
    # data_dir should have train, valid, and test subdirectories
    data_dir: str
    
    checkpoint_path: str
    targets: str | list[str]
    output_dir_base: str
    conditioning: str | None = None
    log_dir: str = os.path.expanduser("~/tmp/musicbert_hf_logs")
    # We will always load from a checkpoint so we don't need to specify architecture
    # architecture: Literal["base", "tiny"] = "base"
    num_epochs: int = 0
    batch_size: int = 4
    learning_rate: float = 2.5e-4
    warmup_steps: int = 0
    max_steps: int = -1
    # if saving at hf
    hf_repository: str | None = None
    hf_token: str | None = None
    # debug mode uses few datapoints and fewer steps
    DEBUG: bool = True
    # run nas (removed)
    RUN_NAS : bool = False
    # number of otpuna trials to run
    num_trials: int = 1
    wandb_project: str | None = None
    wandb_name: str | None = None
    # If None, freeze no layers; if int, freeze all layers up to and including
    #   the specified layer; if sequence of ints, freeze the specified layers
    freeze_layers: int | Sequence[int] | None = None

    # In general, we want to leave job_id as None and set automatically, but for
    #   local testing we can set it manually
    job_id: str | None = None
    seed: int | None = 42
    name: Optional[str] = ""
    # path to optuna db
    optuna_storage: str | None = None
    # name of the optuna study
    optuna_name: str | None = None
    # we need to save the sampler as pickle 
    sampler_path: str | None = None
    # time limit for each trial in the optuna run (removed)
    time_limit : int | None = None
    # if we run the baseline model or the optuna trial
    baseline: str | None = None
    # if we run an optuna trial, we need its number
    trial_number: int | None = None
    # folder with metadata
    data_dir_for_metadata: str = None
    file_prefix : str = ""
    # debug
    msdebug: bool = False
    # for save predictions
    overwrite: bool = False
    compound_token_ratio: int = 8
    ignore_specials: int = 4
    task: str = "musicbert_multitask_sequence_tagging"
    head: str = "sequence_multitask_tagging_head"
    max_examples: Optional[int] = None
    dataset: str = "test"
    ref_dir: Optional[str] = None  # contains target_names.json and label[x]/dict.txt
   

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
        if self.name is None:
            self.name = self.job_id
        return os.path.join(self.output_dir_base, self.name)

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
    
def _load_yaml_(path: Path) -> dict:
    raw = path.read_text()
    if path.suffix.lower() in (".yml", ".yaml"):
        data = yaml.safe_load(raw) or {}
    return data

def load_config(path: str | os.PathLike) -> Config:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    data = _load_yaml_(p)
    # allow hyphenated keys in file
    data = {k.replace("-", "_"): v for k, v in data.items()}
    # validate keys
    valid = set(Config.__annotations__.keys())
    unknown = set(data) - valid
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    return Config(**data)
