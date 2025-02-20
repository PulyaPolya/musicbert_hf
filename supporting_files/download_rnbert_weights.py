import json
import os

from huggingface_hub import hf_hub_download

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "..", "rnbert_weights")

os.makedirs(WEIGHTS_DIR, exist_ok=True)

with open(os.path.join(SCRIPT_DIR, "rnbert_config.json"), "r") as f:
    config = json.load(f)


for key, value in config.items():
    if key.endswith("_path"):
        print(f"Downloading {value}")
        path = hf_hub_download(
            repo_id="msailor/rnbert_weights",
            filename=os.path.basename(value),
            local_dir=WEIGHTS_DIR,
        )
        print(f"Saved to {path}")
