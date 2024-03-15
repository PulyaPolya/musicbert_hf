import os
import pytest
import torch

from musicbert_hf.checkpoints import load_from_fairseq_checkpoint

SMALL_CHECKPOINT = os.getenv("SMALL_CHECKPOINT")
BASE_CHECKPOINT = os.getenv("BASE_CHECKPOINT")


"""
To get fairseq outputs:
python /Users/malcolm/google_drive/python/data_science/musicbert_fork/misc_scripts/get_example_output.py checkpoint="/Volumes/Reicha/large_checkpoints/musicbert_provided_checkpoints/checkpoint_last_musicbert_small.pt" output_path=/Users/malcolm/google_drive/python/data_science/musicbert_hf/tests/resources/fairseq_small_320.pt 
python /Users/malcolm/google_drive/python/data_science/musicbert_fork/misc_scripts/get_example_output.py checkpoint="/Volumes/Reicha/large_checkpoints/musicbert_provided_checkpoints/checkpoint_last_musicbert_base.pt" output_path=/Users/malcolm/google_drive/python/data_science/musicbert_hf/tests/resources/fairseq_base_320.pt 
"""


def _do_load(arch, checkpoint_path):

    model = load_from_fairseq_checkpoint(checkpoint_path)
    model.eval()

    sample_input = torch.arange(320).reshape(1, -1)
    hf_output = model(input_ids=sample_input).logits

    # TODO: (Malcolm 2024-03-15) update
    fairseq_output_path = (
        f"/Users/malcolm/output/musicbert/example_outputs/fairseq_{arch}_320.pt"
    )
    fairseq_output = torch.load(fairseq_output_path)

    assert torch.isclose(hf_output, fairseq_output, atol=5).all()


@pytest.mark.skipif(
    SMALL_CHECKPOINT is None, reason="SMALL_CHECKPOINT environment variable unset"
)
def test_load_small_checkpoint():
    _do_load("small", SMALL_CHECKPOINT)


@pytest.mark.skipif(
    BASE_CHECKPOINT is None, reason="BASE_CHECKPOINT environment variable unset"
)
def test_load_base_checkpoint():
    _do_load("base", BASE_CHECKPOINT)
