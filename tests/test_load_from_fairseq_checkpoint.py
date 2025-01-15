import os

import pytest
import torch

from musicbert_hf.checkpoints import (
    load_musicbert_from_fairseq_checkpoint,
    load_musicbert_multitask_token_classifier_from_fairseq_checkpoint,
    load_musicbert_token_classifier_from_fairseq_checkpoint,
)
from musicbert_hf.utils import zip_longest_with_error

SMALL_CHECKPOINT = os.getenv("SMALL_CHECKPOINT")
BASE_CHECKPOINT = os.getenv("BASE_CHECKPOINT")

SMALL_TOKEN_CLS = os.getenv("SMALL_TOKEN_CLS")
SMALL_TOKEN_MULTI_CLS = os.getenv("SMALL_TOKEN_MULTI_CLS")

FAIRSEQ_OUTPUT_DIR = os.path.join(
    os.path.dirname((os.path.realpath(__file__))), "fairseq_outputs"
)

"""
To get fairseq outputs:
python /Users/malcolm/google_drive/python/data_science/musicbert_fork/misc_scripts/get_example_output.py checkpoint="/Volumes/Reicha/large_checkpoints/musicbert_provided_checkpoints/checkpoint_last_musicbert_small.pt" output_path=/Users/malcolm/google_drive/python/data_science/musicbert_hf/tests/resources/fairseq_small_320.pt 
python /Users/malcolm/google_drive/python/data_science/musicbert_fork/misc_scripts/get_example_output.py checkpoint="/Volumes/Reicha/large_checkpoints/musicbert_provided_checkpoints/checkpoint_last_musicbert_base.pt" output_path=/Users/malcolm/google_drive/python/data_science/musicbert_hf/tests/resources/fairseq_base_320.pt 

Token classifier
python /Users/malcolm/google_drive/python/data_science/musicbert_fork/misc_scripts/get_example_token_classifier_output.py output_path=~/google_drive/python/data_science/musicbert_hf/tests/resources/fairseq_token_small_320.pt
"""


def _do_load(
    arch,
    checkpoint_path,
    load_f,
    fairseq_output_path,
    sample_input,
    sample_labels,
    sample_attention_mask=None,
):
    model = load_f(checkpoint_path)
    model.eval()

    result = model(
        input_ids=sample_input,
        labels=sample_labels,
        attention_mask=sample_attention_mask,
    )
    hf_output = result.logits

    fairseq_output = torch.load(fairseq_output_path)

    if isinstance(hf_output, list):
        assert isinstance(fairseq_output, list)
        assert len(hf_output) == len(fairseq_output)
        for hf_out, fairseq_out in zip_longest_with_error(hf_output, fairseq_output):
            torch.isclose(hf_out, fairseq_out, atol=5).all()
    else:
        assert torch.isclose(hf_output, fairseq_output, atol=5).all()


def _mlm_input_and_labels():
    sample_input = torch.arange(320).reshape(1, -1)
    sample_labels = torch.tile(torch.arange(160), (2,)).reshape(1, -1)
    return sample_input, sample_labels


@pytest.mark.skipif(
    SMALL_CHECKPOINT is None, reason="SMALL_CHECKPOINT environment variable unset"
)
def test_load_small_checkpoint():
    fairseq_output_path = os.path.join(FAIRSEQ_OUTPUT_DIR, "fairseq_small_320.pt")
    _do_load(
        "small",
        SMALL_CHECKPOINT,
        load_musicbert_from_fairseq_checkpoint,
        fairseq_output_path,
        *_mlm_input_and_labels(),
    )


# @pytest.mark.skipif(
#     BASE_CHECKPOINT is None, reason="BASE_CHECKPOINT environment variable unset"
# )
# def test_load_base_checkpoint():
#     fairseq_output_path = os.path.join(RESOURCES_DIR, f"fairseq_base_320.pt")
#     _do_load(
#         "base",
#         BASE_CHECKPOINT,
#         load_musicbert_from_fairseq_checkpoint,
#         fairseq_output_path,
#         *_mlm_input_and_labels(),
#     )


def _token_class_input_and_labels():
    sample_input = torch.arange(320).reshape(1, -1)
    # sample_labels should have 1/8 the seq length of sample_input
    sample_labels = torch.tile(torch.arange(4), (10,)).reshape(1, -1)
    return sample_input, sample_labels


@pytest.mark.skipif(
    SMALL_TOKEN_CLS is None, reason="SMALL_TOKEN_CLS environment variable unset"
)
def test_load_small_token_classifier():
    fairseq_output_path = os.path.join(FAIRSEQ_OUTPUT_DIR, "fairseq_token_small_320.pt")
    _do_load(
        "small",
        SMALL_TOKEN_CLS,
        load_musicbert_token_classifier_from_fairseq_checkpoint,
        fairseq_output_path,
        *_token_class_input_and_labels(),
    )


def _token_multi_class_input_and_labels():
    sample_input = torch.arange(320).reshape(1, -1)
    # sample_labels should have 1/8 the seq length of sample_input
    sample_labels = torch.tile(torch.arange(2), (20,)).reshape(1, -1)
    sample_attention_mask = torch.zeros_like(sample_labels)
    sample_attention_mask[0, :18] = 1

    # 11 tasks in test model
    sample_labels = [sample_labels for _ in range(11)]
    return sample_input, sample_labels, sample_attention_mask


@pytest.mark.skipif(
    SMALL_TOKEN_MULTI_CLS is None,
    reason="SMALL_TOKEN_MULTI_CLS environment variable unset",
)
def test_load_small_multitask_token_classifier():
    fairseq_output_path = os.path.join(
        FAIRSEQ_OUTPUT_DIR, "fairseq_token_multi_small_320.pt"
    )
    _do_load(
        "small",
        SMALL_TOKEN_MULTI_CLS,
        load_musicbert_multitask_token_classifier_from_fairseq_checkpoint,
        fairseq_output_path,
        *_token_multi_class_input_and_labels(),
    )
