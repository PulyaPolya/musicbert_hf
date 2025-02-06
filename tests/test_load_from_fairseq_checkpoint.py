import os

import numpy as np
import pytest
import torch

from musicbert_hf.checkpoints import (
    load_musicbert_from_fairseq_checkpoint,
    load_musicbert_multitask_token_classifier_from_fairseq_checkpoint,
    load_musicbert_multitask_token_classifier_with_conditioning_from_fairseq_checkpoint,
    load_musicbert_token_classifier_from_fairseq_checkpoint,
)

# from musicbert_hf.musicbert_class import MusicBertForTokenClassification
from musicbert_hf.utils import zip_longest_with_error

SMALL_CHECKPOINT = os.getenv("SMALL_CHECKPOINT")
BASE_CHECKPOINT = os.getenv("BASE_CHECKPOINT")

SMALL_TOKEN_CLS = os.getenv("SMALL_TOKEN_CLS")
SMALL_TOKEN_MULTI_CLS = os.getenv("SMALL_TOKEN_MULTI_CLS")
SMALL_TOKEN_MULTI_CLS_COND = os.getenv("SMALL_TOKEN_MULTI_CLS_COND")

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

ATOL = 1e-3
assert ATOL <= 1e-2

TORCH_SEED = 42


def _do_load(
    checkpoint_path,
    load_f,
    fairseq_output_path,
    sample_input,
    sample_labels,
    sample_attention_mask=None,
    sample_conditioning_input=None,
):
    model = load_f(checkpoint_path)
    model.eval()

    model_kwargs = dict(
        input_ids=sample_input,
        labels=sample_labels,
        attention_mask=sample_attention_mask,
    )
    if sample_conditioning_input is not None:
        model_kwargs["conditioning_ids"] = sample_conditioning_input

    result = model(**model_kwargs)

    hf_output = result.logits

    fairseq_output = torch.load(fairseq_output_path)

    if isinstance(hf_output, list):
        assert isinstance(fairseq_output, list)
        assert len(hf_output) == len(fairseq_output)
        for hf_out, fairseq_out in zip_longest_with_error(hf_output, fairseq_output):
            assert torch.isclose(hf_out, fairseq_out, atol=ATOL).all()
    else:
        assert torch.isclose(hf_output, fairseq_output, atol=ATOL).all()


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
        SMALL_CHECKPOINT,
        load_musicbert_from_fairseq_checkpoint,
        fairseq_output_path,
        *_mlm_input_and_labels(),
    )


@pytest.mark.slow
@pytest.mark.skipif(
    BASE_CHECKPOINT is None, reason="BASE_CHECKPOINT environment variable unset"
)
def test_load_base_checkpoint():
    fairseq_output_path = os.path.join(FAIRSEQ_OUTPUT_DIR, f"fairseq_base_320.pt")
    _do_load(
        BASE_CHECKPOINT,
        load_musicbert_from_fairseq_checkpoint,
        fairseq_output_path,
        *_mlm_input_and_labels(),
    )


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
        SMALL_TOKEN_CLS,
        load_musicbert_token_classifier_from_fairseq_checkpoint,
        fairseq_output_path,
        *_token_class_input_and_labels(),
    )


def _token_multi_class_input_and_labels():
    sample_input = torch.arange(320).reshape(1, -1)
    # sample_labels should have 1/8 the seq length of sample_input
    sample_labels = torch.tile(torch.arange(2), (20,)).reshape(1, -1)

    # (Malcolm 2025-01-21) It would be nice to get a fairseq version using an attention
    # mask but I'm not actually sure how to do that and probably not worth the effort.
    # In meantime we use an attention mask that is all ones
    sample_attention_mask = torch.ones_like(sample_labels)
    # sample_attention_mask = torch.zeros_like(sample_labels)
    # sample_attention_mask[0, :18] = 1

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
        SMALL_TOKEN_MULTI_CLS,
        load_musicbert_multitask_token_classifier_from_fairseq_checkpoint,
        fairseq_output_path,
        *_token_multi_class_input_and_labels(),
    )


def _token_multi_class_input_and_labels_and_conditioning():
    sample_input = torch.arange(320).reshape(1, -1)
    sample_conditioning = torch.arange(40).reshape(1, -1) % 16
    # sample_labels should have 1/8 the seq length of sample_input
    sample_labels = torch.tile(torch.arange(2), (20,)).reshape(1, -1)

    # (Malcolm 2025-01-21) It would be nice to get a fairseq version using an attention
    # mask but I'm not actually sure how to do that and probably not worth the effort.
    # In meantime we use an attention mask that is all ones
    sample_attention_mask = torch.ones_like(sample_labels)
    # sample_attention_mask = torch.zeros_like(sample_labels)
    # sample_attention_mask[0, :18] = 1

    # 11 tasks in test model
    sample_labels = [sample_labels for _ in range(11)]
    return sample_input, sample_labels, sample_attention_mask, sample_conditioning


@pytest.mark.skipif(
    SMALL_TOKEN_MULTI_CLS_COND is None,
    reason="SMALL_TOKEN_MULTI_CLS_COND environment variable unset",
)
def test_load_small_multitask_with_conditioning_token_classifier():
    fairseq_output_path = os.path.join(
        FAIRSEQ_OUTPUT_DIR, "fairseq_token_multi_cond_small_320.pt"
    )
    _do_load(
        SMALL_TOKEN_MULTI_CLS_COND,
        load_musicbert_multitask_token_classifier_with_conditioning_from_fairseq_checkpoint,
        fairseq_output_path,
        *_token_multi_class_input_and_labels_and_conditioning(),
    )


@pytest.mark.skipif(
    SMALL_CHECKPOINT is None, reason="SMALL_CHECKPOINT environment variable unset"
)
def test_load_small_token_classifier_expected_loss():
    num_labels = 10
    model = load_musicbert_token_classifier_from_fairseq_checkpoint(
        SMALL_CHECKPOINT,
        checkpoint_type="musicbert",
        num_labels=num_labels,
    )
    model.eval()

    seq_len = 10
    batch_size = 2
    n_iters = 100
    losses = []
    torch.manual_seed(TORCH_SEED)
    for _ in range(n_iters):
        input_ids = torch.randint(0, 1237, (batch_size, seq_len * 8))
        labels = torch.randint(0, num_labels, (batch_size, seq_len))

        output = model(input_ids, labels)
        loss = model.compute_loss(output, labels, num_items_in_batch=batch_size)
        losses.append(loss.item())
    actual_loss = sum(losses) / n_iters
    expected_loss = np.log(num_labels)

    assert actual_loss == pytest.approx(expected_loss, abs=1e-2)


@pytest.mark.skipif(
    SMALL_CHECKPOINT is None, reason="SMALL_CHECKPOINT environment variable unset"
)
def test_load_small_multitask_token_classifier_expected_loss():
    num_labels = [2, 5, 10]
    model = load_musicbert_multitask_token_classifier_from_fairseq_checkpoint(
        SMALL_CHECKPOINT,
        checkpoint_type="musicbert",
        num_labels=num_labels,
    )
    model.eval()

    seq_len = 10
    batch_size = 2
    n_iters = 100

    torch.manual_seed(TORCH_SEED)
    losses = []
    for _ in range(n_iters):
        input_ids = torch.randint(0, 1237, (batch_size, seq_len * 8))
        labels = [
            torch.randint(0, num_labels[i], (batch_size, seq_len))
            for i in range(len(num_labels))
        ]

        output = model(input_ids=input_ids, labels=labels)
        losses.append(output["loss"].item())
    actual_loss = sum(losses) / n_iters
    expected_loss = np.log(num_labels).mean()

    assert actual_loss == pytest.approx(expected_loss, abs=1e-2)
