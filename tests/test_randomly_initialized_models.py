import numpy as np
import pytest
import torch

from musicbert_hf.musicbert_class import (
    BERT_PARAMS,
    MusicBertForMultiTaskTokenClassification,
    MusicBertForTokenClassification,
    MusicBertMultiTaskTokenClassConditioned,
    MusicBertMultiTaskTokenClassConditionedConfig,
    MusicBertMultiTaskTokenClassificationConfig,
    MusicBertTokenClassificationConfig,
)

TORCH_SEED = 42


@pytest.mark.skip(reason="Not finished implementing")
def test_musicbert_multitask_token_classifier_conditioned():
    input_vocab_size = 1237
    conditioning_vocab_size = 10
    sequence_length = 10
    batch_size = 2
    config = MusicBertMultiTaskTokenClassConditionedConfig(
        z_mlp_layers=2,
        z_vocab_size=conditioning_vocab_size,
        z_mlp_norm="yes",
        z_combine_procedure="concat",
        **BERT_PARAMS["tiny"],
    )
    model = MusicBertMultiTaskTokenClassConditioned(config)
    model.eval()

    input_ids = torch.randint(0, input_vocab_size, (batch_size, sequence_length * 8))
    conditioning_ids = torch.randint(
        0, conditioning_vocab_size, (batch_size, sequence_length)
    )

    output = model(conditioning_ids=conditioning_ids, input_ids=input_ids)
    breakpoint()


def test_token_classifier():
    input_vocab_size = 1237
    sequence_length = 20
    batch_size = 2
    num_labels = 10
    n_iters = 100
    config = MusicBertTokenClassificationConfig(
        num_labels=num_labels,
        **BERT_PARAMS["tiny"],
    )
    model = MusicBertForTokenClassification(config)
    model.eval()
    torch.manual_seed(TORCH_SEED)
    losses = []
    for _ in range(n_iters):
        input_ids = torch.randint(
            0, input_vocab_size, (batch_size, sequence_length * 8)
        )
        labels = torch.randint(0, num_labels, (batch_size, sequence_length))
        output = model(input_ids=input_ids, labels=labels)
        losses.append(output["loss"].item())
    actual_loss = sum(losses) / n_iters
    expected_loss = np.log(num_labels)
    print(f"actual_loss: {actual_loss}, expected_loss: {expected_loss}")
    assert actual_loss == pytest.approx(expected_loss, abs=1e-2)


def test_multitask_token_classifier():
    input_vocab_size = 1237
    sequence_length = 10
    batch_size = 2
    num_multi_labels = [2, 4, 8]
    n_iters = 100
    config = MusicBertMultiTaskTokenClassificationConfig(
        num_multi_labels=num_multi_labels,
        **BERT_PARAMS["tiny"],
    )
    model = MusicBertForMultiTaskTokenClassification(config)
    model.eval()
    torch.manual_seed(TORCH_SEED)
    losses = []
    for _ in range(n_iters):
        input_ids = torch.randint(
            0, input_vocab_size, (batch_size, sequence_length * 8)
        )
        labels = [
            torch.randint(0, num_multi_labels[i], (batch_size, sequence_length))
            for i in range(len(num_multi_labels))
        ]
        output = model(input_ids=input_ids, labels=labels)
        losses.append(output["loss"].item())

    actual_loss = sum(losses) / n_iters
    expected_loss = np.log(num_multi_labels).mean()
    print(f"actual_loss: {actual_loss}, expected_loss: {expected_loss}")
    assert actual_loss == pytest.approx(expected_loss, abs=1e-2)
