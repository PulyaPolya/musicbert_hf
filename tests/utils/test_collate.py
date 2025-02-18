import pytest
import torch
from einops import repeat

from musicbert_hf.utils.collate import collate_logits, collate_slice_ids

AVALIABLE_DEVICES = ["cpu"]
if torch.cuda.is_available():
    AVALIABLE_DEVICES.append("cuda")
if torch.backends.mps.is_available():
    AVALIABLE_DEVICES.append("mps")


@pytest.mark.parametrize(
    "logits,overlap_size,attention_masks,expected",
    [
        ([torch.zeros(10), torch.ones(10)], 10, None, torch.linspace(0.0, 1.0, 10)),
        (
            [torch.zeros(10), torch.ones(20), torch.zeros(10)],
            10,
            None,
            torch.cat(
                [torch.linspace(0.0, 1.0, 10), torch.linspace(1.0, 0.0, 10)], dim=-1
            ),
        ),
        (
            [
                repeat(torch.arange(10).float(), "b -> b c", c=20),
                repeat(torch.arange(10, 20).float(), "b -> b c", c=20),
            ],
            5,
            None,
            torch.cat(
                [
                    repeat(torch.arange(10).float(), "b -> b c", c=15),
                    repeat(torch.linspace(0.0, 10.0, 5), "c -> b c", b=10)
                    + torch.arange(10).float()[:, None],
                    repeat(torch.arange(10, 20).float(), "b -> b c", c=15),
                ],
                dim=-1,
            ),
        ),
        (
            [torch.zeros(10), torch.ones(5)],
            10,
            None,
            torch.cat([torch.zeros(5), torch.linspace(0.0, 1.0, 5)]),
        ),
    ],
)
@pytest.mark.parametrize("device", AVALIABLE_DEVICES)
def test_collate_logits(logits, overlap_size, attention_masks, expected, device):
    # For testing purposes, we use 2-d logits (batch_size, sequence_length) and
    # just repeat to get the vocab_size dimension.
    logits = [repeat(item, "... ->  ... v", v=20).to(device) for item in logits]
    expected = repeat(expected, "... ->  ... v", v=20).to(device)
    attention_masks = (
        attention_masks.to(device) if attention_masks is not None else None
    )

    result = collate_logits(logits, overlap_size, attention_masks)
    assert torch.allclose(result, expected)


@pytest.mark.parametrize(
    "slice_ids,overlap_size,expected",
    [
        ([torch.arange(10), torch.arange(10, 20)], 0, torch.arange(20)),
        ([torch.arange(10), torch.arange(0, 20)], 50, torch.arange(20)),
        ([torch.arange(5), torch.arange(10), torch.arange(15)], 10, torch.arange(15)),
        (
            [torch.arange(20), torch.arange(10, 30), torch.arange(20, 30)],
            10,
            torch.arange(30),
        ),
    ],
)
def test_collate_slice_ids(slice_ids, overlap_size, expected):
    result = collate_slice_ids(slice_ids, overlap_size, check_overlap=True)
    assert torch.all(result == expected)
