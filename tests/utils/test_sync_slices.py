import numpy as np
import pytest
import torch
from einops import rearrange, repeat

from musicbert_hf.utils.sync_slices import sync_slices

AVALIABLE_DEVICES = ["cpu"]
if torch.cuda.is_available():
    AVALIABLE_DEVICES.append("cuda")
if torch.backends.mps.is_available():
    AVALIABLE_DEVICES.append("mps")


@pytest.mark.parametrize(
    "logits,slice_ids,expected,expected_per_slice",
    [
        (
            torch.linspace(0.0, 1.0, 10),
            torch.arange(10),
            torch.linspace(0.0, 1.0, 10),
            torch.linspace(0.0, 1.0, 10),
        ),
        (
            torch.arange(10).float() % 2,
            rearrange(repeat(torch.arange(5), "y -> x y", x=2), "x y -> (y x)"),
            torch.full((10,), 0.5, dtype=torch.float),
            torch.full((5,), 0.5, dtype=torch.float),
        ),
        (
            torch.tensor(
                [
                    [0.0, 2.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0, 2.0],
                    [0.0, 1.0, 1.0, 2.0, 1.0, 0.0, 0.0, 2.0, 2.0],
                ]
            ),
            torch.tensor(
                [
                    [0, 0, 1, 1, 1, 2, 2, 2, 2],
                    [0, 0, 0, 0, 1, 2, 2, 2, 2],
                ]
            ),
            torch.ones((2, 9)),
            torch.ones((2, 3)),
        ),
        (
            torch.tensor(
                [
                    [0.0, 2.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0, 2.0],
                    [0.0, 1.0, 1.0, 2.0, 1.0, 0.0, 2.0, -1.0, 3.0],
                ]
            ),
            torch.tensor(
                [
                    [0, 0, 1, 1, 1, 2, 2, 2, 2],
                    [0, 0, 0, 0, 1, 2, 2, 3, 3],
                ]
            ),
            torch.ones((2, 9)),
            torch.tensor([[1.0, 1.0, 1.0, np.nan], [1.0, 1.0, 1.0, 1.0]]),
        ),
    ],
)
@pytest.mark.parametrize("device", AVALIABLE_DEVICES)
def test_sync_slices(logits, slice_ids, expected, expected_per_slice, device):
    # For testing purposes, we use 1 or 2-d logits and repeat to get the vocab_size
    # dimension.
    logits = repeat(logits, "... -> ... v", v=8).to(device)
    expected = repeat(expected, "... -> ... v", v=8).to(device)
    expected_per_slice = repeat(expected_per_slice, "... -> ... v", v=8).to(device)
    slice_ids = slice_ids.to(device)

    result = sync_slices(logits, slice_ids)
    assert torch.allclose(result, expected)

    result_per_slice = sync_slices(logits, slice_ids, return_per_slice=True)
    assert torch.allclose(result_per_slice, expected_per_slice, equal_nan=True)
