import torch
from einops import rearrange, repeat


def sync_slices(
    logits: torch.Tensor, slice_ids: torch.Tensor, return_per_slice: bool = False
):
    """
    Args:
        logits: A tensor of shape (..., sequence_length, vocab_size).
        slice_ids: A tensor of shape (..., sequence_length) containing monotonically
            increasing integers where the first integer is 0 and each subsequent
            value is incremented by 1. (We depend on getting the number of different
            slices as slice_ids[..., -1].max() + 1.)
        return_per_slice: if False, returns a tensor of the same shape as logits
            where each value has been replaced by the average of all values in the
            slice. If True, returns a tensor of shape (..., num_slices, vocab_size)
            with the average logits for each slice.

    Returns:
        A tensor of shape (..., sequence_length, vocab_size) if return_per_slice is
        False, or (..., num_slices, vocab_size) if return_per_slice is True. Note that
        num_slices is the maximum number of slices in any of the examples in the batch,
        so other examples will be padded with nans.
    """
    assert logits.ndim >= 2
    assert slice_ids.ndim >= 1
    assert logits.shape[-2] == slice_ids.shape[-1]

    num_slices = slice_ids[..., -1].max() + 1

    vocab_size = logits.shape[-1]

    sums = torch.zeros(
        (*logits.shape[:-2], num_slices, vocab_size),
        device=logits.device,
        dtype=logits.dtype,
    )
    counts = torch.zeros(
        (*logits.shape[:-2], num_slices),
        device=logits.device,
        dtype=logits.dtype,
    )

    sums.scatter_add_(-2, repeat(slice_ids, "... -> ... v", v=vocab_size), logits)
    counts.scatter_add_(
        -1,
        slice_ids,
        torch.ones(logits.shape[:-1], device=logits.device, dtype=logits.dtype),
    )

    averages = sums / rearrange(counts, "... -> ... 1")

    if return_per_slice:
        return averages
    else:
        return torch.gather(
            averages, -2, repeat(slice_ids, "... -> ... v", v=vocab_size)
        )
