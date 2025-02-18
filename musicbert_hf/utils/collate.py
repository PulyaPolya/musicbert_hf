from typing import Sequence

import torch

TARGET_PAD = -100


def collate_logits(
    logits: Sequence[torch.Tensor],
    overlap_size: int,
    attention_masks: Sequence[torch.Tensor] | None = None,
    trim_start: int | bool = False,
    trim_end: int | bool = False,
) -> torch.Tensor:
    """
    Args:
        logits: A list of logits tensors of shape (sequence_length_i, vocab_size) or
            (batch_size, sequence_length_i, vocab_size).
        overlap_size: The size of the overlap between the segments.
        attention_masks: A list of attention masks of shape (sequence_length) or
            (batch_size, sequence_length).

    Returns:
        A tensor of the collated logits of shape (..., new_sequence_length,
            vocab_size).
    """
    if not logits:
        raise ValueError("No logits provided")

    trim_start = int(trim_start)
    trim_end = int(trim_end)

    out_logits = logits[0]

    assert out_logits.ndim >= 2
    if attention_masks is not None:
        out_logits = out_logits[attention_masks[0].bool()]

    out_logits = out_logits[trim_start:]
    if trim_end:
        out_logits = out_logits[:-trim_end]

    # What would be a lot more efficient would be to save a list of logit segments
    # and concatenate only once at the end. However, we can only do that if we can
    # guarantee that only adjacent segments overlap (e.g., if there can be a region of
    # overlap between segments, 1, 2, and 3, then we would require more complicated
    # logic.)

    for i in range(1, len(logits)):
        logits_i = logits[i]

        assert logits_i.ndim >= 2
        if attention_masks is not None:
            logits_i = logits_i[attention_masks[i].bool()]

        logits_i = logits_i[trim_start:]
        if trim_end:
            logits_i = logits_i[:-trim_end]

        this_overlap_size = min(overlap_size, out_logits.shape[-2], logits_i.shape[-2])

        left_overlap = out_logits[..., -this_overlap_size:, :]
        right_overlap = logits_i[..., :this_overlap_size, :]
        left_overlap = (
            left_overlap
            * torch.linspace(
                1.0,
                0.0,
                this_overlap_size,
                dtype=left_overlap.dtype,
                device=left_overlap.device,
            )[:, None]
        )
        right_overlap = (
            right_overlap
            * torch.linspace(
                0.0,
                1.0,
                this_overlap_size,
                dtype=right_overlap.dtype,
                device=right_overlap.device,
            )[:, None]
        )
        overlap = left_overlap + right_overlap
        out_logits = torch.cat(
            [
                out_logits[..., :-this_overlap_size, :],
                overlap,
                logits_i[..., this_overlap_size:, :],
            ],
            dim=-2,
        )

    return out_logits


def collate_slice_ids(
    slice_ids: Sequence[torch.Tensor], overlap_size: int, check_overlap: bool = False
) -> torch.Tensor:
    if not slice_ids:
        raise ValueError("No slice ids provided")

    out_slice_ids = slice_ids[0]
    out_slice_ids = out_slice_ids[out_slice_ids != TARGET_PAD]
    assert overlap_size >= 0

    for i in range(1, len(slice_ids)):
        slice_ids_i = slice_ids[i]
        # Remove pads
        slice_ids_i = slice_ids_i[slice_ids_i != TARGET_PAD]

        this_overlap_size = min(
            overlap_size, out_slice_ids.shape[-1], slice_ids_i.shape[-1]
        )
        if check_overlap and this_overlap_size > 0:
            left_overlap = out_slice_ids[..., -this_overlap_size:]
            right_overlap = slice_ids_i[..., :this_overlap_size]
            assert torch.all(left_overlap == right_overlap)
        out_slice_ids = torch.cat(
            [out_slice_ids, slice_ids_i[..., this_overlap_size:]], dim=-1
        )

    return out_slice_ids
