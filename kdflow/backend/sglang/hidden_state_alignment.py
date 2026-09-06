"""Alignment contract for SGLang prefill hidden states.

RadixAttention may return only the newly computed suffix. KDFlow only transfers
positions selected by the teacher loss mask, so cached-prefix states are not
needed as long as no selected position falls inside the cached span.
"""

from __future__ import annotations

import numpy as np


def loss_mask_start_positions(loss_masks) -> list[int]:
    """Return the first KD position per sample, or -1 for an empty mask.

    SGLang uses ``logprob_start_len`` as an upper bound for radix-prefix
    matching. Capping the match at the first required position guarantees that
    every hidden state selected by the KD loss is recomputed and returned.
    """
    starts = []
    for sample_index, loss_mask in enumerate(loss_masks):
        mask = np.asarray(loss_mask, dtype=bool)
        if mask.ndim != 1:
            raise ValueError(
                f"Loss mask must be rank 1 for sample {sample_index}, "
                f"got shape {mask.shape}."
            )
        required_positions = np.flatnonzero(mask)
        starts.append(int(required_positions[0]) if required_positions.size else -1)
    return starts


def select_loss_hidden_states(
    hidden_states: np.ndarray,
    loss_mask: np.ndarray,
    *,
    sample_index: int,
) -> tuple[np.ndarray, dict]:
    """Select masked states from a full sequence or a returned suffix.

    SGLang's prefill result is position-preserving: when radix reuse shortens
    prefill, the returned states represent a contiguous suffix ending at the
    final input token. We reject the result if any requested KD position lies
    before that suffix.
    """
    hidden_states = np.asarray(hidden_states)
    loss_mask = np.asarray(loss_mask, dtype=bool)
    if hidden_states.ndim != 2:
        raise ValueError(
            f"Teacher hidden states must be rank 2 for sample {sample_index}, "
            f"got shape {hidden_states.shape}."
        )

    full_length = int(loss_mask.shape[0])
    returned_length = int(hidden_states.shape[0])
    if returned_length > full_length:
        raise ValueError(
            f"Teacher returned more prefill positions than input positions for "
            f"sample {sample_index}: returned={returned_length}, input={full_length}."
        )

    returned_start = full_length - returned_length
    required_positions = np.flatnonzero(loss_mask)
    if required_positions.size and int(required_positions[0]) < returned_start:
        raise ValueError(
            "SGLang radix-cache hidden states do not cover all KD positions for "
            f"sample {sample_index}: input_length={full_length}, "
            f"returned_length={returned_length}, returned_start={returned_start}, "
            f"first_required_position={int(required_positions[0])}. Refusing to "
            "pad or fabricate cached-prefix hidden states."
        )

    selected = hidden_states[loss_mask[returned_start:]]
    expected = int(loss_mask.sum())
    if selected.shape[0] != expected:
        raise RuntimeError(
            f"Teacher hidden-state alignment produced {selected.shape[0]} states "
            f"for {expected} KD positions in sample {sample_index}."
        )
    return selected, {
        "input_length": full_length,
        "returned_length": returned_length,
        "returned_start": returned_start,
        "inferred_cached_prefix_length": returned_start,
        "selected_length": expected,
    }
