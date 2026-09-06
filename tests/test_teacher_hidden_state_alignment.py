import numpy as np
import pytest

from kdflow.backend.sglang.hidden_state_alignment import select_loss_hidden_states


def test_selects_completion_positions_from_full_hidden_states():
    hidden = np.arange(8 * 3).reshape(8, 3)
    mask = np.array([False] * 4 + [True] * 3 + [False])
    selected, info = select_loss_hidden_states(hidden, mask, sample_index=0)
    np.testing.assert_array_equal(selected, hidden[4:7])
    assert info["inferred_cached_prefix_length"] == 0


def test_selects_completion_positions_from_uncached_suffix():
    hidden = np.arange(4 * 3).reshape(4, 3)
    mask = np.array([False] * 4 + [True] * 3 + [False])
    selected, info = select_loss_hidden_states(hidden, mask, sample_index=2)
    np.testing.assert_array_equal(selected, hidden[:3])
    assert info["inferred_cached_prefix_length"] == 4


def test_rejects_cache_hit_that_hides_required_position():
    hidden = np.arange(3 * 2).reshape(3, 2)
    mask = np.array([False] * 4 + [True] * 3 + [False])
    with pytest.raises(ValueError, match="Refusing to pad or fabricate"):
        select_loss_hidden_states(hidden, mask, sample_index=1)


def test_rejects_non_sequence_hidden_states():
    with pytest.raises(ValueError, match="rank 2"):
        select_loss_hidden_states(np.ones(3), np.ones(3), sample_index=0)
