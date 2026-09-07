import pytest

from kdflow.utils.token_suppression import load_kd_suppress_token_ids


def test_loads_unique_suppressed_token_ids_from_environment(monkeypatch):
    monkeypatch.setenv("KDFLOW_KD_SUPPRESS_TOKEN_IDS", "98, 100,101,98")

    assert load_kd_suppress_token_ids() == (98, 100, 101)


def test_rejects_invalid_suppressed_token_ids():
    with pytest.raises(ValueError, match="comma-separated integers"):
        load_kd_suppress_token_ids("98,nope")

    with pytest.raises(ValueError, match="negative"):
        load_kd_suppress_token_ids("98,-1")
