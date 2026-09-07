import os
from typing import Optional, Tuple


KD_SUPPRESS_TOKEN_IDS_ENV = "KDFLOW_KD_SUPPRESS_TOKEN_IDS"


def load_kd_suppress_token_ids(value: Optional[str] = None) -> Tuple[int, ...]:
    """Parse the optional comma-separated token IDs excluded from KD logits."""
    value = value if value is not None else os.environ.get(KD_SUPPRESS_TOKEN_IDS_ENV)
    if not value or not value.strip():
        return ()

    try:
        token_ids = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",")))
    except ValueError as error:
        raise ValueError(
            f"{KD_SUPPRESS_TOKEN_IDS_ENV} must contain comma-separated integers"
        ) from error
    if any(token_id < 0 for token_id in token_ids):
        raise ValueError(f"{KD_SUPPRESS_TOKEN_IDS_ENV} cannot contain negative IDs")
    return token_ids
