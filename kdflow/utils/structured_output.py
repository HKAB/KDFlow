import os
import re
from typing import Iterable, Optional


ROLLOUT_REGEX_FILE_ENV = "KDFLOW_ROLLOUT_REGEX_FILE"


def load_rollout_regex(path: Optional[str] = None) -> Optional[str]:
    """Load the optional SGLang rollout regex from disk."""
    path = path if path is not None else os.environ.get(ROLLOUT_REGEX_FILE_ENV)
    if not path:
        return None

    try:
        with open(path, encoding="utf-8") as file:
            regex = file.read()
    except OSError as error:
        raise RuntimeError(f"Cannot read rollout regex file {path!r}") from error

    if not regex.strip():
        raise ValueError(f"Rollout regex file is empty: {path!r}")
    return regex.rstrip("\r\n")


def validate_rollout_regex_outputs(
    outputs: Iterable[str], regex: Optional[str]
) -> None:
    """Fail before training if SGLang returns text outside the configured regex."""
    if regex is None:
        return
    try:
        pattern = re.compile(regex)
    except re.error as error:
        raise ValueError(
            "The configured rollout regex cannot be validated by Python's regex "
            "engine. Refusing to train without structured-output validation."
        ) from error

    for index, output in enumerate(outputs):
        if pattern.fullmatch(output) is None:
            preview = output[:200].replace("\n", "\\n")
            raise RuntimeError(
                "SGLang returned an output that violates the configured rollout "
                f"regex (sample={index}, preview={preview!r})."
            )
