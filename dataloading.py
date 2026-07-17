"""Dataset loading helpers for HuWildBench style JSONL files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

PromptRecord = Dict[str, Any]


PROMPT_REPLACEMENTS = (
    ("a kérdés az: ", "\n\nA kérdés: "),
    ("a leírás: ", "\n\nA leírás: "),
    ("a válasz: ", ""),
)


@dataclass
class PromptSample:
    """Container holding a single prompt together with its source record."""

    index: int
    prompt: str
    raw: PromptRecord


def _read_jsonl(path: Path) -> Iterable[PromptRecord]:
    """Yield JSON objects from a JSONL file."""

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _tokenize_key_path(key_path: str) -> Sequence[str]:
    """Split dot/bracket notation into a list of keys or indices."""

    sanitized = key_path.replace("[", ".").replace("]", "").strip(".")
    return [part for part in sanitized.split(".") if part]


def _resolve_key_path(record: Any, key_path: str) -> Any:
    """Traverse dictionaries/lists following the supplied key path."""

    current: Any = record
    for part in _tokenize_key_path(key_path):
        if isinstance(current, list):
            try:
                idx = int(part)
            except ValueError as exc:  # pragma: no cover - defensive
                raise KeyError(
                    f"List index expected in key path segment '{part}'"
                ) from exc
            try:
                current = current[idx]
            except IndexError as exc:  # pragma: no cover - defensive
                raise KeyError(
                    f"List index {idx} is out of range for '{part}'"
                ) from exc
        elif isinstance(current, dict):
            if part not in current:
                raise KeyError(f"Key '{part}' not present while resolving '{key_path}'")
            current = current[part]
        else:  # pragma: no cover - sanity check
            raise KeyError(f"Cannot descend into non-container type at '{part}'")
    return current


def _normalize_prompt(text: str) -> str:
    """Apply the notebook's spacing tweaks to the prompt and trim it."""

    for old, new in PROMPT_REPLACEMENTS:
        text = text.replace(old, new)
    return text.strip()


def load_prompt_samples(
    dataset_path: Path | str,
    prompt_key: str,
    *,
    normalize: bool = True,
) -> List[PromptSample]:
    """Load all prompts from a JSONL dataset.

    Args:
        dataset_path: Path to the JSONL file.
        prompt_key: Dot/bracket path pointing at the prompt field (e.g. 'turn[0].content').
        normalize: When true, apply the spacing replacements detailed in test.ipynb.
    """

    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    samples: List[PromptSample] = []
    for idx, record in enumerate(_read_jsonl(path)):
        value = _resolve_key_path(record, prompt_key)
        if not isinstance(value, str):
            raise TypeError(
                f"Prompt at index {idx} resolved via '{prompt_key}' is not a string"
            )
        prompt = _normalize_prompt(value) if normalize else value
        samples.append(PromptSample(index=idx, prompt=prompt, raw=record))
    return samples


__all__ = ["PromptSample", "load_prompt_samples"]
