"""Helpers for offline open-vocabulary prompt-bank files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PROMPT_BANK_SCHEMA_VERSION = 1
PROMPT_BANK_TYPE = "hydra_open_vocab_prompt_bank"


def normalize_feature(feature) -> np.ndarray:
    """Return one normalized float32 feature vector."""

    array = np.asarray(feature, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise ValueError("feature must be non-empty")
    if not np.all(np.isfinite(array)):
        raise ValueError("feature must contain only finite values")

    norm = float(np.linalg.norm(array))
    if norm <= 1.0e-9:
        raise ValueError("feature norm must be positive")

    return array / norm


def load_prompt_file(path: str | Path) -> list[str]:
    """Load one prompt per line, skipping blank lines and comments."""

    prompts: list[str] = []
    for raw_line in Path(path).expanduser().read_text(encoding="utf-8").splitlines():
        prompt = raw_line.strip()
        if not prompt or prompt.startswith("#"):
            continue
        prompts.append(prompt)

    return prompts


def build_prompt_bank_payload(encoder_id: str, prompts: list[str], features) -> dict:
    """Build the JSON payload written to disk."""

    if not encoder_id:
        raise ValueError("encoder_id must be non-empty")
    if len(prompts) != len(features):
        raise ValueError("prompt and feature counts must match")

    entries = []
    for prompt, feature in zip(prompts, features, strict=True):
        entries.append(
            {
                "prompt": str(prompt),
                "embedding": normalize_feature(feature).tolist(),
            }
        )

    return {
        "schema_version": PROMPT_BANK_SCHEMA_VERSION,
        "type": PROMPT_BANK_TYPE,
        "normalized": True,
        "encoder_id": encoder_id,
        "entries": entries,
    }


def save_prompt_bank(
    path: str | Path,
    *,
    encoder_id: str,
    prompts: list[str],
    features,
    overwrite: bool = False,
) -> Path:
    """Persist a prompt bank as JSON."""

    output_path = Path(path).expanduser().resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file '{output_path}'")

    payload = build_prompt_bank_payload(encoder_id, prompts, features)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path
