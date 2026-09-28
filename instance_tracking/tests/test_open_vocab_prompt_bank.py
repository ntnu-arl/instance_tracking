"""Tests for offline open-vocabulary prompt-bank helpers."""

import importlib.util
import json
import pathlib

import numpy as np
import pytest

MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "instance_tracking_ros"
    / "instance_tracking_ros"
    / "prompt_bank.py"
)
SPEC = importlib.util.spec_from_file_location("instance_tracking_prompt_bank", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
prompt_bank = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prompt_bank)

PROMPT_BANK_SCHEMA_VERSION = prompt_bank.PROMPT_BANK_SCHEMA_VERSION
PROMPT_BANK_TYPE = prompt_bank.PROMPT_BANK_TYPE
load_prompt_file = prompt_bank.load_prompt_file
normalize_feature = prompt_bank.normalize_feature
save_prompt_bank = prompt_bank.save_prompt_bank


def test_normalize_feature_rejects_invalid_vectors():
    with pytest.raises(ValueError):
        normalize_feature([])

    with pytest.raises(ValueError):
        normalize_feature([0.0, 0.0, 0.0])

    with pytest.raises(ValueError):
        normalize_feature([1.0, np.nan])


def test_save_prompt_bank_writes_normalized_entries(tmp_path):
    output_path = tmp_path / "prompt_bank.json"
    saved_path = save_prompt_bank(
        output_path,
        encoder_id="openclip:test:model",
        prompts=["wall", "floor"],
        features=[[3.0, 4.0], [1.0, 0.0]],
    )

    assert saved_path == output_path.resolve()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == PROMPT_BANK_SCHEMA_VERSION
    assert payload["type"] == PROMPT_BANK_TYPE
    assert payload["normalized"] is True
    assert payload["encoder_id"] == "openclip:test:model"
    assert [entry["prompt"] for entry in payload["entries"]] == ["wall", "floor"]

    wall_embedding = np.asarray(payload["entries"][0]["embedding"], dtype=np.float32)
    assert np.isclose(np.linalg.norm(wall_embedding), 1.0)
    assert np.allclose(wall_embedding, np.array([0.6, 0.8], dtype=np.float32))


def test_load_prompt_file_skips_comments_and_blank_lines(tmp_path):
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text(
        "\n# comment\na photo of a wall\n\n a photo of a floor \n",
        encoding="utf-8",
    )

    assert load_prompt_file(prompt_file) == [
        "a photo of a wall",
        "a photo of a floor",
    ]
