# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from vllm.transformers_utils.repo_utils import (
    any_pattern_in_repo_files,
    get_hf_file_to_dict,
    is_mistral_model_repo,
    list_filtered_repo_files,
)


@pytest.mark.parametrize(
    "allow_patterns,expected_relative_files",
    [
        (
            ["*.json", "correct*.txt"],
            ["json_file.json", "subfolder/correct.txt", "correct_2.txt"],
        ),
    ],
)
def test_list_filtered_repo_files(
    allow_patterns: list[str], expected_relative_files: list[str]
):
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Prep folder and files
        path_tmp_dir = Path(tmp_dir)
        subfolder = path_tmp_dir / "subfolder"
        subfolder.mkdir()
        (path_tmp_dir / "json_file.json").touch()
        (path_tmp_dir / "correct_2.txt").touch()
        (path_tmp_dir / "uncorrect.txt").touch()
        (path_tmp_dir / "uncorrect.jpeg").touch()
        (subfolder / "correct.txt").touch()
        (subfolder / "uncorrect_sub.txt").touch()

        def _glob_path() -> list[str]:
            return [
                str(file.relative_to(path_tmp_dir))
                for file in path_tmp_dir.glob("**/*")
                if file.is_file()
            ]

        # Patch list_repo_files called by fn
        with patch(
            "vllm.transformers_utils.repo_utils.list_repo_files",
            MagicMock(return_value=_glob_path()),
        ) as mock_list_repo_files:
            out_files = sorted(
                list_filtered_repo_files(
                    tmp_dir, allow_patterns, "revision", "model", "token"
                )
            )
        assert out_files == sorted(expected_relative_files)
        assert mock_list_repo_files.call_count == 1
        assert mock_list_repo_files.call_args_list[0] == call(
            repo_id=tmp_dir,
            revision="revision",
            repo_type="model",
            token="token",
        )


@pytest.mark.parametrize(
    ("allow_patterns", "expected_bool"),
    [
        (["*.json", "correct*.txt"], True),
        (
            ["*.jpeg"],
            True,
        ),
        (
            ["not_found.jpeg"],
            False,
        ),
    ],
)
def test_one_filtered_repo_files(allow_patterns: list[str], expected_bool: bool):
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Prep folder and files
        path_tmp_dir = Path(tmp_dir)
        subfolder = path_tmp_dir / "subfolder"
        subfolder.mkdir()
        (path_tmp_dir / "uncorrect.jpeg").touch()
        (subfolder / "correct.txt").touch()

        def _glob_path() -> list[str]:
            return [
                str(file.relative_to(path_tmp_dir))
                for file in path_tmp_dir.glob("**/*")
                if file.is_file()
            ]

        # Patch list_repo_files called by fn
        with patch(
            "vllm.transformers_utils.repo_utils.list_repo_files",
            MagicMock(return_value=_glob_path()),
        ) as mock_list_repo_files:
            assert (
                any_pattern_in_repo_files(
                    tmp_dir, allow_patterns, "revision", "model", "token"
                )
            ) is expected_bool
        assert mock_list_repo_files.call_count == 1
        assert mock_list_repo_files.call_args_list[0] == call(
            repo_id=tmp_dir,
            revision="revision",
            repo_type="model",
            token="token",
        )


@pytest.mark.parametrize(
    ("files", "expected_bool"),
    [
        (["consolidated.safetensors", "incorrect.txt"], True),
        (["consolidated-1.safetensors", "incorrect.txt"], True),
        (
            ["consolidated-1.json"],
            False,
        ),
    ],
)
def test_is_mistral_model_repo(files: list[str], expected_bool: bool):
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Prep folder and files
        path_tmp_dir = Path(tmp_dir)
        for file in files:
            (path_tmp_dir / file).touch()

        def _glob_path() -> list[str]:
            return [
                str(file.relative_to(path_tmp_dir))
                for file in path_tmp_dir.glob("**/*")
                if file.is_file()
            ]

        # Patch list_repo_files called by fn
        with patch(
            "vllm.transformers_utils.repo_utils.list_repo_files",
            MagicMock(return_value=_glob_path()),
        ) as mock_list_repo_files:
            assert (
                is_mistral_model_repo(tmp_dir, "revision", "model", "token")
                is expected_bool
            )
        assert mock_list_repo_files.call_count == 1
        assert mock_list_repo_files.call_args_list[0] == call(
            repo_id=tmp_dir,
            revision="revision",
            repo_type="model",
            token="token",
        )


# ---------------------------------------------------------------------------
# Tests for HFValidationError handling in get_hf_file_to_dict
# ---------------------------------------------------------------------------

def test_get_hf_file_to_dict_returns_none_on_hf_validation_error():
    """get_hf_file_to_dict must return None (not raise) when hf_hub_download
    raises HFValidationError.

    This happens when `model` is a local path that does not look like a
    valid HuggingFace repo-id — e.g. a .mar decoder path such as
    "decoder/audio/NCZSCosybigvganDecoder.mar" used by the
    HyperCLOVAX-SEED-Omni-8B audio decoder stage.
    """
    from huggingface_hub.errors import HFValidationError

    with patch(
        "vllm.transformers_utils.repo_utils.hf_hub_download",
        side_effect=HFValidationError("not a valid repo id"),
    ):
        result = get_hf_file_to_dict("config.json", "decoder/audio/NCZSCosybigvganDecoder.mar")
    assert result is None, (
        "get_hf_file_to_dict must return None for local .mar paths "
        "that trigger HFValidationError, not propagate the exception"
    )


def test_get_hf_file_to_dict_reads_local_json_file():
    """get_hf_file_to_dict reads a local JSON file without hitting HF Hub."""
    import json
    payload = {"model_type": "hcx_omni", "vocab_size": 200064}

    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = Path(tmp_dir) / "config.json"
        config_path.write_text(json.dumps(payload))

        result = get_hf_file_to_dict("config.json", tmp_dir)

    assert result == payload


def test_get_hf_file_to_dict_returns_none_for_missing_local_file():
    """get_hf_file_to_dict returns None when local file does not exist."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        result = get_hf_file_to_dict("nonexistent.json", tmp_dir)
    assert result is None
