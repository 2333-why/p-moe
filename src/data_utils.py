"""Dataset helpers for Stage 2 WikiText perplexity evaluation."""

from __future__ import annotations

from typing import Any

from datasets import DownloadConfig, load_dataset


LOCAL_DATASET_SUGGESTION = """Dataset loading failed in local_files_only mode.
Suggestions:
1. Run once without --local_files_only to populate the Hugging Face datasets cache.
2. Check HF_HOME and HF_DATASETS_CACHE point to the expected cache location.
3. Use --dataset_name with a local dataset path that already exists on disk.
4. Verify the requested dataset_config_name and split are cached."""

DATASET_DOWNLOAD_SUGGESTION = """Dataset loading failed.
Suggestions:
1. Check network access and Hugging Face availability.
2. Check whether HF_ENDPOINT should be set.
3. Check free disk space for the datasets cache.
4. Verify dataset_name, dataset_config_name, and split."""


def load_wikitext_dataset(
    dataset_name: str = "wikitext",
    dataset_config_name: str | None = "wikitext-2-raw-v1",
    split: str = "test",
    local_files_only: bool = False,
):
    """
    Load a WikiText split, optionally using only locally cached files.

    This function wraps Hugging Face Datasets with a clear offline-mode error
    message so evaluation runs fail with actionable guidance instead of a long
    cache traceback.
    """
    download_config = DownloadConfig(local_files_only=local_files_only)
    try:
        if dataset_config_name:
            return load_dataset(
                dataset_name,
                dataset_config_name,
                split=split,
                download_config=download_config,
            )
        return load_dataset(dataset_name, split=split, download_config=download_config)
    except Exception as exc:
        if local_files_only:
            raise RuntimeError(LOCAL_DATASET_SUGGESTION) from exc
        raise RuntimeError(DATASET_DOWNLOAD_SUGGESTION) from exc


def extract_nonempty_texts(dataset: Any, text_column: str = "text") -> list[str]:
    """
    Return stripped, non-empty text rows from a dataset text column.
    """
    if text_column not in dataset.column_names:
        available = ", ".join(dataset.column_names)
        raise ValueError(
            f"Text column '{text_column}' was not found. Available columns: {available}"
        )

    texts: list[str] = []
    for value in dataset[text_column]:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            texts.append(text)
    if not texts:
        raise ValueError(f"No non-empty text found in column '{text_column}'.")
    return texts


def concatenate_texts(texts: list[str], separator: str = "\n\n") -> str:
    """
    Join dataset rows into one evaluation string for causal LM perplexity.
    """
    return separator.join(texts)
