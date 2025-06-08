"""Utility functions for loading and processing datasets.

This module provides functions for loading datasets from JSON files and handling
dataset-related operations in the R1 project.
"""

import json
import os
from typing import Any, Dict, List

from src.data import Dataset, TrainDataset


def load_dataset(
    dataset: Dataset, task_type: str, use_rule: bool = False
) -> List[Dict[str, Any]]:
    """Load a dataset from a JSON file.

    Loads and parses a JSON dataset file based on the provided dataset enum.
    The file path is constructed based on whether it's a training or testing dataset.

    Args:
        dataset: A Dataset enum value specifying which dataset to load.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the dataset records.
            Each dictionary represents one example in the dataset.

    Raises:
        ValueError: If the dataset file cannot be found, contains invalid JSON,
            or encounters other file access errors.

    Example:
        >>> load_dataset(TrainDataset.AIME)
        [{'problem': 'Find x...', 'solution': '42', ...}, ...]
    """

    if not use_rule:
        try:
            file_path = (
                f"dataset/{task_type}/train.wo-rule.json"
                if isinstance(dataset, TrainDataset)
                else f"dataset/{task_type}/test.wo-rule.json"
            )
            if not os.path.exists(file_path):
                raise ValueError(f"Dataset file not found: {file_path}")
        except ValueError:
            file_path = f"dataset/{task_type}/{task_type}.json"
    else:
        try:
            file_path = (
                f"dataset/{task_type}/train.w-rule.json"
                if isinstance(dataset, TrainDataset)
                else f"dataset/{task_type}/test.w-rule.json"
            )
            if not os.path.exists(file_path):
                file_path = (
                    f"dataset/{task_type}/train.w-rule.0.64.json"
                )
                if not os.path.exists(file_path):
                    raise ValueError(f"Dataset file not found: {file_path}")
        except ValueError:
            file_path = f"dataset/{task_type}/{task_type}.json"

    if not os.path.exists(file_path):
        raise ValueError(f"Dataset file not found: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in {file_path}")
    except Exception as exc:  # pylint: disable=broad-except
        raise ValueError(f"Error loading dataset: {exc}") from exc


if __name__ == "__main__":
    load_dataset(TrainDataset.NUMINA_OLYMPIAD)
