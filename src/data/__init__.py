"""Data module for R1.

This module provides dataset type definitions and utilities for working with
training and evaluation datasets in R1.

Exports:
    TrainDataset: Enum for training datasets like AIME, AMC, etc.
    TestDataset: Enum for testing/evaluation datasets.
    Dataset: Union type for either training or testing datasets.
"""

from src.data.dataset_types import TrainDataset, TestDataset, Dataset

__all__ = [
    "TrainDataset",
    "TestDataset",
    "Dataset",
]
