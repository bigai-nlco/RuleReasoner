#!/usr/bin/env python

"""Script to prepare RLVR training and test datasets, saving them to parquet files."""

import os
import argparse
from typing import Dict, Optional, Any, Callable

import pandas as pd

from src.data.utils import load_dataset
from src.data.dataset_types import TrainDataset, TestDataset


def create_processor(split: str) -> Callable:
    """Create a function to process dataset examples.

    Args:
        split: Dataset split name ('train' or 'test')

    Returns:
        Function that standardizes individual dataset examples
    """

    def process_fn(
        example: Dict[str, Any], idx: int, data_source: str
    ) -> Optional[Dict[str, Any]]:
        if example.get("type", None):
            data_source = example.pop("type")
        question = (
            example.pop("input") if "input" in example else example.pop("problem")
        )
        instruction = example.pop("instruction") if "instruction" in example else ""
        question = f"{question} {instruction}" if instruction != "" else f"{question}"
        answer = example.pop("output") if "output" in example else example.pop("answer")
        ability = (
            "logical reasoning"
            if data_source not in ["amc", "aime24", "aime25"]
            else "mathematical reasoning"
        )

        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": question}],
            "ability": ability,
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {"split": split, "index": idx},
        }

    return process_fn


# Dataset configuration mapping
DATASET_CONFIG = {
    "prontoqa": ("ProntoQA", TrainDataset.PRONTOQA, TestDataset.PRONTOQA),
    "proofwriter": ("ProofWriter", TrainDataset.PROOFWRITER, TestDataset.PROOFWRITER),
    "clutrr": ("Clutrr", TrainDataset.CLUTRR, TestDataset.CLUTRR),
    "boxes": ("Boxes", TrainDataset.BOXES, TestDataset.BOXES),
    "natural_reasoning": (
        "Natural Reasoning",
        TrainDataset.NATURAL_REASONING,
        TestDataset.NATURAL_REASONING,
    ),
    "folio": ("Folio", TrainDataset.FOLIO, TestDataset.FOLIO),
    "ar_lsat": ("AR-LSAT", TrainDataset.AR_LSAT, TestDataset.AR_LSAT),
    "logic_nli": ("Logic NLI", TrainDataset.LOGIC_NLI, TestDataset.LOGIC_NLI),
    "logical_deduction": (
        "Logical Deduction",
        TrainDataset.LOGICAL_DEDUCTION,
        TestDataset.LOGICAL_DEDUCTION,
    ),
    "logiqa": ("LogiQA", TrainDataset.LOGIQA, TestDataset.LOGIQA),
    "bigbench": ("BigBench", TrainDataset.BIGBENCH, TestDataset.BIGBENCH),
    "bigbench_hard": (
        "BigBench Hard",
        TrainDataset.BIGBENCH_HARD,
        TestDataset.BIGBENCH_HARD,
    ),
    "bigbench_extra_hard": (
        "BigBench Extra Hard",
        TrainDataset.BIGBENCH_EXTRA_HARD,
        TestDataset.BIGBENCH_EXTRA_HARD,
    ),
    "proverqa": ("ProverQA", TrainDataset.PROVERQA, TestDataset.PROVERQA),
    "aime24": ("AIME24", TrainDataset.AIME24, TestDataset.AIME24),
    "aime25": ("AIME25", TrainDataset.AIME25, TestDataset.AIME25),
    "amc": ("AMC", TrainDataset.AMC, TestDataset.AMC),
    "mix": ("Mix", TrainDataset.MIX, TestDataset.MIX),
}


def process_dataset(dataset_type: str, local_dir: str, use_rule: bool):
    """Process and save a specific dataset type.

    Args:
        dataset_type: Type of dataset to process
        local_dir: Directory to save processed data
        use_rule: Whether to use rule-based processing
    """
    data_source, train_dataset_enum, test_dataset_enum = DATASET_CONFIG[dataset_type]

    # Load datasets
    train_dataset = load_dataset(
        train_dataset_enum, task_type=dataset_type, use_rule=use_rule
    )
    test_dataset = load_dataset(
        test_dataset_enum, task_type=dataset_type, use_rule=use_rule
    )

    # Process training data
    train_processor = create_processor("train")
    train_data = [
        train_processor(example, idx, data_source)
        for idx, example in enumerate(train_dataset)
    ]

    # Process test data
    test_processor = create_processor("test")
    test_data = [
        test_processor(example, idx, data_source)
        for idx, example in enumerate(test_dataset)
    ]

    # Save processed data
    dataset_name = test_dataset_enum.value.lower()
    suffix = "_rule" if use_rule else ""

    # Save train data
    train_df = pd.DataFrame(train_data)
    train_df.to_parquet(os.path.join(local_dir, f"train{suffix}.parquet"))
    print(f"{dataset_type} train data size: {len(train_data)}")

    # Save test data
    if data_source != "Mix":
        test_df = pd.DataFrame(test_data)
        test_df.to_parquet(os.path.join(local_dir, f"{dataset_name}{suffix}.parquet"))
        print(f"{dataset_name} test data size: {len(test_data)}")

    print("=====================================")


def main():
    """Main function to process datasets based on command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process datasets for R1-Zero training"
    )
    parser.add_argument(
        "--dataset",
        default="",
        help="Dataset to process (e.g., prontoqa, proofwriter, etc.)",
    )
    parser.add_argument(
        "--use_rule", action="store_true", help="Use rule for reasoning"
    )
    args = parser.parse_args()

    # Create output directory
    dataset_dir = f"dataset/{args.dataset}" if args.dataset else "dataset"
    os.makedirs(dataset_dir, exist_ok=True)

    if args.dataset:
        # Process a single specified dataset
        process_dataset(args.dataset, dataset_dir, args.use_rule)
    else:
        # Process all available datasets
        for dataset_type in DATASET_CONFIG:
            dataset_specific_dir = f"dataset/{dataset_type}"
            os.makedirs(dataset_specific_dir, exist_ok=True)
            process_dataset(dataset_type, dataset_specific_dir, args.use_rule)


if __name__ == "__main__":
    main()
