"""Dataset type definitions for Rule Reasoner.

This module defines enums for training and testing datasets used in R1,
as well as a union type for both dataset types.
"""

import enum
from typing import Union


class TrainDataset(enum.Enum):
    """Enum for training datasets.

    Contains identifiers for various math problem datasets used during training.
    """

    AIME24 = "AIME24"  # American Invitational Mathematics Examination
    AIME25 = "AIME25"  # American Invitational Mathematics Examination
    AMC = "AMC"  # American Mathematics Competition
    OMNI_MATH = "OMNI_MATH"  # Omni Math
    NUMINA_OLYMPIAD = "OLYMPIAD"  # Unique Olympiad problems from NUMINA
    MATH = "MATH"  # Dan Hendrycks Math Problems
    STILL = "STILL"  # STILL dataset
    LAWGPT = "LAWGPT"  # Law Reasoning
    PRONTOQA = "PRONTOQA"  # ProntoQA
    PROOFWRITER = "PROOFWRITER"  # ProofWriter
    CLUTRR = "CLUTRR"  # CLUTRR dataset
    BOXES = "BOXES"  # Boxed problems
    NATURAL_REASONING = "NATURAL_REASONING"  # Natural Reasoning
    FOLIO = "FOLIO"  # FOLIO dataset
    AR_LSAT = "AR_LSAT"  # AR-LSAT dataset
    LOGIC_NLI = "LOGIC_NLI"  # Logic NLI dataset
    LOGICAL_DEDUCTION = "LOGICAL_DEDUCTION"  # Logical Deduction dataset
    BIGBENCH = "BIGBENCH"  # BIG-Bench dataset
    BIGBENCH_HARD = "BIGBENCH_HARD"  # BIG-Bench Hard dataset
    BIGBENCH_EXTRA_HARD = "BIGBENCH_EXTRA_HARD"  # BIG-Bench Extra Hard dataset
    PROVERQA = "PROVERQA"  # ProverQA dataset
    LOGIQA = "LOGIQA"  # LogicQA dataset
    MIX = "MIX"  # Mixed dataset


class TestDataset(enum.Enum):
    """Enum for testing/evaluation datasets.

    Contains identifiers for datasets used to evaluate model performance.
    """

    AIME24 = "AIME24"  # American Invitational Mathematics Examination
    AIME25 = "AIME25"  # American Invitational Mathematics Examination
    AMC = "AMC"  # American Mathematics Competition
    MATH = "MATH"  # Math 500 problems
    MINERVA = "MINERVA"  # Minerva dataset
    OLYMPIAD_BENCH = "OLYMPIAD_BENCH"  # Olympiad benchmark problems
    LAWGPT = "LAWGPT"  # Law Reasoning
    PRONTOQA = "PRONTOQA"  # ProntoQA
    PROOFWRITER = "PROOFWRITER"  # ProofWriter
    CLUTRR = "CLUTRR"  # CLUTRR dataset
    BOXES = "BOXES"  # Boxed problems
    NATURAL_REASONING = "NATURAL_REASONING"  # Natural Reasoning
    FOLIO = "FOLIO"  # FOLIO dataset
    AR_LSAT = "AR_LSAT"  # AR-LSAT dataset
    LOGIC_NLI = "LOGIC_NLI"  # Logic NLI dataset
    LOGICAL_DEDUCTION = "LOGICAL_DEDUCTION"  # Logical Deduction dataset
    BIGBENCH = "BIGBENCH"  # BIG-Bench dataset
    BIGBENCH_HARD = "BIGBENCH_HARD"  # BIG-Bench Hard dataset
    BIGBENCH_EXTRA_HARD = "BIGBENCH_EXTRA_HARD"  # BIG-Bench Extra Hard dataset
    PROVERQA = "PROVERQA"  # ProverQA dataset
    LOGIQA = "LOGIQA"  # LogicQA dataset
    MIX = "MIX"  # Mixed dataset


"""Type alias for either training or testing dataset types."""
Dataset = Union[TrainDataset, TestDataset]
