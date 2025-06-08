#!/usr/bin/env python

"""This script builds various datasets for reasoning tasks, 
including natural language reasoning, logical deduction, and mathematical reasoning.
"""

import os
import json
import uuid
import random
import argparse

from rich.console import Console
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer

random_seed = 42
console = Console()

think_token_begin = "<think>"
think_token_end = "</think>"
answer_token_begin = "<answer>"
answer_token_end = "</answer>"

natural_reasoning_instruction = "Please answer the question based on the facts you know. Fill in the answer between <answer> and </answer>. Provide your step by step reasoning process between <think> and </think>."
math_reasoning_instruction = "Please answer the math question in the following. Fill in the answer between <answer> and </answer>. Provide your step by step reasoning process between <think> and </think>."


def contains_digit_any(input_string: str) -> bool:
    """Checks if the string contains any digit using any()."""
    return any(char.isdigit() for char in input_string)


def load_model(model_name: str):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def sample_from_multiple_datasets(
    datasets,
    num_samples: int = 10000,
    by_weight: bool = False,
    dataset_name: str = "",
):
    data_available = []
    if by_weight:
        weights = [len(d) for d in datasets]
        weights = [w / sum(weights) for w in weights]
    else:
        weights = [1 / len(datasets)] * len(datasets)
    console.print(f"Sampling dataset by weights: {weights}")
    for i, dataset in enumerate(datasets):
        num_samples_i = int(num_samples * weights[i])
        sampled_dataset = random.sample(dataset, num_samples_i)
        console.print(
            f"Sampled {len(sampled_dataset)} data points from dataset {i} ({dataset_name})"
        )
        data_available.extend(sampled_dataset)
    random.shuffle(data_available)
    return data_available


def extract_keywords(model, tokenizer, text: str):
    inputs = tokenizer(text, return_tensors="pt")
    generation_config = {
        "do_sample": False,
        "use_cache": True,
        # "max_length": 128,
        "temperature": 0,
        # "top_k": 50,
        # "top_p": 1.0,
        # "repetition_penalty": 1.0,
        # "min_new_tokens": 4,
        "max_new_tokens": 128,
        # "early_stopping": False,
        # "num_beams": 1,
        # "num_beam_groups": 4,
        # "diversity_penalty": 1.0,
        "output_scores": False,
        "output_logits": False,
        "output_hidden_states": False,
        "output_attentions": False,
        "return_dict_in_generate": False,
        "num_return_sequences": 1,
    }
    outputs = model.generate(**inputs, **generation_config)
    generated_texts = [
        tokenizer.decode(output, skip_special_tokens=True) for output in outputs
    ]
    console.print(f"Generated text: {generated_texts}")
    return generated_texts


def build_dataset(
    dset_type: str, with_rule: bool = False, max_train_nums: int = 10_000
) -> None:
    """Builds a dataset based on the specified type and whether to include rules.

    Args:
        - dset_type (str): The type of dataset to build. Options include:
            - "prontoqa"
            - "proofwriter"
            - "natural_reasoning"
            - "clutrr"
            - "boxes"
            - "folio"
            - "ar_lsat"
            - "logic_nli"
            - "logical_deduction"
            - "aime25"
            - "logiqa"
        - with_rule (bool): Whether to include rules in the dataset.
        - max_train_nums (int): Maximum number of training samples to include.

    Returns:
        - None: The function saves the dataset to disk in JSON format.
    """
    if with_rule:
        suffix = "w-rule"
    else:
        suffix = "wo-rule"

    if dset_type == "prontoqa":
        with open("dataset/prontoqa/prontoqa.json", "r") as f:
            data = json.load(f)

        max_num_per_hop = 3200
        count_by_n_hop = dict()

        data_available = []
        for dp in data:
            id_ = str(uuid.uuid4())
            type_ = "prontoqa"
            instruction = dp["instruction"]
            rule = dp["rule"]
            keywords = ""
            input_text = dp["input"].replace("Facts:  ", "Facts: ")
            output_text = dp["output"].rsplit("</think>", 1)[-1].strip()
            ground_truth = (
                output_text.split("<answer>")[-1].split("</answer>")[0].strip()
            )
            n_hop = dp["n_hop"]

            new_dp = {
                "id": id_,
                "type": type_,
                "instruction": instruction,
                "rule": rule if with_rule else "",
                "input": input_text
                if with_rule
                else input_text.replace("Rules: ", "").replace(rule, "").strip(),
                "output": output_text,
                "ground_truth": ground_truth,
                "n_hop": n_hop,
            }
            if n_hop not in count_by_n_hop:
                count_by_n_hop[n_hop] = 0
            if count_by_n_hop[n_hop] < max_num_per_hop:
                data_available.append(new_dp)
                count_by_n_hop[n_hop] += 1
            else:
                continue

        random.shuffle(data_available)
        data_available = random.sample(
            data_available, min(len(data_available), 10000)
        )  # limit to 10k samples

        train_data, test_data = train_test_split(
            data_available, test_size=0.2, random_state=random_seed
        )
        train_data = random.sample(train_data, min(len(train_data), max_train_nums))
        test_data = (
            random.sample(test_data, min(len(test_data), 500))
            if len(test_data) > 500
            else test_data
        )

        os.makedirs("dataset/prontoqa", exist_ok=True)
        with open(f"dataset/prontoqa/train.{suffix}.json", "w") as f:
            json.dump(train_data, f, indent=4, ensure_ascii=False)
        with open(f"dataset/prontoqa/test.{suffix}.json", "w") as f:
            json.dump(test_data, f, indent=4, ensure_ascii=False)
    elif dset_type == "proofwriter":
        test_sets = [
            "dataset/proofwriter/norm-proofwriter-depth1.json",
            "dataset/proofwriter/norm-proofwriter-depth2.json",
            "dataset/proofwriter/norm-proofwriter-depth3.json",
            "dataset/proofwriter/norm-proofwriter-depth5.json",
        ]
        datasets = []
        dataset_name = "proofwriter"
        for test_set in test_sets:
            with open(test_set, "r") as f:
                data = json.load(f)
                datasets.append(data)

        data = sample_from_multiple_datasets(
            datasets,
            num_samples=10000,
            by_weight=True,
            dataset_name=dataset_name,
        )

        data_available = []
        for dp in data:
            id_ = str(uuid.uuid4())
            type_ = "proofwriter"
            instruction = dp["instruction"]
            rule = dp["rule"] if with_rule else ""
            keywords = ""
            input_text = (
                dp["input"]
                if with_rule
                else dp["input"].replace("Rules: ", "").replace(dp["rule"], "").strip()
            )
            output_text = dp["output"]
            ground_truth = (
                output_text.split("<answer>")[-1].split("</answer>")[0].strip()
            )
            n_hop = dp["n_hop"]

            new_dp = {
                "id": id_,
                "type": type_,
                "instruction": instruction,
                "rule": rule,
                "input": input_text,
                "output": output_text,
                "ground_truth": ground_truth,
                "n_hop": n_hop,
            }
            data_available.append(new_dp)

        random.shuffle(data_available)

        train_data, test_data = train_test_split(
            data_available, test_size=0.2, random_state=random_seed
        )
        train_data = random.sample(train_data, min(len(train_data), max_train_nums))
        test_data = (
            random.sample(test_data, min(len(test_data), 500))
            if len(test_data) > 500
            else test_data
        )

        os.makedirs("dataset/proofwriter", exist_ok=True)
        with open(f"dataset/proofwriter/train.{suffix}.json", "w") as f:
            json.dump(train_data, f, indent=4, ensure_ascii=False)
        with open(f"dataset/proofwriter/test.{suffix}.json", "w") as f:
            json.dump(test_data, f, indent=4, ensure_ascii=False)
    elif dset_type == "natural_reasoning":
        dataset = load_dataset("facebook/natural_reasoning")
        max_input_len = 1024
        max_output_len = 2
        max_sample_num = 10000
        data_available = []
        for dp in dataset["train"]:
            id_ = str(uuid.uuid4())
            type_ = "natural_reasoning"
            # TODO: add constraint for question type
            # subset_type = "boolq" # mcq, free text
            instruction = natural_reasoning_instruction
            rule = dp.get("rule", "") if with_rule else ""
            keywords = ""
            reference_answer = dp.get("reference_answer", "").strip()
            input_text = dp["question"].strip() if with_rule else dp["question"].strip()
            output_text = f"<answer>{reference_answer}</answer>"
            ground_truth = dp["reference_answer"].strip()

            # filter by constraints
            if len(data_available) >= max_sample_num:
                break
            if (
                reference_answer == ""
                or len(input_text.split()) > max_input_len
                or len(output_text.split()) > max_output_len
                or reference_answer.lower() in ["yes", "no", "true", "false"]
                or contains_digit_any(input_text)
                or contains_digit_any(output_text)
            ):
                continue

            new_dp = {
                "id": id_,
                "type": type_,
                "instruction": instruction,
                "rule": rule,
                "input": input_text,
                "output": output_text,
                "ground_truth": ground_truth,
            }
            data_available.append(new_dp)

        random.shuffle(data_available)
        train_data, test_data = train_test_split(
            data_available, test_size=0.2, random_state=random_seed
        )
        train_data = random.sample(train_data, min(len(train_data), max_train_nums))
        test_data = (
            random.sample(test_data, min(len(test_data), 500))
            if len(test_data) > 500
            else test_data
        )

        os.makedirs("dataset/natural_reasoning", exist_ok=True)
        with open(f"dataset/natural_reasoning/train.{suffix}.json", "w") as f:
            json.dump(train_data, f, indent=4, ensure_ascii=False)
        with open(f"dataset/natural_reasoning/test.{suffix}.json", "w") as f:
            json.dump(test_data, f, indent=4, ensure_ascii=False)
    elif dset_type == "clutrr":
        path_list = [
            "dataset/clutrr/rule_unshuffle/eval_data_2.json",
            "dataset/clutrr/rule_unshuffle/eval_data_34.json",
            "dataset/clutrr/rule_unshuffle/eval_data_56_all.json",
        ]
        datasets = []
        dataset_name = "clutrr"
        for path in path_list:
            with open(path, "r") as f:
                data = json.load(f)
                datasets.extend(data)
        data_available = []
        for dp in datasets:
            id_ = str(uuid.uuid4())
            type_ = "clutrr"
            instruction = natural_reasoning_instruction
            rule = " ".join(dp["rule"]) if with_rule else ""
            keywords = ""
            input_text = dp["query"].strip() if with_rule else dp["query"].strip()
            output_text = f"<answer>{dp['answer']}</answer>"
            ground_truth = dp["answer"].strip()

            new_dp = {
                "id": id_,
                "type": type_,
                "instruction": instruction,
                "rule": rule,
                "input": input_text,
                "output": output_text,
                "ground_truth": ground_truth,
            }
            data_available.append(new_dp)
        random.shuffle(data_available)
        train_data, test_data = train_test_split(
            data_available, test_size=0.2, random_state=random_seed
        )
        train_data = random.sample(train_data, min(len(train_data), max_train_nums))
        test_data = (
            random.sample(test_data, min(len(test_data), 500))
            if len(test_data) > 500
            else test_data
        )

        os.makedirs("dataset/clutrr", exist_ok=True)
        with open(f"dataset/clutrr/train.{suffix}.json", "w") as f:
            json.dump(train_data, f, indent=4, ensure_ascii=False)
        with open(f"dataset/clutrr/test.{suffix}.json", "w") as f:
            json.dump(test_data, f, indent=4, ensure_ascii=False)
    elif dset_type == "boxes":
        data_available = []
        with open("dataset/boxes/boxes.json", "r") as f:
            data = json.load(f)
            for dp in data:
                id_ = str(uuid.uuid4())
                type_ = "boxes"
                instruction = natural_reasoning_instruction
                rule = dp.get("rule", "") if with_rule else ""
                keywords = ""
                input_text = f'{dp["context"]} {dp["question"]}'
                output_text = f"<answer>{','.join(dp['answers'])}</answer>"
                ground_truth = ",".join(dp["answers"])

                new_dp = {
                    "id": id_,
                    "type": type_,
                    "instruction": instruction,
                    "rule": rule,
                    "input": input_text,
                    "output": output_text,
                    "ground_truth": ground_truth,
                }
                data_available.append(new_dp)
        random.shuffle(data_available)
        train_data, test_data = train_test_split(
            data_available, test_size=0.2, random_state=random_seed
        )
        train_data = random.sample(train_data, min(len(train_data), max_train_nums))
        test_data = (
            random.sample(test_data, min(len(test_data), 500))
            if len(test_data) > 500
            else test_data
        )

        os.makedirs("dataset/boxes", exist_ok=True)
        with open(f"dataset/boxes/train.{suffix}.json", "w") as f:
            json.dump(train_data, f, indent=4, ensure_ascii=False)
        with open(f"dataset/boxes/test.{suffix}.json", "w") as f:
            json.dump(test_data, f, indent=4, ensure_ascii=False)
    elif dset_type == "folio":
        with open("dataset/folio/folio.json", "r") as f:
            dataset = json.load(f)
        data_available = []
        for dp in dataset:
            id_ = str(uuid.uuid4())
            type_ = "folio"
            instruction = dp.get("instruction", natural_reasoning_instruction)
            rule = dp.get("rule", "") if with_rule else ""
            keywords = ""
            input_text = dp["input"].strip() if with_rule else dp["input"].strip()
            output_text = dp["output"].strip()
            ground_truth = (
                output_text.split("<answer>")[-1].split("</answer>")[0].strip()
            )

            new_dp = {
                "id": id_,
                "type": type_,
                "instruction": instruction,
                "rule": rule,
                "input": input_text,
                "output": output_text,
                "ground_truth": ground_truth,
            }
            data_available.append(new_dp)
        random.shuffle(data_available)
        train_data, test_data = train_test_split(
            data_available, test_size=0.2, random_state=random_seed
        )
        train_data = random.sample(train_data, min(len(train_data), max_train_nums))
        test_data = (
            random.sample(test_data, min(len(test_data), 500))
            if len(test_data) > 500
            else test_data
        )

        os.makedirs("dataset/folio", exist_ok=True)
        with open(f"dataset/folio/train.{suffix}.json", "w") as f:
            json.dump(train_data, f, indent=4, ensure_ascii=False)
        with open(f"dataset/folio/test.{suffix}.json", "w") as f:
            json.dump(test_data, f, indent=4, ensure_ascii=False)
    elif dset_type == "ar_lsat":
        with open("dataset/ar_lsat/ar_lsat.json", "r") as f:
            dataset = json.load(f)
        data_available = []
        for dp in dataset:
            id_ = str(uuid.uuid4())
            type_ = "ar_lsat"
            instruction = dp.get("instruction", natural_reasoning_instruction)
            rule = dp.get("rule", "") if with_rule else ""
            keywords = ""
            input_text = dp["input"].strip() if with_rule else dp["input"].strip()
            output_text = dp["output"].strip()
            ground_truth = (
                output_text.split("<answer>")[-1].split("</answer>")[0].strip()
            )

            new_dp = {
                "id": id_,
                "type": type_,
                "instruction": instruction,
                "rule": rule,
                "input": input_text,
                "output": output_text,
                "ground_truth": ground_truth,
            }
            data_available.append(new_dp)
        random.shuffle(data_available)
        train_data, test_data = train_test_split(
            data_available, test_size=0.2, random_state=random_seed
        )
        train_data = random.sample(train_data, min(len(train_data), max_train_nums))
        test_data = (
            random.sample(test_data, min(len(test_data), 500))
            if len(test_data) > 500
            else test_data
        )

        os.makedirs("dataset/ar_lsat", exist_ok=True)
        with open(f"dataset/ar_lsat/train.{suffix}.json", "w") as f:
            json.dump(train_data, f, indent=4, ensure_ascii=False)
        with open(f"dataset/ar_lsat/test.{suffix}.json", "w") as f:
            json.dump(test_data, f, indent=4, ensure_ascii=False)
    elif dset_type == "logic_nli":
        with open("dataset/logic_nli/logic_nli.json", "r") as f:
            dataset = json.load(f)
        data_available = []
        for dp in dataset:
            id_ = str(uuid.uuid4())
            type_ = "logic_nli"
            instruction = dp.get("instruction", natural_reasoning_instruction)
            rule = dp["input"].rsplit("\nFacts:", 1)[0].strip() if with_rule else ""
            keywords = ""
            input_text = (
                dp["input"].strip()
                if with_rule
                else "Facts: " + dp["input"].rsplit("\nFacts:", 1)[-1].strip()
            )
            output_text = dp["output"].strip()
            ground_truth = (
                output_text.split("<answer>")[-1].split("</answer>")[0].strip()
            )

            new_dp = {
                "id": id_,
                "type": type_,
                "instruction": instruction,
                "rule": rule,
                "input": input_text,
                "output": output_text,
                "ground_truth": ground_truth,
            }
            data_available.append(new_dp)
        random.shuffle(data_available)
        data_available = random.sample(
            data_available, min(len(data_available), 10000)
        )  # limit to 10k samples
        train_data, test_data = train_test_split(
            data_available, test_size=0.2, random_state=random_seed
        )
        train_data = random.sample(train_data, min(len(train_data), max_train_nums))
        test_data = (
            random.sample(test_data, min(len(test_data), 500))
            if len(test_data) > 500
            else test_data
        )

        os.makedirs("dataset/logic_nli", exist_ok=True)
        with open(f"dataset/logic_nli/train.{suffix}.json", "w") as f:
            json.dump(train_data, f, indent=4, ensure_ascii=False)
        with open(f"dataset/logic_nli/test.{suffix}.json", "w") as f:
            json.dump(test_data, f, indent=4, ensure_ascii=False)
    elif dset_type == "logical_deduction":
        with open("dataset/logical_deduction/logical_deduction.json", "r") as f:
            dataset = json.load(f)
        data_available = []
        for dp in dataset:
            id_ = str(uuid.uuid4())
            type_ = "logical_deduction"
            instruction = dp.get("instruction", natural_reasoning_instruction)
            rule = dp.get("rule", "") if with_rule else ""
            keywords = ""
            input_text = dp["input"].strip() if with_rule else dp["input"].strip()
            output_text = dp["output"].strip()
            ground_truth = (
                output_text.split("<answer>")[-1].split("</answer>")[0].strip()
            )

            new_dp = {
                "id": id_,
                "type": type_,
                "instruction": instruction,
                "rule": rule,
                "input": input_text,
                "output": output_text,
                "ground_truth": ground_truth,
            }
            data_available.append(new_dp)
        random.shuffle(data_available)
        train_data, test_data = train_test_split(
            data_available, test_size=0.2, random_state=random_seed
        )
        train_data = random.sample(train_data, min(len(train_data), max_train_nums))
        test_data = (
            random.sample(test_data, min(len(test_data), 500))
            if len(test_data) > 500
            else test_data
        )

        os.makedirs("dataset/logical_deduction", exist_ok=True)
        with open(f"dataset/logical_deduction/train.{suffix}.json", "w") as f:
            json.dump(train_data, f, indent=4, ensure_ascii=False)
        with open(f"dataset/logical_deduction/test.{suffix}.json", "w") as f:
            json.dump(test_data, f, indent=4, ensure_ascii=False)
    elif dset_type in ["bigbench", "bigbench_hard", "bigbench_extra_hard"]:
        data_available = []
        with open(f"dataset/{dset_type}/{dset_type}.json", "r") as f:
            data = json.load(f)
            for dp in data:
                id_ = str(uuid.uuid4())
                type_ = dset_type
                instruction = dp.get("instruction", natural_reasoning_instruction)
                rule = dp.get("rule", "") if with_rule else ""
                keywords = ""
                input_text = dp["input"].strip() if with_rule else dp["input"].strip()
                output_text = dp["output"].strip()
                ground_truth = (
                    output_text.split("<answer>")[-1].split("</answer>")[0].strip()
                )

                new_dp = {
                    "id": id_,
                    "type": type_,
                    "instruction": instruction,
                    "rule": rule,
                    "input": input_text,
                    "output": output_text,
                    "ground_truth": ground_truth,
                }
                data_available.append(new_dp)
        random.shuffle(data_available)

        test_data = data_available
        test_data = (
            random.sample(test_data, min(len(test_data), 500))
            if len(test_data) > 500
            else test_data
        )
        os.makedirs(f"dataset/{dset_type}", exist_ok=True)
        with open(f"dataset/{dset_type}/test.{suffix}.json", "w") as f:
            json.dump(test_data, f, indent=4, ensure_ascii=False)
    elif dset_type == "proverqa":
        data_available = []
        with open(f"dataset/proverqa/proverqa.json", "r") as f:
            data = json.load(f)
            for dp in data:
                id_ = str(uuid.uuid4())
                type_ = dp["source"].lower()
                if type_ == "proverqa-easy":
                    type_ = "ProverQA-Easy"
                elif type_ == "proverqa-medium":
                    type_ = "ProverQA-Medium"
                elif type_ == "proverqa-hard":
                    type_ = "ProverQA-Hard"
                else:
                    raise ValueError(f"Invalid source level '{type_}'")
                instruction = dp.get("instruction", natural_reasoning_instruction)
                rule = dp.get("rule", "") if with_rule else ""
                keywords = ""
                input_text = dp["input"].strip() if with_rule else dp["input"].strip()
                output_text = dp["output"].strip()
                ground_truth = (
                    output_text.split("<answer>")[-1].split("</answer>")[0].strip()
                )

                new_dp = {
                    "id": id_,
                    "type": type_,
                    "instruction": instruction,
                    "rule": rule,
                    "input": input_text,
                    "output": output_text,
                    "ground_truth": ground_truth,
                }
                data_available.append(new_dp)
        random.shuffle(data_available)
        test_data = data_available
        test_data = (
            random.sample(test_data, min(len(test_data), 500))
            if len(test_data) > 500
            else test_data
        )
        os.makedirs("dataset/proverqa", exist_ok=True)
        with open(f"dataset/proverqa/test.{suffix}.json", "w") as f:
            json.dump(test_data, f, indent=4, ensure_ascii=False)
    elif dset_type == "aime25":
        data_available = []
        data = load_dataset("yentinglin/aime_2025", split="train")
        for dp in data:
            id_ = str(uuid.uuid4())
            type_ = "aime25"
            instruction = dp.get("instruction", math_reasoning_instruction)
            rule = dp.get("rule", "") if with_rule else ""
            keywords = ""
            input_text = dp["problem"].strip()
            output_text = f'<answer>{dp["answer"].strip()}</answer>'
            ground_truth = dp["answer"].strip()

            new_dp = {
                "id": id_,
                "type": type_,
                "instruction": instruction,
                "rule": rule,
                "input": input_text,
                "output": output_text,
                "ground_truth": ground_truth,
            }
            data_available.append(new_dp)
        random.shuffle(data_available)
        test_data = data_available
        os.makedirs("dataset/aime25", exist_ok=True)
        with open(f"dataset/aime25/aime25.json", "w") as f:
            json.dump(test_data, f, indent=4, ensure_ascii=False)
    elif dset_type == "logiqa":
        data_available = []
        with open("dataset/logiqa/logiqa.json", "r") as f:
            data = json.load(f)
            for dp in data:
                id_ = str(uuid.uuid4())
                type_ = "logiqa"
                instruction = dp.get("instruction", natural_reasoning_instruction)
                rule = dp.get("rule", "") if with_rule else ""
                keywords = ""
                input_text = dp["input"].strip() if with_rule else dp["input"].strip()
                output_text = dp["output"].strip()
                ground_truth = (
                    output_text.split("<answer>")[-1].split("</answer>")[0].strip()
                )

                new_dp = {
                    "id": id_,
                    "type": type_,
                    "instruction": instruction,
                    "rule": rule,
                    "input": input_text,
                    "output": output_text,
                    "ground_truth": ground_truth,
                }
                data_available.append(new_dp)
        random.shuffle(data_available)
        train_data, test_data = train_test_split(
            data_available, test_size=0.2, random_state=random_seed
        )
        train_data = random.sample(train_data, min(len(train_data), max_train_nums))
        test_data = (
            random.sample(test_data, min(len(test_data), 500))
            if len(test_data) > 500
            else test_data
        )

        os.makedirs("dataset/logiqa", exist_ok=True)
        with open(f"dataset/logiqa/train.{suffix}.json", "w") as f:
            json.dump(train_data, f, indent=4, ensure_ascii=False)
        with open(f"dataset/logiqa/test.{suffix}.json", "w") as f:
            json.dump(test_data, f, indent=4, ensure_ascii=False)
    else:
        raise ValueError(f"Invalid dataset type '{dset_type}'")

    console.print(f'Dataset type: "{dset_type}" | With Rule: "{with_rule}"')
    console.print(f"Number of data points: {len(data_available)}")

    try:
        console.print(f"Number of train data points: {len(train_data)}")
    except NameError:
        console.print("No train data available")
    try:
        console.print(f"Number of test data points: {len(test_data)}")
    except NameError:
        console.print("No test data available")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[
            "prontoqa",
            "proofwriter",
            "natural_reasoning",
            "clutrr",
            "boxes",
            "folio",
            "ar_lsat",
            "logic_nli",
            "logical_deduction",
            "aime25",
            "logiqa",
        ],
        default=None,
        help="Dataset type to build",
    )
    args = parser.parse_args()
    if args.dataset is None:
        for dataset in [
            "prontoqa",
            "proofwriter",
            "natural_reasoning",
            "clutrr",
            "boxes",
            "folio",
            "ar_lsat",
            "logic_nli",
            "logical_deduction",
            "bigbench",
            "bigbench_hard",
            "bigbench_extra_hard",
            "proverqa",
            "aime25",
            "logiqa",
        ]:
            console.rule(characters=".")
            build_dataset(dataset, with_rule=False)
            build_dataset(dataset, with_rule=True)
            console.rule()
    else:
        console.rule(characters=".")
        build_dataset(args.dataset, with_rule=False)
        build_dataset(args.dataset, with_rule=True)
        console.rule()
