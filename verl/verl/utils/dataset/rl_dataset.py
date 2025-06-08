# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import uuid
from omegaconf import ListConfig
import os
from typing import Dict, List, Union, Optional
import copy
import random
import pandas as pd
import logging
import os
import re
from collections import defaultdict

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)


def collate_fn(data_list: list[dict]) -> dict:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


def process_image(image: dict, max_pixels: int = 2048 * 2048, min_pixels: int = 512 * 512):
    import math
    from io import BytesIO
    from PIL import Image

    if isinstance(image, dict):
        image = Image.open(BytesIO(image['bytes']))

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image


def generate_uuid5(s: str) -> str:
    """
    Generate a UUID5 hash from a string.
    """
    default_namespace_uuid = uuid.UUID('f9115e71-0a5d-42c3-9a6a-0a2f3d1e5c8b')
    return str(uuid.uuid5(default_namespace_uuid, s))


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, (List, ListConfig)):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            prompt_key = self.prompt_key
            self.dataframe = self.dataframe.filter(
                lambda doc: len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True)) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(self.dataframe)}")

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                for segment in re.split("(<image>|<video>)", content):
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        messages = self._build_messages(row_dict)
        model_inputs = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            multi_modal_data = {}

            images = None
            if self.image_key in row_dict:
                images = [process_image(image) for image in row_dict.pop(self.image_key)]
                multi_modal_data["image"] = images

            videos = None
            if self.video_key in row_dict:
                videos = [process_video(video) for video in row_dict.pop(self.video_key)]
                multi_modal_data["video"] = [video.numpy() for video in videos]

            model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data
            row_dict["multi_modal_inputs"] = dict(model_inputs)

            # second_per_grid_ts isn't used for training, just for mrope
            row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and self.processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)

        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()


class DomainWeightedRLHFDataset(Dataset):
    """
    A dataset that loads RLHF data from multiple domains with configurable sampling weights.
    """

    def __init__(
        self,
        domain_parquet_files: Dict[str, Union[str, List[str]]],
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin] = None,
        prompt_key="prompt",
        image_key="images",
        max_prompt_length=1024,
        filter_prompts=True,
        cache_dir="~/.cache/verl/rlhf",
        chat_template_func=None,
        return_raw_chat=False,
        truncation="error",
        filter_overlong_prompts=False,
    ):
        """
        Initialize a domain-weighted RLHF dataset.

        Args:
            domain_parquet_files: Dictionary mapping domain names to parquet file paths
            tokenizer: Tokenizer for processing text
            processor: Optional processor for multi-modal data
            prompt_key: Key for prompts in the parquet files
            image_key: Key for images in the parquet files
            max_prompt_length: Maximum length for prompts
            filter_prompts: Whether to filter prompts
            cache_dir: Directory for caching
            chat_template_func: Function for chat templates
            return_raw_chat: Whether to return raw chat
            truncation: How to handle truncation
            filter_overlong_prompts: Whether to filter overlong prompts
        """
        # Validate domains
        print(f"Domain parquet files: {domain_parquet_files}")

        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.filter_prompts = filter_prompts
        self.cache_dir = os.path.expanduser(cache_dir)
        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation
        self.filter_overlong_prompts = filter_overlong_prompts

        # Whether to store the dataset in state_dict()
        # Default not store
        self.serialize_dataset = False

        # Store original parquet files for resume
        self.original_domain_parquet_files = copy.deepcopy(domain_parquet_files)
        self.domain_parquet_files = copy.deepcopy(domain_parquet_files)

        # Create domain datasets
        self.domain_dataframes = {}
        self._download()
        self._read_files_and_tokenize()

        # Create mapping from flat index to (domain, domain_index)
        self.index_mapping = []
        for domain, dataframe in self.domain_dataframes.items():
            _ = len(self.index_mapping)
            for i in range(len(dataframe)):
                self.index_mapping.append((domain, i))

        self.total_size = len(self.index_mapping)

    def _download(self, use_origin_parquet=False):
        """Download parquet files to local cache."""
        from verl.utils.fs import copy_to_local

        parquet_files_dict = (
            self.domain_parquet_files
            if not use_origin_parquet
            else self.original_domain_parquet_files
        )

        for domain, parquet_files in parquet_files_dict.items():
            if not isinstance(parquet_files, (List, ListConfig)):
                parquet_files = [parquet_files]

            local_files = []
            for parquet_file in parquet_files:
                local_file = copy_to_local(src=parquet_file, cache_dir=self.cache_dir)
                local_files.append(local_file)

            self.domain_parquet_files[domain] = local_files

    def _read_files_and_tokenize(self):
        """Read parquet files and preprocess data for each domain."""
        for domain, parquet_files in self.domain_parquet_files.items():
            dataframes = []
            for parquet_file in parquet_files:
                dataframe = pd.read_parquet(parquet_file)
                dataframes.append(dataframe)

            domain_dataframe = pd.concat(dataframes)
            print(f"Domain {domain} dataset len: {len(domain_dataframe)}")

            # Filter out too long prompts
            if self.filter_overlong_prompts:
                tokenizer = self.tokenizer
                prompt_key = self.prompt_key
                domain_dataframe = domain_dataframe[
                    domain_dataframe.apply(
                        lambda doc: len(
                            tokenizer.apply_chat_template(
                                doc[prompt_key], add_generation_prompt=True
                            )
                        )
                        <= self.max_prompt_length,
                        axis=1,
                    )
                ]
                print(f"Domain {domain} filtered dataset len: {len(domain_dataframe)}")

            self.domain_dataframes[domain] = domain_dataframe

    def resume_dataset_state(self):
        """Resume dataset state from checkpoint."""
        self.serialize_dataset = (
            False if hasattr(self, "original_domain_parquet_files") else True
        )

        # Resume dataframe if not serialized in data.pt
        if not self.serialize_dataset:
            self._download(
                use_origin_parquet=True
            )  # Download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(
                r"Old dataloader checkpoint file is used, please train from scratch for better checkpoint performance"
            )

    def get_domain_size(self, domain: str) -> int:
        """Return the size of a specific domain."""
        if domain not in self.domain_dataframes:
            raise ValueError(f"Domain '{domain}' not found in dataset.")
        return len(self.domain_dataframes[domain])

    def __len__(self):
        """Return the total number of samples across all domains."""
        return self.total_size

    def __getitem__(self, idx: int):
        """Get a sample based on its flat index."""
        domain, domain_idx = self.index_mapping[idx]
        row_dict = self.domain_dataframes[domain].iloc[domain_idx].to_dict()

        # Add domain information to the row
        row_dict["domain"] = domain

        chat = row_dict.pop(self.prompt_key)

        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, tokenize=False
        )

        is_multi_modal = self.image_key in row_dict
        if is_multi_modal:  # Expand image token
            raw_prompt = prompt_with_chat_template.replace(
                "<image>", "<|vision_start|><|image_pad|><|vision_end|>"
            )
            row_dict["multi_modal_data"] = {
                "image": [
                    process_image(image) for image in row_dict.pop(self.image_key)
                ]
            }
            image_inputs = self.processor.image_processor(
                row_dict["multi_modal_data"]["image"], return_tensors="pt"
            )
            image_grid_thw = image_inputs["image_grid_thw"]
            row_dict["multi_modal_inputs"] = {
                key: val for key, val in image_inputs.items()
            }

            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while "<image>" in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        "<image>",
                        "<|vision_start|>"
                        + "<|placeholder|>"
                        * (image_grid_thw[index].prod() // merge_length)
                        + "<|vision_end|>",
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace(
                    "<|placeholder|>", self.processor.image_token
                )
        else:
            raw_prompt = prompt_with_chat_template

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_with_chat_template,
            tokenizer=self.tokenizer,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if is_multi_modal:
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_len)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]
        row_dict["raw_prompt_ids"] = self.tokenizer.encode(
            raw_prompt, add_special_tokens=False
        )

        # Encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = chat if isinstance(chat, list) else [chat]

        # Add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "domain_dataframes" in state:
                del state["domain_dataframes"]
            return state
        return self.__dict__.copy()


class DomainSampler:
    """A batch sampler that ensures each batch has the correct domain proportions."""

    def __init__(
        self,
        dataset: DomainWeightedRLHFDataset,
        batch_size: int,
        domain_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize a domain sampler.

        Args:
            dataset: The dataset to sample from
            batch_size: Size of each batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.domain_weights = domain_weights
        self.domains = list(domain_weights.keys())

        # Calculate how many samples from each domain should be in a batch
        self.domain_counts = {}
        remaining = batch_size
        for domain, weight in self.domain_weights.items():
            count = int(batch_size * weight)
            self.domain_counts[domain] = count
            remaining -= count

        # Distribute remaining samples to highest weight domains
        sorted_domains = sorted(
            self.domains, key=lambda d: self.domain_weights[d], reverse=True
        )
        for domain in sorted_domains:
            if remaining > 0:
                self.domain_counts[domain] += 1
                remaining -= 1
            else:
                break

        # Create domain indices mapping
        self.domain_indices = {domain: [] for domain in self.domains}
        for i, (domain, _) in enumerate(dataset.index_mapping):
            self.domain_indices[domain].append(i)

        # For each domain, create a shuffled list of indices
        self.domain_iterators = {domain: [] for domain in self.domains}
        for domain in self.domains:
            self._refill_domain_indices(domain)

    def domain_weights(self) -> Dict[str, float]:
        """Return the current domain weights."""
        return self.domain_weights

    def update_weights(self, weights: Optional[Dict[str, float]]=None) -> None:
        """Update the domain weights."""
        if weights is None:
            return
        self.domain_weights = weights
        self.domains = list(weights.keys())
        # Re-calculate how many samples from each domain should be in a batch
        self.domain_counts = {}
        remaining = self.batch_size
        for domain, weight in self.domain_weights.items():
            count = int(self.batch_size * weight)
            self.domain_counts[domain] = count
            remaining -= count
        # Distribute remaining samples to highest weight domains
        sorted_domains = sorted(
            self.domains, key=lambda d: self.domain_weights[d], reverse=True
        )
        for domain in sorted_domains:
            if remaining > 0:
                self.domain_counts[domain] += 1
                remaining -= 1
            else:
                break

    def _refill_domain_indices(self, domain: str) -> None:
        """Refill indices for a specific domain."""
        indices = self.domain_indices[domain].copy()
        random.shuffle(indices)
        self.domain_iterators[domain] = indices

    def __len__(self):
        """Return the total number of batches."""
        return len(self.dataset) // self.batch_size

    def __iter__(self):
        """Yield batches of indices that respect domain weights."""
        while True:
            batch_indices = []

            # For each domain, select the required number of indices
            for domain, count in self.domain_counts.items():
                # Skip if the domain has no data
                if not self.domain_indices[domain]:
                    continue

                # Ensure we have enough indices
                if len(self.domain_iterators[domain]) < count:
                    self._refill_domain_indices(domain)

                # Get at most count indices (in case domain has fewer samples than needed)
                to_take = min(count, len(self.domain_iterators[domain]))
                domain_batch_indices = self.domain_iterators[domain][:to_take]
                self.domain_iterators[domain] = self.domain_iterators[domain][to_take:]
                batch_indices.extend(domain_batch_indices)

            # Shuffle the batch indices
            random.shuffle(batch_indices)
            yield batch_indices
