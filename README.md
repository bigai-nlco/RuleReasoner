<div id="top">

<!-- HEADER STYLE: CLASSIC -->
<div align="center">

# RuleReasoner: Reinforced Rule-based Reasoning via Domain-aware Dynamic Sampling

<p align="center">
    | <a href="https://github.com/bigai-nlco/RuleReasoner"><b>Code</b></a> |
    <a href="https://huggingface.co/RuleReasoner"><b>Model</b></a> |
    <a href="https://huggingface.co/datasets/RuleReasoner/rule-reasoning"><b>Data</b></a> |
</p>

<!-- BADGES -->

[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://openreview.net/forum?id=MQV4TJyqnb)    [![DeepWiki](https://img.shields.io/badge/-blue?style=for-the-badge&logo=snowflake&logoColor=white&labelColor=black)](https://deepwiki.com/bigai-nlco/RuleReasoner)    [![Github](https://img.shields.io/badge/Code-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/bigai-nlco/RuleReasoner)    [![Hugging Face Models](https://img.shields.io/badge/RuleReasoner-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/RuleReasoner)

</div>
<br>

## News
- **[2026.01.26]** Our paper has been accepted by **ICLR 2026 ✨**.
- **[2025.01.26]** Our project introduction has been featured on [DeepWiki](https://deepwiki.com/bigai-nlco/RuleReasoner).
- **[2025.06.11]** ![Our post](https://x.com/mayayoko98/status/1932743802059698308) on X (aka. Twitter) has received many likes.
- **[2025.06.11]** We were featured as HuggingFace Daily Paper ![#3](https://huggingface.co/papers/2506.08672).

## Overview
Reinforced Rule-based Reasoning (RuleReasoner) is a method for enabling small reasoning models (SRMs) to effectively learn rule-based reasoning. Unlike large models that require complex training, RuleReasoner uses a curated collection of tasks and a domain-aware dynamic sampling approach, adjusting training based on historical performance. This technique allows SRMs to outperform frontier Large Reasoning Models (LRMs) by +4.1% on in-distribution tasks and +10.4% on out-of-distribution tasks, while also being more computationally efficient.

## Table of Contents

- [News](#news)
- [Overview](#overview)
- [Quick Start](#quick-start)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Training](#training)
    - [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## Quick Start

### Prerequisites

Running `RuleReasoner` requires the dependencies listed in `requirements.txt`.

### Installation

Build RuleReasoner from the source and install dependencies:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/bigai-nlco/RuleReasoner.git
    ```

2. **Navigate to the project directory:**

    ```sh
    cd RuleReasoner
    ```

3. **Install the dependencies:**

	```bash
	pip install -r requirements.txt
	pip install -e ./verl
	pip install -e .
	```

### Training

Run the training with:

```bash
./scripts/train/train_mix.sh
```

### Evaluation

Run the evaluation with:

```bash
./scripts/eval/eval_model.sh \
    --model $MODEL_PATH \
    --datasets $DATASET_PATH \
    --output-dir $OUTPUT_DIR
```

## Project Structure

```bash
└── RuleReasoner
    ├── LICENSE
    ├── README.md
    ├── requirements.txt
    ├── scripts
    │   ├── build_dataset.py
    │   ├── data
    │   ├── eval
    │   └── train
    ├── setup.py
    ├── src
    │   ├── __init__.py
    │   ├── data
    │   ├── globals.py
    │   ├── system_prompts.py
    │   └── utils.py
    └── verl
	└── ...
```

## Contributing

- **Discussions**: Share your insights, provide feedback, or ask questions in the [discussions section](https://github.com/bigai-nlco/RuleReasoner/discussions).
- **Issues**: Submit bugs found or log feature requests for the `RuleReasoner` project in the [issues section](https://github.com/bigai-nlco/RuleReasoner/issues).
- **Pull Requests**: Review open PRs, and submit your own PRs.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your own account.
2. **Clone Locally**: Clone the forked repository to your local machine.
   ```bash
   git clone https://github.com/<YOUR-USERNAME>/RuleReasoner.git
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```bash
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```bash
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to your fork**: Push the changes to your forked repository.
   ```bash
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch.
</details>

## License

Rulereasoner is distributed under the terms of the [MIT License](https://choosealicense.com/licenses/mit/).

## Citation
```latex
@article{liu2025rulereasoner,
      title={RuleReasoner: Reinforced Rule-based Reasoning via Domain-aware Dynamic Sampling},
      author={Yang Liu and Jiaqi Li and Zilong Zheng},
      year={2025},
      eprint={2506.08672},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.08672},
}
```
