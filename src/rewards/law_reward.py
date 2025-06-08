"""
This module contains the RewardLawFn class, which evaluates law answers
and assigns rewards based on their correctness, rule violation, format, length.
It utilizes a language model to validate answers when necessary.
"""

import json
from typing import Any, Dict, List, Union

from evaluate import load
from rich.console import Console

from src.globals import THOUGHT_DELIMITER_START, THOUGHT_DELIMITER_END, OAI_RM_MODEL
from src.rewards import RewardConfig, RewardFn, RewardInput, RewardOutput, RewardType
from src.rewards.math_utils.utils import (
    validate_answer_format,
    parse_generation,
    grade_generation_length,
    grade_language_monotony,
    grade_language_repetition,
    grade_law_solution_by_outcome,
    grade_law_solution_by_process,
)
from src.system_prompts import ORM_PROMPT
from src.utils import call_oai_rm_llm

console = Console()

ORM_USER_TEMPLATE = """
Problem: {problem}
Answer 1: {answer_1}
Answer 2: {answer_2}
"""


def compute_meteor(
    preds: List[str],
    labels: List[str],
) -> float:
    """compute the METEOR score of the model predictions.

    Args:
        preds (List[str]): predicted texts
        labels (List[str]): ground-truth texts

    Returns:
        float: METEOR score
    """
    assert len(preds) == len(labels), "preds and labels must have the same length"

    meteor_scorer = load("meteor")
    meteor_score = meteor_scorer.compute(predictions=preds, references=labels).get(
        "meteor", 0.0
    )

    return round(meteor_score, 3)


class RewardLawFn(RewardFn):
    """
    Reward function for evaluating law reasoning answers.

    This class implements the __call__ method to process the input and determine
    the reward based on the correctness of the provided answer compared to the ground truth.
    """

    def _aggregate_rewards(self, rewards: Dict[str, float]) -> float:
        """Aggregate multiple rewards into a single reward value.

        Args:
            rewards (Dict[str, float]): Dictionary of rewards to aggregate.

        Returns:
            float: Aggregated reward value.
        """
        rewards_sum = 0.0
        for name, value in rewards.items():
            if "reward" in name:
                rewards_sum += value

        return rewards_sum

    def __call__(self, input: RewardInput) -> RewardOutput:
        assert (
            input.problem_type == RewardType.LAW
        ), "Invalid problem type: expected 'LAW, but got '{}'".format(
            input.problem_type
        )

        problem = input.problem  # instruction + input
        model_response = (
            input.model_response.rsplit("<|im_end|>", 1)[-1]
            .split("assistant", 1)[-1]
            .rsplit("<|endoftext|>", 1)[0]
            .strip()
        )  # output
        ground_truth = input.ground_truth  # answer
        predicted_answer = parse_generation([model_response])[0]
        reference_answer = parse_generation([ground_truth])[0]
        # meteor_score = compute_meteor([model_response], [ground_truth])

        # print(f"Problem: {problem}")
        # print(f"Model Response: {model_response}")
        eval_result = {
            "input": problem,
            "output": model_response,
            "reference": ground_truth,
            "predicted_answer": predicted_answer,
            "reference_answer": reference_answer,
            # "meteor": meteor_score,
            "format_rewards": 0,
            "length_rewards": 0,
            "unk_error_rewards": 0,
            "repetition_rewards": 0,
            "language_monotony_rewards": 0,
            "correctness_rewards": 0,
            "soft_exact_match": 0,
            "hard_exact_match": 0,
        }

        # 2nd parse trial
        model_answer = model_response

        # Step 0. Check if the model response is valid
        if model_response is None or validate_answer_format(model_response) is False:
            if "<think>" in model_response and "</think>" in model_response:
                eval_result["format_rewards"] = 0.25
            elif "<think>" in model_response or "</think>" in model_response:
                eval_result["format_rewards"] = 0.125
            elif "<answer>" in model_response and "</answer>" in model_response:
                eval_result["format_rewards"] = 0.25
            elif "<answer>" in model_response or "</answer>" in model_response:
                eval_result["format_rewards"] = 0.125
            else:
                eval_result["format_rewards"] = self.config.format_error_reward

        # Check if the model response is too long or too short
        eval_result["length_rewards"] = grade_generation_length(model_response)

        # Step 1. check the language monotony / repetition reward of the model response
        language_monotony_score = grade_language_monotony(model_response, language="zh")
        if not language_monotony_score:
            eval_result["language_monotony_rewards"] = (
                self.config.language_monotony_reward
            )

        language_repetition_score = grade_language_repetition(
            model_response, language="zh", ngram=1, tau=1.0, steepness=4.0
        )
        if language_repetition_score < -0.5:
            eval_result["repetition_rewards"] = language_repetition_score

        # Step 1. extract the answer from the model response
        # considered answer types: [刑期], [金额]
        """
        if (
            THOUGHT_DELIMITER_START in model_response
            and THOUGHT_DELIMITER_END in model_response
        ):
            model_solution = model_response.split(THOUGHT_DELIMITER_END)[1]
        else:
            return RewardOutput(
                reward=self.config.format_error_reward, is_correct=False
            ), eval_result
        """

        # 1st parse trial
        if (
            predicted_answer is None
            or predicted_answer == ""
            or reference_answer is None
            or reference_answer == ""
            or model_response.count("<think>") != 1
            or model_response.count("</think>") != 1
            or model_response.count("<answer>") != 1
            or model_response.count("</answer>") != 1
            or (
                model_response.count("[刑期]") != 1
                and model_response.count("[金额]") != 1
            )
        ):
            eval_result["format_rewards"] = self.config.format_error_reward

        # Step 2. Process the ground truth(s)
        ground_truth = (
            input.ground_truth.get("answer", None)
            if isinstance(input.ground_truth, dict)
            else input.ground_truth
        )
        keywords = (
            input.ground_truth.get("keywords", None)
            if isinstance(input.ground_truth, dict)
            else None
        )
        if ground_truth is None:
            eval_result["unk_error_rewards"] = self.config.unk_error_reward

        # Step 3. Convert single answer to list for uniform processing
        if isinstance(ground_truth, (str, float, int)):
            ground_truths = [ground_truth]

        processed_ground_truths = []
        for truth in ground_truths:
            truth = str(truth)
            processed_ground_truths.append(truth)

        if not processed_ground_truths:
            eval_result["unk_error_rewards"] = self.config.unk_error_reward

        # Step 4. Check if the answer is correct against all possible correct answers
        # (Add float range for soft match: Penalty: +/- 3 month, Money: +/- 1000 RMB)
        for ground_truth in processed_ground_truths:
            is_soft_correct = grade_law_solution_by_outcome(
                model_answer,
                ground_truth,
                enable_soft_match=True,
                enable_fuzzy_match=False,
            ) or grade_law_solution_by_process(model_answer, ground_truth)
            if is_soft_correct:
                eval_result["correctness_rewards"] = self.config.correct_reward
                eval_result["soft_exact_match"] += 1
                is_hard_correct = grade_law_solution_by_outcome(
                    model_answer,
                    ground_truth,
                    enable_soft_match=False,
                    enable_fuzzy_match=False,
                ) or grade_law_solution_by_process(model_answer, ground_truth)
                if is_hard_correct:
                    eval_result["hard_exact_match"] += 1

        # Step 5. If rule-based heuristics fail and ORM is enabled, use LLM as ORM to evaluate correctness
        if self.config.use_law_orm:
            raise NotImplementedError("ORM is not supported for law problems.")

        # Step 6. If all else fails, assign incorrect reward and return
        if eval_result["correctness_rewards"] == 0:
            eval_result["correctness_rewards"] = self.config.incorrect_reward
        elif eval_result["correctness_rewards"] == 1:
            # pass
            # set all other rewards to 0 if the answer is correct
            eval_result["format_rewards"] = 0
            eval_result["length_rewards"] = 0
            eval_result["unk_error_rewards"] = 0
            eval_result["repetition_rewards"] = 0
            eval_result["language_monotony_rewards"] = 0

        # Step 7. Aggregate rewards and return
        reward = self._aggregate_rewards(eval_result)

        return RewardOutput(reward=reward, is_correct=reward), eval_result


def law_reward_fn(
    problem_str: str,
    solution_str: str,
    ground_truth: Union[str, List[str], Dict[str, str]],
    enable_llm=False,
):
    reward_config = RewardConfig()
    reward_config.use_law_orm = enable_llm
    reward_fn = RewardLawFn(reward_config)
    reward_response, eval_result = reward_fn(
        RewardInput(
            problem=problem_str,
            problem_type=RewardType.LAW,
            model_response=solution_str,
            # ground_truth={"answer": ground_truth, "keywords": None},
            ground_truth=ground_truth,
        )
    )
    return reward_response.is_correct, eval_result


if __name__ == "__main__":
    reward = RewardLawFn(RewardConfig)
    input = RewardInput(
        problem="请你给出回复的时候，在<DTK>标签前给出你的思考过程后再作答。\n根据下列事实、罪名和刑法法条预测判   决刑期。只需给出判决刑期为多少月，请将答案填在[刑期]与<eoa>之间。例如[刑期]12月<eoa>。",
        problem_type=RewardType.LAW,
        # model_response="<think> I am omniscient. </think> The answer is \\boxed{24 + 14*x + (-13)*x^2 - 2*x^3 + x^4}.",
        model_response="<think> 我是一名法官. </think> 案情的推理结果为：[刑期]6月<eoa>。",
        ground_truth={"answer": ["[刑期]6月"]},
    )
    output = reward(input)
    print(output)
