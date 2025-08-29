"""
Built-in reward functions are sourced from: https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py
Credits to the creators at hugging face and perhaps DeepSeek team, referenced the HF OpenSource AI Cookbook

The overall rewards design took inspiration from: https://platform.openai.com/docs/guides/graders
We followed the OpenAI RFL graders fine tuning design to support a wide-varity of reward config.

# System prompt that instructs the model to use a specific XML format.
SYSTEM_PROMPT = \"""
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
\"""

# XML chain-of-thought format template.
XML_COT_FORMAT = \"""
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
\"""

# thinking format template.
THINK_FORMAT = \"""
<think>
{reasoning}
</think>

Each reward function will be called with the following kwargs:
prompts, completions, completions_ids, trainer_state, **dataset_columns

Your dataset should contain either a solutions column or an answers column (for now you cannot specify a custom column name and auto-detect)
1. If it is a solutions column, you should use the "expression_accuracy" or "numerical_accuracy" reward function because they extract the final answer based on tags
2. If it is an answers column, you should use the "correctness" reward function because it does a direct string match
3. For format matching you may use all other functions (not customizable for now, either thinking or reasoning XML tags)
4. You may also use LLM-as-judge

Generally you would do the following combination:
1. Always use a format reward based on what format you are specifying in the system prompt / dataset (customizable in the future)
2. Optionally use a text similarty sort of reward if you have the reasoning process
3. Always check the final answer using a verifier (expression_accuracy, numerical_accuracy, correctness, etc) requires extracting from both
4. Optionally, or ONLY, use LLM-as-judge directly (score and label model)

In the future, we will support bringing in your custom Python reward function code!
"""

import re
import logging
from typing import Callable, Dict, List, Optional
from schema import RewardConfig


# Helper function used by multiple rewards
def extract_xml_answer(text: str) -> str:
    """Extracts content from the first <answer> tag."""
    if "<answer>" not in text:
        return text
    # Split by the start tag and take the last part
    after_start_tag = text.split("<answer>", 1)[-1]
    # Split by the end tag and take the first part
    if "</answer>" not in after_start_tag:
        return after_start_tag
    return after_start_tag.split("</answer>", 1)[0].strip()


def extract_hash_answer(text: str) -> str:
    """Extracts content after the '####' delimiter, or returns the whole string."""
    if "####" in text:
        return text.split("####", 1)[-1].strip()
    return text.strip()


def think_format_reward(
    completions: List[List[Dict[str, str]]], **kwargs
) -> List[float]:
    """
    Format Enforcement: Ensures that the generation follows a specific format
    using <think> </think> <answer> </answer> tags for reasoning.

    Args:
        completions: List of completion messages for each example
        **kwargs: Additional arguments (unused)

    Returns:
        List of rewards (1.0 if format correct, 0.0 otherwise)
    """
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    rewards = []

    for completion in completions:
        # Extract text content from completion
        if isinstance(completion, list) and len(completion) > 0:
            content = completion[0].get("content", "")
        else:
            content = str(completion)

        # Check if content matches the required format
        match = re.match(pattern, content.strip(), re.DOTALL | re.MULTILINE)
        rewards.append(1.0 if match else 0.0)

    return rewards


def expression_accuracy_reward(
    completions: List[List[Dict[str, str]]], solution: List[str], **kwargs
) -> List[Optional[float]]:
    """
    Solution Accuracy: Verifies whether the solution to the problem is correct,
    comparing it to the solution column in the dataset.

    Uses math verification for mathematical expressions when available,
    falls back to normalized text comparison.

    Args:
        completions: List of completion messages for each example
        solution: List of ground truth solutions from dataset
        **kwargs: Additional arguments (unused)

    Returns:
        List of rewards (1.0 if correct, 0.0 if incorrect, None if verification failed)
    """
    try:
        from math_verify import LatexExtractionConfig, parse, verify
        from latex2sympy2_extended import NormalizationConfig

        math_verify_available = True
    except ImportError:
        logging.warning(
            "math_verify not available, falling back to text comparison only"
        )
        math_verify_available = False

    rewards = []

    for completion, sol in zip(completions, solution):
        try:
            # Extract content from completion
            if isinstance(completion, list) and len(completion) > 0:
                content = completion[0].get("content", "")
            else:
                content = str(completion)

            if math_verify_available:
                # Try parsing ground truth
                try:
                    gold_parsed = parse(sol, extraction_mode="first_match")
                except Exception:
                    gold_parsed = []

                if len(gold_parsed) != 0:
                    # Try parsing predicted answer with robust config
                    try:
                        answer_parsed = parse(
                            content,
                            extraction_config=[
                                LatexExtractionConfig(
                                    normalization_config=NormalizationConfig(
                                        nits=False,
                                        malformed_operators=False,
                                        basic_latex=True,
                                        boxed="all",
                                        units=True,
                                    ),
                                    boxed_match_priority=0,
                                    try_extract_without_anchor=False,
                                )
                            ],
                            extraction_mode="first_match",
                        )
                        reward = float(verify(gold_parsed, answer_parsed))
                    except Exception as e:
                        logging.debug(
                            f"Math verification failed: {e}, falling back to text comparison"
                        )
                        reward = None
                else:
                    # Fallback to text match
                    reward = float(content.strip().lower() == sol.strip().lower())
            else:
                # Simple text comparison fallback
                reward = float(content.strip().lower() == sol.strip().lower())

        except Exception as e:
            logging.warning(f"Error in accuracy reward calculation: {e}")
            reward = None

        rewards.append(reward)

    return rewards


def numerical_accuracy_reward(
    completions: List[List[Dict[str, str]]], solution: List[str], **kwargs
) -> List[Optional[float]]:
    """
    Numerical Accuracy: Verifies whether the numerical answer in a completion is correct.

    This function is designed for formats where the answer is clearly delimited,
    such as mathematical reasoning problems (e.g., GSM8K). It expects the model's
    completion to be in an XML format with <answer> tags, and the ground-truth
    solution to have the final answer after a '####' delimiter.

    Args:
        completions: List of completion messages for each example.
        solution: List of ground truth solutions from the dataset.
        **kwargs: Additional arguments (unused).

    Returns:
        List of rewards (1.0 if correct, 0.0 if incorrect).
    """

    def _normalize_and_compare(model_ans: str, gt_ans: str) -> bool:
        """Normalize numerical strings and compare them."""
        # Basic normalization
        model_ans_norm = model_ans.strip().replace(",", "")
        gt_ans_norm = gt_ans.strip().replace(",", "")

        if model_ans_norm == gt_ans_norm:
            return True

        # Try float comparison for cases like '500.0' vs '500'
        try:
            return float(model_ans_norm) == float(gt_ans_norm)
        except (ValueError, TypeError):
            return False

    rewards = []
    for completion, sol in zip(completions, solution):
        try:
            # Extract model's answer from the XML format
            if isinstance(completion, list) and len(completion) > 0:
                content = completion[0].get("content", "")
            else:
                content = str(completion)
            model_answer = extract_xml_answer(content)

            # Extract ground truth answer from the '####' format
            gt_answer = extract_hash_answer(sol)

            # Compare and assign reward
            reward = float(_normalize_and_compare(model_answer, gt_answer))
        except Exception as e:
            logging.warning(f"Error in numerical accuracy reward calculation: {e}")
            reward = (
                0.0  # Assign 0 reward if any error occurs during extraction/comparison
            )

        rewards.append(reward)

    return rewards


# Reward function to check correctness: compares the extracted answer from the response with the known answer.
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print(
        "-" * 20,
        f"Question:\n{q}",
        f"\nAnswer:\n{answer[0]}",
        f"\nResponse:\n{responses[0]}",
        f"\nExtracted:\n{extracted_responses[0]}",
    )
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


# Reward function that checks if the response is a digit.
def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


# Reward function that checks if the response strictly follows the desired XML format.
def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


# Reward function with a softer check for the XML format.
def soft_format_reward_func(completions, **kwargs) -> list[float]:
    # pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    # matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in responses]


# Function to count specific XML tokens and award a small reward for each.
def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


# Reward function that uses the XML token count.
def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


# Registry of available reward functions
BUILT_IN_REWARDS = {
    "think_format": think_format_reward,
    "expression_accuracy": expression_accuracy_reward,
    "numerical_accuracy": numerical_accuracy_reward,
    "correctness": correctness_reward_func,
    "is_integer": int_reward_func,
    "strict_format": strict_format_reward_func,
    "soft_format": soft_format_reward_func,
    "xml_count": xmlcount_reward_func,
}


def load_reward_functions_from_config(reward_config: RewardConfig) -> List[Callable]:
    """
    Load reward functions from configuration.

    Args:
        reward_config: List of reward function specifications with 'name' field

    Returns:
        List of callable reward functions
    """
    if not reward_config or not reward_config.functions:
        return []

    reward_functions = []

    # 1. Load built in functions
    for function_name in reward_config.functions:
        if function_name in BUILT_IN_REWARDS:
            reward_functions.append(BUILT_IN_REWARDS[function_name])
            logging.info(f"Loaded reward function: {function_name}")
        else:
            logging.error(
                f"Unknown reward function: {function_name}. "
                f"Available functions: {list(BUILT_IN_REWARDS.keys())}"
            )

    return reward_functions
