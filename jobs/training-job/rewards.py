"""
This reward functions module supports GRPO and related RL fine tuning algorithms.
It includes two types of reward functions:

1. Simple reward engineering functions inspired by Huggingface OpenR1, commonly used in simple reasoning tasks like GSM8K
2. Advanced graders inspired by OpenAI Platform Graders, which may leverage LLM and embedding models
3. Special grader "RULER" inspired by OpenPipe's ART, most recommended for all use cases
"""

import re
import logging
import os
import numpy as np
from typing import List, Dict, Any, Callable, Optional
from functools import partial
from google import genai
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.gleu_score import sentence_gleu
from nltk.translate.meteor_score import single_meteor_score
from pydantic import BaseModel
from schema import (
    AnyGraderConfig,
    StringCheckRewardConfig,
    TextSimilarityRewardConfig,
    ScoreModelRewardConfig,
    LabelModelRewardConfig,
    PythonRewardConfig,
    BuiltInRewardConfig,
    RulerRewardConfig,
)


# --- Built-in Reward Functions ---
# These functions requires the system prompt to instruct the model to output in a certain format
# This is usually a XML based format with <reasoning> and <answer> tags
# You should provide these tags in the configurations so we can adapt to parse them properly
# If you require other use case, please consider Graders instead of Reward Functions
# Inspiration: https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py


def extract_xml_answer(text: str, answer_tag: str = "answer") -> str:
    """
    Extracts content from the first specified tag in a string.

    Args:
        text (str): The input string containing XML-like tags.
        answer_tag (str): The tag to extract content from (default is 'answer').

    Returns:
        str: The content within the specified tags, or the original text if tags are not found.
    """
    start_tag = f"<{answer_tag}>"
    end_tag = f"</{answer_tag}>"
    if start_tag not in text:
        return text
    after_start_tag = text.split(start_tag, 1)[-1]
    if end_tag not in after_start_tag:
        return after_start_tag
    return after_start_tag.split(end_tag, 1)[0].strip()


def extract_answer_from_dataset(text: str) -> str:
    """
    Extracts content after the '####' delimiter.
    This works with most datasets like GSM8K, MATH, etc.
    """
    if "####" in text:
        return text.split("####", 1)[-1].strip()
    if "<answer>" in text:
        return extract_xml_answer(text, answer_tag="answer")
    return text.strip()


def expression_accuracy_reward(
    completions: List[str], solution: List[str], **kwargs
) -> List[Optional[float]]:
    """
    Verifies mathematical expressions against a solution.
    Requires the math-verify package.
    You should ensure that the model outputs valid LaTeX expressions, otherwise it defaults to simple text comparison.
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
        reward = None
        try:
            if math_verify_available:
                gold_parsed = parse(sol, extraction_mode="first_match") if sol else []
                if gold_parsed:
                    answer_parsed = parse(
                        completion,
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
                else:
                    reward = float(completion.strip().lower() == sol.strip().lower())
            else:
                reward = float(completion.strip().lower() == sol.strip().lower())
        except Exception as e:
            logging.debug(
                f"Math verification failed for completion '{completion}': {e}"
            )
        rewards.append(reward)
    return rewards


def numerical_accuracy_reward(
    completions: List[str], solution: List[str], answer_tag: str = "answer", **kwargs
) -> List[Optional[float]]:
    """
    Verifies a numerical answer by extracting it from the XML tags and extracting the ground truth from the dataset.
    """

    def _normalize_and_compare(model_ans: str, gt_ans: str) -> bool:
        model_ans_norm, gt_ans_norm = (
            model_ans.strip().replace(",", ""),
            gt_ans.strip().replace(",", ""),
        )
        if model_ans_norm == gt_ans_norm:
            return True
        try:
            return float(model_ans_norm) == float(gt_ans_norm)
        except (ValueError, TypeError):
            return False

    rewards = []
    for completion, sol in zip(completions, solution):
        try:
            model_answer, gt_answer = (
                extract_xml_answer(completion, answer_tag=answer_tag),
                extract_answer_from_dataset(sol),
            )
            rewards.append(float(_normalize_and_compare(model_answer, gt_answer)))
        except Exception as e:
            logging.warning(f"Error in numerical accuracy reward: {e}")
            rewards.append(0.0)
    return rewards


def format_reward_func(
    completions, think_tag="reasoning", answer_tag="answer", **kwargs
) -> list[float]:
    """
    Versatile reward function to check if the model output matches a specific XML format."""
    pattern = (
        rf"^<{think_tag}>\n.*?\n</{think_tag}>\n<{answer_tag}>\n.*?\n</{answer_tag}>\n$"
    )
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text, think_tag="reasoning", answer_tag="answer") -> float:
    """Function to count specific XML tokens and award a small reward for each."""
    count = 0.0
    if text.count(f"<{think_tag}>\n") == 1:
        count += 0.125
    if text.count(f"\n</{think_tag}>\n") == 1:
        count += 0.125
    if text.count(f"\n<{answer_tag}>\n") == 1:
        count += 0.125
        count -= len(text.split(f"\n</{answer_tag}>\n")[-1]) * 0.001
    if text.count(f"\n</{answer_tag}>") == 1:
        count += 0.125
        count -= (len(text.split(f"\n</{answer_tag}>")[-1]) - 1) * 0.001
    return count


BUILT_IN_REWARDS = {
    "format_reward": format_reward_func,
    "count_xml": count_xml,
    "expression_accuracy": expression_accuracy_reward,
    "numerical_accuracy": numerical_accuracy_reward,
}


# -- Grader reward functions and helpers --
# These are graders adapted from the OpenAI Fine tuning platform Graders design
# Reference: https://platform.openai.com/docs/guides/graders
# The schema is pretty much identical but simplified and certain features are reduced.
# All graders available on the docs are supported except for Python custom functions since that requires sandboxing
# In addition, RULER grader is added you may refer to its docstring for details.

_genai_client = None


def _ensure_genai_configured(api_key: Optional[str] = None):
    """Checks if the Gemini API is configured and configures it if not."""
    global _genai_client
    if _genai_client:
        return

    # Prioritize the key from config, fall back to environment variable
    final_api_key = api_key or os.environ.get("GOOGLE_API_KEY")

    if final_api_key:
        try:
            _genai_client = genai.Client(api_key=final_api_key)
            logging.info("Successfully configured Gemini API.")
        except Exception as e:
            logging.error(f"Failed to configure Gemini API: {e}")
            _genai_client = None
    else:
        logging.warning(
            "Gemini API key not provided in config or GOOGLE_API_KEY environment variable. Model-based graders will not function."
        )


def _resolve_template(
    template_string: str, completion: str, sample: Dict[str, Any]
) -> str:
    """
    Resolves template placeholders in a string with actual data.
        1. Replaces {{ completion }} with the actual completion text.
        2. Replaces {{{ sample.field_name }}} with the corresponding value from the sample dictionary.

    Args:
        template_string (str): The template string containing placeholders.
        completion (str): The model-generated completion text.
        sample (Dict[str, Any]): A dictionary representing the current data sample,
            this involves all the columns in the dataset at the present row

    Returns:
        str: The resolved string with placeholders replaced by actual values.
    """
    resolved_string = template_string.replace("{{ completion }}", completion)
    for match in re.finditer(r"{{{\s*sample\.([^\s]+)\s*}}}", resolved_string):
        key = match.group(1)
        if key in sample:
            resolved_string = resolved_string.replace(match.group(0), str(sample[key]))
    return resolved_string


def _call_gemini_grader(
    model: str,
    message: str,
    system_prompt: Optional[str] = None,
    schema: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
):
    """
    Calls the Gemini API for model-based grading after ensuring configuration.

    Args:
        model (str): The Gemini model to use for grading.
        message (str): The user prompt or message to send to the model.
        system_prompt (Optional[str]): An optional system prompt to guide the model's behavior.
        schema (Optional[Dict[str, Any]]): An optional JSON schema to enforce structured responses.
        api_key (Optional[str]): An optional API key for Gemini, if not provided, it will use the environment variable.

    Returns:
        The response in the format specified by the schema, or an empty string on failure.
    """
    _ensure_genai_configured(api_key)
    if not _genai_client:
        return ""
    try:
        response = _genai_client.models.generate_content(
            model=model,
            contents=[message],
            config={
                "system_instruction": system_prompt,
                "response_schema": schema,
            },
        )
        logging.debug(f"Gemini grader response: {response}")
        return response
    except Exception as e:
        logging.error(f"Error calling Gemini API: {e}")
        return ""


def string_check_reward(
    config: StringCheckRewardConfig, completions: List[str], **kwargs
) -> List[float]:
    """
    Calculates reward based on simple string comparisons.
    NOTE: You should avoid using this and use format or accuracy rewards if possible.
    If you need to check text, try to use text_similarity_reward because this is very brittle.
    """
    reference_values = kwargs.get(config.reference_field, [])
    op_map = {
        "eq": lambda c, r: c == r,
        "ne": lambda c, r: c != r,
        "like": lambda c, r: r in c,
        "ilike": lambda c, r: r.lower() in c.lower(),
    }
    op = op_map[config.operation]
    return [
        1.0 if op(comp, ref) else 0.0
        for comp, ref in zip(completions, reference_values)
    ]


def text_similarity_reward(
    config: TextSimilarityRewardConfig, completions: List[str], **kwargs
) -> List[float]:
    """
    Calculates reward based on various text similarity metrics.
    Requires a reference field in the dataset to compare against.
    For example if you have a "solution" column in the dataset, it will be picked up and used here.
    """
    reference_values = kwargs.get(config.reference_field, [])
    metric = config.evaluation_metric
    rewards = []
    for comp, ref in zip(completions, reference_values):
        score = 0.0
        if metric == "cosine":
            _ensure_genai_configured(config.gemini_api_key)
            if _genai_client:
                try:
                    # Generate embeddings
                    comp_emb = _genai_client.models.embed_content(
                        model=config.embedding_model, content=comp
                    )["embeddings"]
                    ref_emb = _genai_client.models.embed_content(
                        model=config.embedding_model, content=ref
                    )["embeddings"]
                    # Calculate cosine similarity
                    score = np.dot(comp_emb, ref_emb) / (
                        np.linalg.norm(comp_emb) * np.linalg.norm(ref_emb)
                    )
                except Exception as e:
                    logging.error(f"Cosine similarity failed: {e}")
        else:
            comp_tokens, ref_tokens = comp.split(), ref.split()
            if metric == "bleu":
                score = sentence_bleu([ref_tokens], comp_tokens)
            elif metric == "gleu":
                score = sentence_gleu([ref_tokens], comp_tokens)
            elif metric == "meteor":
                score = single_meteor_score(ref_tokens, comp_tokens)
            elif metric.startswith("rouge"):
                score = (
                    rouge_scorer.RougeScorer([metric], use_stemmer=True)
                    .score(ref, comp)[metric]
                    .fmeasure
                )
            # We can support other metrics if possible, but for GLEU and cosine are the best
        rewards.append(score)
    return rewards


SCORE_MODEL_SYSTEM_PROMPT = """You are an AI judge. Your task is to provide a numerical score for a given model response based on a user's instruction.

You must respond in a valid JSON format. The JSON object should have the following schema:
{
  "explanation": "A short, clear explanation for your score.",
  "score": "A float between 0.0 and 1.0, where 1.0 is the best score."
}"""

LABEL_MODEL_SYSTEM_PROMPT = """You are an AI judge. Your task is to classify a given model response with one of the provided labels based on a user's instruction.

You must respond in a valid JSON format. The JSON object should have the following schema:
{
  "explanation": "A short, clear explanation for your choice.",
  "label": "One of the provided labels."
}"""


class ScoreModelResponse(BaseModel):
    """Pydantic model for parsing the score model judge's response."""

    explanation: str
    score: float


class LabelModelResponse(BaseModel):
    """Pydantic model for parsing the label model judge's response."""

    explanation: str
    label: str


def score_model_reward(
    config: ScoreModelRewardConfig,
    completions: List[str],
    **kwargs,
) -> List[float]:
    """
    Calculates reward by calling a score-based LLM judge.
    This function uses a Gemini model to evaluate each completion and assign a numerical score.
    The user should specify a prompt that includes the {{completion}} placeholder and any relevant context from the dataset.
    For example, this prompt can include how should the model judge, expected format, example responses and their scores, etc.

    NOTE: This runs a loop over all the completions instead of batching them into one call, so that the judge LLM only
    sees one completion at a time. This is to avoid the model comparing completions against each other. You should use RULER
    if you intend otherwise, it is also more cost effective by reducing the number of judge LLM calls. e.g. this will call
    the judge LLM N times for N completions, while RULER calls it only once for N completions (N = number of generations).
    """
    rewards = []
    system_prompt = SCORE_MODEL_SYSTEM_PROMPT
    if config.range:
        system_prompt += f"\nThe score must be within the range {config.range}."

    for i, completion in enumerate(completions):
        # structure: {col_name: value_for_this_iter, ...}
        # We skip any values that are not indexable or have the wrong length e.g. trainer_state
        current_sample = {
            key: values[i]
            for key, values in kwargs.items()
            if hasattr(values, "__getitem__")
            and hasattr(values, "__len__")
            and len(values) == len(completions)
        }
        user_prompt = _resolve_template(config.prompt, completion, current_sample)

        response_text = _call_gemini_grader(
            config.model,
            user_prompt,
            system_prompt,
            ScoreModelResponse.model_json_schema(),
            api_key=config.gemini_api_key,
        )
        score = 0.0
        try:
            parsed_response = ScoreModelResponse.model_validate_json(response_text)
            score = parsed_response.score
            if config.range:
                score = max(config.range[0], min(score, config.range[1]))
        except Exception as e:
            logging.warning(
                f"Failed to parse score model judge response: {e}\nResponse was: {response_text}"
            )
        rewards.append(score)
    return rewards


def label_model_reward(
    config: LabelModelRewardConfig,
    completions: List[str],
    **kwargs,
) -> List[float]:
    """
    Calculates reward by calling a label-based LLM judge.
    The user specificiation is very similar to score_model_reward, except that the model must choose from a set of labels.
    It returns 1.0 if the chosen label is in the passing_labels list, otherwise 0.0.
    """
    rewards = []
    labels_text = ", ".join(config.labels)
    system_prompt = f"{LABEL_MODEL_SYSTEM_PROMPT}\nYou must choose one of the following labels: [{labels_text}]"

    for i, completion in enumerate(completions):
        current_sample = {
            key: values[i]
            for key, values in kwargs.items()
            if hasattr(values, "__getitem__")
            and hasattr(values, "__len__")
            and len(values) == len(completions)
        }
        user_prompt = _resolve_template(config.prompt, completion, current_sample)

        response_text = _call_gemini_grader(
            config.model,
            user_prompt,
            system_prompt,
            LabelModelResponse.model_json_schema(),
            api_key=config.gemini_api_key,
        )
        score = 0.0
        try:
            parsed_response = LabelModelResponse.model_validate_json(response_text)
            if parsed_response.label in config.passing_labels:
                score = 1.0
        except Exception as e:
            logging.warning(
                f"Failed to parse label model judge response: {e}\nResponse was: {response_text}"
            )
        rewards.append(score)
    return rewards


@NotImplementedError(
    "Executing python reward functions is not supported since it requires sandboxing."
)
def python_reward(
    config: PythonRewardConfig, completions: List[str], **kwargs
) -> List[float]:
    """Placeholder for executing custom Python code."""
    logging.warning(
        f"Python grader '{config.name}' is a placeholder and will return 0.0 for all samples."
    )
    return [0.0] * len(completions)


RULER_SYSTEM_PROMPT = """
You are an impartial AI judge. Your role is to evaluate a batch of model-generated responses based on a given set of rules, and score them relative to each other.

You will be provided with:
1. A list of rules.
2. A list of model-generated responses, each with a unique `completion_id`.

Your task is to evaluate each response and provide a score from 0.0 to 1.0 for each one. A score of 1.0 is best. You should consider the relative quality of the responses when assigning scores.

You must respond in a valid JSON format. Do not add any text outside of the JSON structure.

The JSON object should have the following schema:
{
  "scores": [
    {
      "completion_id": "The integer ID of the completion you are scoring.",
      "explanation": "A short, clear explanation for your score.",
      "score": "A float between 0.0 and 1.0."
    }
  ]
}
"""


class RulerScore(BaseModel):
    """Pydantic model for a single RULER score."""

    completion_id: int
    explanation: str
    score: float


class RulerBatchResponse(BaseModel):
    """Pydantic model for parsing the RULER judge's batch response."""

    scores: List[RulerScore]


def ruler_reward(
    config: RulerRewardConfig,
    completions: List[str],
    **kwargs,
) -> List[float]:
    """
    Calculates reward by calling a RULER-style LLM judge on a batch of completions.
    This is a reference-free grader that evaluates responses against a set of rules / rubric.

    For RULER, refer to: https://art.openpipe.ai/fundamentals/ruler
    RULER (Relative Universal LLM Elicit Rewards) is a type of LLM-as-judge reward function.
    However, it differs from directing asking for score and labels in the sense that it evaluates responses based on their relative quality to each other.
    It also considers a set of explicit rules from the user, which acts like "soft" reward engineering.
    This implementation is adapted and simplified from ART (Agent reinforcement trainer) by OpenPipe, replaces "trajectories" for agents with more generic use cases.

    Our implementation is based on ART's RULER's steps:
    1. Generate N trajectories for a given scenario
    2. Pass all N trajectories to RULER
    3. RULER deduplicates common prefixes (e.g., identical system messages)
    4. An LLM judge scores each trajectory from 0 to 1 based on goal achievement
    5. These scores are used directly as rewards in GRPO training

    NOTE: We cannot provide a default rubric because unlike ART's agent scenarios, we don't know what the user wants to achieve.
    TODO: We are missing some optimizations from ART, such as deduplication of common prefixes
    """
    if not completions:
        return []

    rules_text = "\n".join(f"- {rule}" for rule in config.rules)

    # Prepare the single prompt with all completions
    completions_text = []
    for i, completion in enumerate(completions):
        completions_text.append(f'<completion id="{i}">\n{completion}\n</completion>')

    user_prompt = f"""Please evaluate the following responses based on these rules:

    **Rules:**
    {rules_text}

    **Responses to Evaluate:**
    {chr(10).join(completions_text)}
    """

    response_text = _call_gemini_grader(
        config.model,
        user_prompt,
        RULER_SYSTEM_PROMPT,
        RulerBatchResponse.model_json_schema(),
        api_key=config.gemini_api_key,
    )

    # Initialize rewards with a default of 0.0
    rewards = [0.0] * len(completions)
    try:
        parsed_response = RulerBatchResponse.model_validate_json(response_text)

        if len(parsed_response.scores) != len(completions):
            logging.warning(
                f"RULER judge returned a different number of scores ({len(parsed_response.scores)}) than completions ({len(completions)})."
            )

        for score_obj in parsed_response.scores:
            comp_id = score_obj.completion_id
            if 0 <= comp_id < len(rewards):
                # Ensure score is within the valid range
                score = max(0.0, min(score_obj.score, 1.0))
                rewards[comp_id] = score
            else:
                logging.warning(
                    f"RULER judge returned an invalid completion_id: {comp_id}"
                )

    except Exception as e:
        logging.warning(
            f"Failed to parse RULER judge batch response: {e}\nResponse was: {response_text}"
        )

    return rewards


# --- Dispatcher and Loader ---

GRADER_DISPATCHER = {
    "built_in": BUILT_IN_REWARDS,
    "ruler": ruler_reward,
    "string_check": string_check_reward,
    "text_similarity": text_similarity_reward,
    "score_model": score_model_reward,
    "label_model": label_model_reward,
    "python": python_reward,
}


def load_reward_functions_from_config(
    reward_config: List[AnyGraderConfig],
) -> List[Callable]:
    """
    Loads and prepares a list of reward functions from the configuration.

    This factory interprets the `RewardConfig` and returns a list of callables that the TRL trainer can execute.

    Args:
        reward_config: The Pydantic model instance defining which rewards to load.

    Returns:
        A list of callable reward functions.
    """
    graders = []
    for config in reward_config:
        func = None
        if isinstance(config, BuiltInRewardConfig):
            func = GRADER_DISPATCHER["built_in"].get(config.function_name)
            if func:
                if config.parameters:
                    graders.append(
                        partial(func, **config.parameters.model_dump(exclude_none=True))
                    )
                else:
                    graders.append(func)
                logging.info(
                    f"Loaded built-in reward function: '{config.function_name}'"
                )
        else:
            func = GRADER_DISPATCHER.get(config.type)
            if func:
                graders.append(partial(func, config))
                logging.info(
                    f"Loaded grader-style reward function: type='{config.type}', name='{config.name}'"
                )

        if not func:
            logging.error(f"Could not find reward function for config: {config}")

    return graders
