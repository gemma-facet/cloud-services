import logging
import numpy as np
import evaluate
from typing import List, Dict, Any, Tuple
from datasets import Dataset

logger = logging.getLogger(__name__)


class EvaluationSuite:
    """
    Evaluation suite for computing metrics on model generations.
    Supports individual metrics and combined metric suites for specific tasks.
    """

    def __init__(self):
        # Combined metric suites for specific task types
        self.task_metric_suites = {
            "conversation": ["bertscore", "rouge"],
            "qa": ["exact_match", "bertscore"],
            "summarization": ["rouge", "bertscore"],
            "translation": ["bleu", "meteor"],
            "classification": ["accuracy", "recall", "precision", "f1"],
            "general": ["bertscore", "rouge"],
        }

        # stores loaded individual or grouped metrics
        self.loaded_metrics = {}

    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types recursively."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        return obj

    def _load_metric(self, metric_name: str):
        """Load a single metric by name."""
        try:
            return evaluate.load(metric_name)
        except Exception as e:
            logger.warning(f"Could not load metric {metric_name}: {e}")
            return None

    def _load_combined_metrics(self, metric_names: List[str]):
        """Load and combine multiple metrics into one evaluator."""
        try:
            return evaluate.combine(metric_names)
        except Exception as e:
            logger.warning(f"Could not load combined metrics {metric_names}: {e}")
            return None

    def load_metrics(self, metric_names: List[str]):
        """Load specified metrics individually."""
        for metric_name in metric_names:
            if metric_name not in self.loaded_metrics:
                metric = self._load_metric(metric_name)
                if metric is not None:
                    self.loaded_metrics[metric_name] = metric
                    logger.info(f"Loaded metric: {metric_name}")

    def compute_metrics(
        self,
        predictions: List[str],
        references: List[str],
        task_type: str = None,
        metrics: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Compute metrics on predictions vs references.

        Args:
            predictions: List of model generated texts
            references: List of reference/ground truth texts
            task_type: Task type to use predefined metric suite (mutually exclusive with metrics)
            metrics: Specific list of metrics to compute (mutually exclusive with task_type)

        Returns:
            Dictionary containing metric scores
        """
        # Determine which metrics to use
        if task_type:
            final_metrics = self.task_metric_suites.get(
                task_type, self.task_metric_suites["general"]
            )
        else:
            final_metrics = metrics or []

        # Load metrics
        self.load_metrics(final_metrics)

        results = {}  # results[metric_name] = int or dict

        # Compute individual metrics
        for metric_name in final_metrics:
            if metric_name not in self.loaded_metrics:
                logger.warning(f"Metric {metric_name} not available, skipping")
                continue

            metric = self.loaded_metrics[metric_name]
            try:
                if metric_name == "bleu":
                    # BLEU expects references as list of lists
                    refs = [[ref] for ref in references]
                    result = metric.compute(predictions=predictions, references=refs)
                    results[metric_name] = result["bleu"]

                elif metric_name == "bertscore":
                    result = metric.compute(
                        predictions=predictions, references=references, lang="en"
                    )
                    # This returns: {'precision': [0.7, 0.5], 'recall': [], 'f1': []} if there are two pred ref pairs
                    # This takes the mean across all the sample similarities and preserves nested structure
                    results["bertscore"] = {
                        "precision_mean": np.mean(result["precision"]),
                        "recall_mean": np.mean(result["recall"]),
                        "f1_mean": np.mean(result["f1"]),
                    }

                elif metric_name == "f1":
                    # F1 returns {'f1': 0.26666666666666666} or {'f1': array([0.8, 0.0, 0.0])} depending on binary or multi-class
                    result = metric.compute(
                        predictions=predictions, references=references
                    )
                    if len(set(references)) <= 2:
                        results["f1"] = result["f1"]  # Binary classification case
                    else:
                        # Multi-class case, alternatively you can set average="micro"
                        results["f1"] = np.mean(result["f1"])
                        results["f1_per_class"] = result["f1"]

                # no need special handling
                else:
                    result = metric.compute(
                        predictions=predictions, references=references
                    )
                    # Handle different result formats
                    if isinstance(result, dict) and metric_name in result:
                        results[metric_name] = result[
                            metric_name
                        ]  # e.g. {"accuracy": 0.8}
                    else:
                        results[metric_name] = (
                            result  # e.g. {"rougeL": xxx, "rouge1": xxx, "rouge2": xxx}
                        )

            except Exception as e:
                logger.error(f"Error computing {metric_name}: {e}")
                results[metric_name] = 0.0

        return self._convert_numpy_types(results)


def prepare_evaluation_data(
    dataset: Dataset,
) -> Tuple[List[List[Dict[str, Any]]], List[str]]:
    """
    Prepare evaluation data by extracting both input messages and reference texts.

    Handles multiple dataset formats:
    1. NEW format: {"messages": [...], "images": [PIL.Image, ...]} - converts to OLD format
    2. OLD format: {"messages": [...]} with images embedded in content
    3. Prompt-only format: {"prompt": [...], "images": [...], "answer": "...", ...}
    4. Preference format: {"prompt": [...], "chosen": [...], "rejected": [...], "images": [...]}
    5. Other formats: {"conversation": [...], etc.}

    Converts NEW format back to OLD format by embedding images from "images" field
    into message content for compatibility with inference pipeline.

    Args:
        dataset: HuggingFace dataset in various formats

    Returns:
        Tuple of (eval_messages, references) where:
        - eval_messages: List of message conversations with only user/system messages (for generation)
        - references: List of reference texts (assistant responses)
    """
    eval_messages = []
    references = []

    for example in dataset:
        images = example.get("images", [])  # PIL images from NEW format

        # Handle different dataset formats
        if "messages" in example:
            # Standard conversational format
            messages = example["messages"]
            if images:
                messages = _convert_new_format_to_old_format(messages, images)

            input_messages, reference = _extract_from_conversational_format(messages)

        elif "prompt" in example:
            # Prompt-only or preference format
            prompt = example["prompt"]
            if images:
                prompt = _convert_new_format_to_old_format(prompt, images)

            input_messages = prompt if isinstance(prompt, list) else [prompt]

            # For prompt-only, look for answer/response fields
            reference = None
            for ref_field in ["answer", "response", "chosen", "output"]:
                if ref_field in example and example[ref_field]:
                    reference = str(example[ref_field])
                    break

            if reference is None:
                reference = ""
                logger.warning("No reference field found in prompt-only format")

        else:
            logger.warning(
                "No recognized conversation field found in example, skipping"
            )
            continue

        # Add to results if we have input messages
        if input_messages:
            eval_messages.append(input_messages)
            references.append(reference.strip() if reference else "")
        else:
            logger.warning("No input messages found in example, skipping")

    return eval_messages, references


def _extract_from_conversational_format(messages: List[Dict]) -> Tuple[List[Dict], str]:
    """
    Extract input messages and reference from conversational format.

    Args:
        messages: List of message dicts in ChatML format

    Returns:
        Tuple of (input_messages, reference)
    """
    input_messages_for_eval = []
    reference = None

    for message in messages:
        if message.get("role") in ["user", "system"]:
            input_messages_for_eval.append(message)
        elif message.get("role") == "assistant":
            # First assistant message becomes the reference (extract text only)
            if reference is None:
                content = message.get("content", "")
                # Handle both new structured format and legacy format
                if isinstance(content, list):
                    # New format: content is list of objects with type/text fields
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_parts.append(str(item.get("text", "")))
                        elif isinstance(item, str):
                            text_parts.append(item)
                        else:
                            # For other types (like images), convert to string
                            text_parts.append(str(item))
                    reference = " ".join(text_parts)
                else:
                    # Legacy format: content is a simple string
                    reference = str(content)
            # Stop at first assistant message - we want to generate from this point
            break

    return input_messages_for_eval, reference or ""


def _convert_new_format_to_old_format(messages: List[Dict], images: List) -> List[Dict]:
    """
    Convert NEW format (separate images field) to OLD format (embedded images).

    NEW format:
    - messages: [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "..."}]}]
    - images: [PIL.Image, PIL.Image, ...]

    OLD format:
    - messages: [{"role": "user", "content": [{"type": "image", "image": PIL.Image}, {"type": "text", "text": "..."}]}]

    Args:
        messages: List of message dicts with image placeholders
        images: List of PIL Image objects

    Returns:
        List of message dicts with images embedded in content
    """
    if not images:
        return messages

    converted_messages = []
    image_index = 0

    for message in messages:
        converted_message = message.copy()
        content = message.get("content", [])

        if isinstance(content, list):
            converted_content = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    # Replace placeholder with actual image
                    if image_index < len(images):
                        converted_content.append(
                            {
                                "type": "image",
                                "image": images[image_index],  # PIL Image object
                            }
                        )
                        image_index += 1
                    else:
                        logger.warning(
                            "Not enough images for placeholders, skipping placeholder"
                        )
                else:
                    # Keep other content as-is
                    converted_content.append(item)

            converted_message["content"] = converted_content

        converted_messages.append(converted_message)

    return converted_messages
