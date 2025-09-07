import logging
import time
import io
import base64
from typing import List, Dict, Any
import numpy as np
from PIL import Image

from storage import StorageStrategyFactory
from utils import infer_modality_from_messages, infer_storage_type_from_path

logger = logging.getLogger(__name__)


class InferenceOrchestrator:
    """
    Main orchestrator for inference operations.

    This class coordinates between storage strategies and inference providers
    to handle the complete inference workflow. It maintains the same public
    API as the original InferenceService for backward compatibility.
    """

    def __init__(self):
        self.providers = {}
        self._register_providers()

    def _register_providers(self):
        """Lazy import and register inference providers"""
        # Import here to avoid circular imports
        from providers import (
            BaseInferenceProvider,
            HuggingFaceInferenceProvider,
            UnslothInferenceProvider,
            VLLMInferenceProvider,
        )

        self.providers: Dict[str, BaseInferenceProvider] = {
            "huggingface": HuggingFaceInferenceProvider(),
            "unsloth": UnslothInferenceProvider(),
            "vllm": VLLMInferenceProvider(),
        }

    def _convert_messages_for_display(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Convert messages for display, handling image conversion to base64.
        Similar to preprocessing service's convert_pil_to_base64 but focused on display.
        """
        converted_messages = []

        for message in messages:
            converted_message = {
                "role": message.get("role"),
                "content": self._convert_content_for_display(
                    message.get("content", "")
                ),
            }
            converted_messages.append(converted_message)

        return converted_messages

    def _convert_content_for_display(self, content):
        """Convert content for display, handling both legacy and new formats."""
        if isinstance(content, str):
            # Legacy format - just return as is
            return content
        elif isinstance(content, list):
            # New structured format
            converted_content = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        converted_content.append(
                            {"type": "text", "text": item.get("text", "")}
                        )
                    elif item.get("type") == "image":
                        # Convert image to base64 for display
                        converted_content.append(
                            {
                                "type": "image",
                                "image": self._convert_image_to_base64(
                                    item.get("image")
                                ),
                            }
                        )
                    else:
                        converted_content.append(item)
                else:
                    converted_content.append(item)
            return converted_content
        else:
            return content

    def _convert_image_to_base64(self, img_data):
        """Convert various image formats to base64 data URL."""
        try:
            # Case 1: PIL Images
            if isinstance(img_data, Image.Image):
                buf = io.BytesIO()
                img_data.save(buf, format="PNG")
                image_bytes = buf.getvalue()
                encoded = base64.b64encode(image_bytes).decode("utf-8")
                return f"data:image/png;base64,{encoded}"

            # Case 2: HuggingFace image format: {"bytes": ..., "path": null}
            elif isinstance(img_data, dict) and "bytes" in img_data:
                image_bytes = img_data["bytes"]
                if isinstance(image_bytes, (bytes, bytearray)):
                    # Convert bytes to PIL Image, then to base64
                    pil_image = Image.open(io.BytesIO(image_bytes))
                    buf = io.BytesIO()
                    pil_image.save(buf, format="PNG")
                    encoded_bytes = buf.getvalue()
                    encoded = base64.b64encode(encoded_bytes).decode("utf-8")
                    return f"data:image/png;base64,{encoded}"

            # Case 3: Already base64 data URL
            elif isinstance(img_data, str) and img_data.startswith("data:image"):
                return img_data

            # Case 4: Plain base64 string
            elif isinstance(img_data, str):
                return f"data:image/png;base64,{img_data}"

        except Exception as e:
            logger.warning(f"Failed to convert image to base64: {e}")
            return "[Image conversion failed]"

        return str(img_data)  # Fallback

    def run_inference(
        self,
        model_source: str,
        model_type: str,
        base_model_id: str,
        prompt: str,
        use_vllm: bool = False,
    ) -> str:
        """
        Run inference with the given model and base model.

        Args:
            model_source (str): Storage path or identifier (GCS path, HF Hub repo ID, or local path)
            model_type (str): Type of model ("adapter", "merged", or "base")
            base_model_id (str): Base model identifier
            prompt (str): Input text to generate a response for
            use_vllm (bool): Whether to use vLLM for inference

        Returns:
            str: Generated text response from the model
        """
        return self.run_batch_inference(
            model_source,
            model_type,
            base_model_id,
            [[{"role": "user", "content": prompt}]],
            use_vllm,
        )[0]

    def run_batch_inference(
        self,
        model_source: str,
        model_type: str,
        base_model_id: str,
        messages: List,
        use_vllm: bool = False,
    ) -> List[str]:
        """
        Run generation for a batch of messages with the given model and base model.

        Args:
            model_source (str): Storage path or identifier (GCS path, HF Hub repo ID, or local path)
            model_type (str): Type of model ("adapter", "merged", or "base")
            base_model_id (str): Base model identifier
            messages (list): List of input message conversations to generate responses for
            use_vllm (bool): Whether to use vLLM for inference

        Returns:
            list[str]: List of generated text responses from the model
        """
        logger.info(
            f"Starting batch inference for model: {model_source} (type: {model_type}, vLLM: {use_vllm})"
        )
        start_time = time.time()

        # Determine storage type and handle model loading
        storage_type = infer_storage_type_from_path(model_source)

        if storage_type == "local":
            raise ValueError(
                "local inference is not yet supported, provide model path from gcs or hf hub"
            )
        else:
            # Need to use storage strategy for GCS or handle HF Hub
            strategy = StorageStrategyFactory.create_strategy(storage_type)
            if storage_type == "gcs":
                artifact = strategy.load_model_info(model_source)
                resolved_model_path = artifact.local_path
                provider_key = artifact.provider
            else:  # hfhub
                artifact = strategy.load_model_info(model_source)
                resolved_model_path = artifact.remote_path
                provider_key = (
                    "unsloth" if base_model_id.startswith("unsloth/") else "huggingface"
                )

        # Determine modality and provider
        modality = infer_modality_from_messages(messages)

        # Override provider selection if vLLM is explicitly requested
        if use_vllm:
            provider_key = "vllm"

        provider = self.providers.get(provider_key, "huggingface")

        try:
            outputs = provider.run_batch_inference(
                base_model_id, resolved_model_path, model_type, messages, modality
            )
        except Exception as e:
            logger.error(f"Batch inference failed with error: {str(e)}", exc_info=True)
            raise e
        finally:
            # Provider cleanup is handled within each provider's run_batch_inference method
            # Additional cleanup for storage artifacts
            if artifact:
                strategy.cleanup(artifact)

        logger.info(
            f"Batch inference for {model_source} completed in {time.time() - start_time:.2f} seconds."
        )
        return outputs

    def run_evaluation(
        self,
        model_source: str,
        model_type: str,
        base_model_id: str,
        dataset_id: str,
        task_type: str = None,
        metrics: List[str] = None,
        max_samples: int = None,
        num_sample_results: int = 3,
        use_vllm: bool = False,
    ) -> Dict[str, Any]:
        """
        Run evaluation of a fine-tuned model on a dataset.

        Args:
            model_source (str): Storage path or identifier (GCS path, HF Hub repo ID, or local path)
            model_type (str): Type of model ("adapter", "merged", or "base")
            base_model_id (str): Base model identifier
            dataset_id (str): Dataset ID to evaluate on (must have eval split)
            task_type (str, optional): Task type for predefined metric suite
            metrics (List[str], optional): Specific list of metrics to compute
            max_samples (int, optional): Maximum number of samples to evaluate
            num_sample_results (int, optional): Number of sample results to include
            use_vllm (bool): Whether to use vLLM for inference

        Returns:
            Dict containing evaluation results and metadata
        """
        from evaluation import (
            EvaluationSuite,
            prepare_evaluation_data,
        )
        from storage import storage_service

        logger.info(
            f"Starting evaluation for model: {model_source} (type: {model_type}) on dataset: {dataset_id}"
        )
        start_time = time.time()

        # Load evaluation dataset
        try:
            _, eval_dataset = storage_service.download_processed_dataset(dataset_id)
            if eval_dataset is None:
                raise ValueError(f"No evaluation split found for dataset {dataset_id}")
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_id}: {e}")
            raise

        # Limit samples if specified
        if max_samples and len(eval_dataset) > max_samples:
            eval_dataset = eval_dataset.select(range(max_samples))
            logger.info(f"Limited evaluation to {max_samples} samples")

        # Prepare evaluation data (messages and references) in a single pass
        eval_messages, references = prepare_evaluation_data(eval_dataset)

        if len(eval_messages) != len(references):
            raise ValueError(
                "Mismatch between number of evaluation messages and references"
            )

        # Store original input messages for sample results (before batch inference)
        input_messages_for_display = [messages.copy() for messages in eval_messages]

        # Generate predictions using batch inference
        try:
            predictions = self.run_batch_inference(
                model_source, model_type, base_model_id, eval_messages, use_vllm
            )
        except Exception as e:
            logger.error(f"Error during batch inference for evaluation: {e}")
            raise

        # Compute metrics
        evaluation_suite = EvaluationSuite()
        metrics_results = evaluation_suite.compute_metrics(
            predictions, references, task_type, metrics
        )

        # Extract sample results for inspection
        sample_results = []
        if len(predictions) > 0:
            sample_indices = np.random.choice(
                len(predictions),
                min(num_sample_results, len(predictions)),
                replace=False,
            )

            for idx in sample_indices:
                sample_result = {
                    "prediction": predictions[idx],
                    "reference": references[idx],
                    "sample_index": int(idx),
                }

                # Add input messages if available
                if idx < len(input_messages_for_display):
                    sample_result["input"] = self._convert_messages_for_display(
                        input_messages_for_display[idx]
                    )

                sample_results.append(sample_result)

        eval_time = time.time() - start_time
        logger.info(f"Evaluation completed in {eval_time:.2f} seconds")
        logger.info(f"Computed metrics: {metrics_results}")

        return {
            "metrics": metrics_results,
            "samples": sample_results,
            "num_samples": len(eval_dataset),
            "dataset_id": dataset_id,
            "evaluation_time": eval_time,
        }


# Default orchestrator instance
inference_orchestrator = InferenceOrchestrator()


def run_inference(
    model_source: str,
    model_type: str,
    base_model_id: str,
    prompt: str,
    use_vllm: bool = False,
) -> str:
    """
    Convenience function for running inference with different storage backends.
    **This is just a wrapper around the batch inference function!**

    Args:
        model_source (str): Storage path or identifier (GCS path, HF Hub repo ID, or local path)
        model_type (str): Type of model ("adapter", "merged", or "base")
        base_model_id (str): Base model identifier
        prompt (str): Input text to generate a response for
        use_vllm (bool): Whether to use vLLM for inference

    Returns:
        str: Generated text response from the model
    """
    return inference_orchestrator.run_inference(
        model_source, model_type, base_model_id, prompt, use_vllm
    )


def run_batch_inference(
    model_source: str,
    model_type: str,
    base_model_id: str,
    messages: List,
    use_vllm: bool = False,
) -> List[str]:
    """
    Convenience function for running batch inference with different storage backends.
    Handles model loading and output generation for both GCS and HF Hub.

    Args:
        model_source (str): Storage path or identifier (GCS path, HF Hub repo ID, or local path)
        model_type (str): Type of model ("adapter", "merged", or "base")
        base_model_id (str): Base model identifier
        messages (list): List of input texts or formatted messages to generate responses for
        use_vllm (bool): Whether to use vLLM for inference

    Returns:
        list[str]: List of generated text responses from the model
    """
    return inference_orchestrator.run_batch_inference(
        model_source, model_type, base_model_id, messages, use_vllm
    )


def run_evaluation(
    model_source: str,
    model_type: str,
    base_model_id: str,
    dataset_id: str,
    task_type: str = None,
    metrics: List[str] = None,
    max_samples: int = None,
    num_sample_results: int = 3,
    use_vllm: bool = False,
) -> Dict[str, Any]:
    """
    Convenience function for running evaluation with different storage backends.

    Args:
        model_source (str): Storage path or identifier (GCS path, HF Hub repo ID, or local path)
        model_type (str): Type of model ("adapter", "merged", or "base")
        base_model_id (str): Base model identifier
        dataset_id (str): Dataset ID to evaluate on (must have eval split)
        task_type (str, optional): Task type for predefined metric suite
        metrics (List[str], optional): Specific list of metrics to compute
        max_samples (int, optional): Maximum number of samples to evaluate
        num_sample_results (int, optional): Number of sample results to include
        use_vllm (bool): Whether to use vLLM for inference

    Returns:
        Dict containing evaluation results and metadata
    """
    return inference_orchestrator.run_evaluation(
        model_source,
        model_type,
        base_model_id,
        dataset_id,
        task_type,
        metrics,
        max_samples,
        num_sample_results,
        use_vllm,
    )
