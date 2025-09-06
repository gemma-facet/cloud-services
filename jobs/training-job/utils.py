import logging
import torch
from typing import Callable, Tuple, Any
from storage import StorageStrategyFactory, CloudStoredModelMetadata
from job_manager import JobTracker
from schema import ExportConfig
import os
import shutil
import numpy as np


def construct_full_model_id(base_model_id: str, provider: str, method: str) -> str:
    """
    Construct the full model ID based on base model, provider, and quantization method.
    NOTE: This is backward compatible, i.e. if you provide google/gemma-3-1b-it it will still work.

    Args:
        base_model_id: Base model name (e.g., "gemma-3-1b-it")
        provider: Training provider ("unsloth" or "huggingface")
        method: Training method ("Full", "LoRA", "QLoRA")

    Returns:
        Full model ID with proper namespace and quantization suffix
    """
    if "/" in base_model_id:
        # Already a full model ID
        return base_model_id

    if provider == "huggingface":
        # For HuggingFace, always use google/ namespace
        return f"google/{base_model_id}"
    elif provider == "unsloth":
        # For Unsloth, use unsloth/ namespace
        if method == "QLoRA":  # default to unsloth dynamic quants
            return f"unsloth/{base_model_id}-unsloth-bnb-4bit"
        else:
            return f"unsloth/{base_model_id}"
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def run_evaluation(trainer):
    """
    Run evaluation on the trainer, log and return metrics.
    """
    eval_results = trainer.evaluate()
    logging.info(f"Evaluation results: {eval_results}")
    return eval_results


def _prepare_model_for_export(
    model, tokenizer, export_config: ExportConfig, provider: str, temp_dir: str
) -> Tuple[Any, Any, str]:
    """
    Prepare model for export based on the export configuration.
    Uses provider-specific saving methods (Unsloth vs HuggingFace).
    We currently don't support ONNX yet might do that soon!

    NOTE: This will save everything locally first, then upload to either gcs or hub later.
    We can choose to directly call push_to_hub or push_to_hub_merged as well they are functionally identical.

    After exporting model, this guide tells you how to match the quantisation when loading e.g. from transformer.js
    https://huggingface.co/docs/transformers.js/en/guides/dtypes

    Args:
        model: The trained model
        tokenizer: The tokenizer
        export_config: Export configuration
        provider: Training provider ("unsloth" or "huggingface")
        temp_dir: Temporary directory for saving

    Returns:
        Tuple of (processed_model, tokenizer, actual_temp_dir)
    """
    logging.info(
        f"Preparing model for export with format: {export_config.format}, provider: {provider}"
    )

    # For unsloth this also works: model.save_pretrained_merged("model", tokenizer, save_method = "lora",)
    if export_config.format == "adapter":
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(temp_dir)
        else:
            model.save_model(temp_dir)  # For trainers
        tokenizer.save_pretrained(temp_dir)
        return model, tokenizer, temp_dir

    # Handle merged format for both unsloth and hf
    elif export_config.format == "merged":
        if provider == "unsloth":
            try:
                # Unsloth's save_pretrained methods save BOTH the model (formatted) AND tokenizer
                # This format (merged 16bit) works for vLLM: https://docs.unsloth.ai/basics/running-and-saving-models/saving-to-vllm
                model.save_pretrained_merged(
                    temp_dir, tokenizer, save_method="merged_16bit"
                )
                # TODO: alternatively we can use "merged_4bit" which is helpful for HF inference engine, but it sometimes breaks
                return model, tokenizer, temp_dir
            except Exception as e:
                logging.error(f"Error during Unsloth model merging: {e}")
                raise
        else:
            try:
                if hasattr(model, "merge_and_unload"):
                    # only for PEFT models generated using get_peft_model()
                    # NOTE: This casts to 4bit by default, and you cannot switch the quantisation later
                    # This seems to be the only method that works, model.merge_adapter() doesn't seem to work
                    merged_model = model.merge_and_unload()
                else:
                    # Already a merged model or full fine-tuning
                    merged_model = model

                # save_pretrained automatically handles 8bit and 4bit using bitsandbytes
                merged_model.save_pretrained(temp_dir, safe_serialization=True)
                tokenizer.save_pretrained(temp_dir)
                return merged_model, tokenizer, temp_dir
            except Exception as e:
                logging.error(f"Error during HuggingFace model merging: {e}")
                raise
    else:
        raise ValueError(f"Unsupported export format: {export_config.format}")


def _prepare_gguf_export(
    model, tokenizer, export_config: ExportConfig, provider: str, temp_dir: str
) -> str:
    """
    Prepare GGUF export separately. This function handles GGUF export for both providers.

    Args:
        model: The trained model
        tokenizer: The tokenizer
        export_config: Export configuration
        provider: Training provider ("unsloth" or "huggingface")
        temp_dir: Temporary directory for saving

    Returns:
        Path to the generated GGUF file
    """
    if provider == "unsloth":
        return _prepare_unsloth_gguf_export(model, tokenizer, export_config, temp_dir)
    else:
        raise NotImplementedError(
            "HuggingFace GGUF export requires llama.cpp conversion tools, do it manually for now!"
        )


def _prepare_unsloth_gguf_export(
    model, tokenizer, export_config: ExportConfig, temp_dir: str
) -> str:
    """Handle Unsloth-specific GGUF export methods"""
    logging.info(
        f"Saving Unsloth GGUF with quantization: {export_config.gguf_quantization}"
    )

    try:
        # First check if temp_dir exists and contain config.json i.e. already merged
        if os.path.exists(temp_dir) and os.path.isfile(
            os.path.join(temp_dir, "config.json")
        ):
            gguf_file_path = model.save_pretrained_gguf(
                temp_dir, quantization_method=export_config.gguf_quantization
            )
        else:
            # We first need to save a merged model then convert to GGUF with quant
            gguf_temp_dir = f"{temp_dir}_gguf_intermediate"
            model.save_pretrained_merged(
                gguf_temp_dir, tokenizer, save_method="merged_16bit"
            )
            # This takes model saved in gguf_temp_dir to create a .gguf file separately
            gguf_file_path = model.save_pretrained_gguf(
                gguf_temp_dir, quantization_method=export_config.gguf_quantization
            )
            logging.info(f"GGUF file converted and saved to {gguf_file_path[0]}")
        return gguf_file_path[0]  # gguf_file_path is List[str]
    except Exception as e:
        logging.error(f"Error during Unsloth GGUF export: {e}")
        raise


def save_and_track(
    export_config: ExportConfig,
    model,
    tokenizer,
    job_id: str,
    base_model_id: str,
    provider: str,
    job_tracker: JobTracker,
    metrics: dict | None = None,
):
    """
    Save the model using storage strategy, cleanup, and mark job as completed.
    Saves the model in the format specified by export_config (adapter or merged).
    Optionally also exports GGUF if requested.
    The inference service will handle both adapter and merged models automatically.
    """
    # Determine temp directory based on export format
    temp_dir = f"/tmp/{job_id}_{export_config.format}"
    gguf_file_path = None

    try:
        # Prepare primary model for export based on configuration
        processed_model, processed_tokenizer, actual_temp_dir = (
            _prepare_model_for_export(
                model, tokenizer, export_config, provider, temp_dir
            )
        )

        # Prepare GGUF export if requested
        if export_config.include_gguf:
            gguf_file_path = _prepare_gguf_export(
                model, tokenizer, export_config, provider, temp_dir
            )

        # Use export_config.destination to determine storage strategy
        storage_strategy = StorageStrategyFactory.create_strategy(
            export_config.destination
        )

        # Save primary model
        metadata = CloudStoredModelMetadata(
            job_id=job_id,
            base_model_id=base_model_id,
            gcs_prefix="",  # Will be set by storage service based on format
            provider=provider,
            local_dir=actual_temp_dir,
            hf_repo_id=export_config.hf_repo_id,
            export_format=export_config.format,
        )

        # The model has already been saved locally by our export logic above
        artifact = storage_strategy.save_model(
            actual_temp_dir,  # Local directory containing saved files
            metadata,
        )

        if gguf_file_path:
            # Use save_file for single GGUF file upload, destination is always same as model
            # NOTE: remote path at GCS is always gguf_models/{job_id}/{filename}
            # This is for easier access given a job_id
            gguf_remote_path = storage_strategy.save_file(
                gguf_file_path,  # Direct path to GGUF file
                f"gguf_models/{job_id}/{os.path.basename(gguf_file_path)}"
                if export_config.destination == "gcs"
                else export_config.hf_repo_id,
            )

        # Cleanup storage artifacts
        storage_strategy.cleanup(artifact)

        # Mark job as completed with primary artifact path
        job_tracker.completed(
            artifact.remote_path,
            artifact.base_model_id,
            metrics,
            gguf_path=gguf_remote_path if gguf_file_path else None,
        )

        return artifact

    except Exception as e:
        logging.error(f"Error during model export: {e}")
        # Clean up temp directory on error
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def preprocess_logits_for_metrics(logits, labels):
    # This is a workaround to avoid storing too many tensors that are not needed.
    # This will preprocess the logits before they are cached for metrics computation
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids, labels


def create_compute_metrics(
    compute_eval_metrics: bool = False, use_batched_eval: bool = False
) -> Callable:
    """
    Create a compute_metrics function for accuracy and perplexity computation.
    Computes metrics manually without external dependencies.

    Args:
        compute_eval_metrics: If True, compute accuracy and perplexity
        use_batched_eval: If True, enables batch evaluation mode for metrics computation.

    Returns:
        A compute_metrics function for use with HuggingFace Trainer
    """
    if not compute_eval_metrics:
        return None

    def compute_metrics(eval_pred) -> dict:
        """
        Compute evaluation metrics manually (regular mode).
        """

        logits, labels = eval_pred
        predictions = (
            np.argmax(logits, axis=-1) if isinstance(logits, np.ndarray) else logits
        )
        results = {}

        # Get loss from eval_pred if available
        loss = (
            eval_pred.metrics.get("eval_loss", 0)
            if hasattr(eval_pred, "metrics")
            else 0
        )
        results["loss"] = loss

        # Flatten and filter out ignored tokens (-100)
        flat_predictions = predictions.flatten()
        flat_labels = labels.flatten()
        valid_mask = flat_labels != -100
        valid_predictions = flat_predictions[valid_mask]
        valid_labels = flat_labels[valid_mask]

        # Compute accuracy manually
        if len(valid_labels) > 0:
            correct = (valid_predictions == valid_labels).sum()
            accuracy = correct / len(valid_labels)
            results["accuracy"] = float(accuracy)
        else:
            results["accuracy"] = 0.0

        # Compute perplexity from loss
        if loss > 0:
            results["perplexity"] = float(2**loss)
        else:
            results["perplexity"] = 1.0

        return results

    if not use_batched_eval:
        return compute_metrics

    # Storage for batch-level statistics
    batch_stats = {
        "total_correct": 0,
        "total_samples": 0,
        "total_loss": 0.0,
        "batch_count": 0,
    }

    def compute_metrics_batched(eval_pred, compute_result: bool = True) -> dict:
        """
        Compute evaluation metrics in batch mode.
        NOTE: If we ever use evaluate library in the future, the class of metric has an `add_batch`
        method that we can use to accumulate batch statistics and then `compute` at the end.

        Args:
            eval_pred: EvalPrediction containing predictions and labels
            compute_result: If True, compute final results. If False, accumulate batch stats.

        Returns:
            Dictionary of computed metrics
        """
        predictions, labels = eval_pred

        # If not computing final result, accumulate batch stats
        if not compute_result:
            # Accumulate batch-level statistics
            flat_predictions = predictions.flatten()
            flat_labels = labels.flatten()
            valid_mask = flat_labels != -100
            valid_predictions = flat_predictions[valid_mask]
            valid_labels = flat_labels[valid_mask]

            # Accumulate accuracy stats
            if len(valid_labels) > 0:
                correct = (valid_predictions == valid_labels).sum()
                batch_stats["total_correct"] += int(correct)
                batch_stats["total_samples"] += len(valid_labels)

            # Accumulate loss if available (for perplexity)
            if hasattr(eval_pred, "metrics"):
                batch_loss = eval_pred.metrics.get("eval_loss", 0)
                batch_stats["total_loss"] += batch_loss
                batch_stats["batch_count"] += 1

            # Return empty dict for batch accumulation
            return {}

        # Compute final results from accumulated statistics
        results = {}

        if batch_stats["total_samples"] > 0:
            accuracy = batch_stats["total_correct"] / batch_stats["total_samples"]
            results["accuracy"] = float(accuracy)
        else:
            results["accuracy"] = 0.0

        if batch_stats["batch_count"] > 0:
            avg_loss = batch_stats["total_loss"] / batch_stats["batch_count"]
            results["perplexity"] = float(2**avg_loss)
            results["loss"] = float(avg_loss)
        else:
            results["perplexity"] = 1.0
            results["loss"] = 0.0

        # Reset batch stats for next evaluation
        batch_stats.update(
            {
                "total_correct": 0,
                "total_samples": 0,
                "total_loss": 0.0,
                "batch_count": 0,
            }
        )

        return results

    return compute_metrics_batched
