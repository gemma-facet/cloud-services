import os
import logging
from typing import Optional
from google.cloud import storage, firestore


class ModelExportUtils:
    """
    Utility class for model export operations including merging and GGUF conversion.
    """

    def __init__(self):
        """Initialize the export utilities with Google Cloud and Firestore clients."""
        self.project_id = os.getenv("PROJECT_ID")
        if not self.project_id:
            raise ValueError("PROJECT_ID environment variable must be set")

        # Initialize Google Cloud clients
        self.storage_client = storage.Client()
        self.db = firestore.Client(project=self.project_id)

        # Environment variables for storage
        self.gcs_export_bucket = os.getenv(
            "GCS_EXPORT_BUCKET_NAME", "gemma-export-bucket"
        )

        self.logger = logging.getLogger(__name__)

    def _get_backend_provider(self, base_model_id: str) -> str:
        """
        Determine the backend provider based on the base model ID.

        Args:
            base_model_id: The base model identifier

        Returns:
            str: Backend provider name ("unsloth" or "huggingface")
        """
        if base_model_id.startswith("unsloth/"):
            return "unsloth"
        else:
            return "huggingface"

    def _merge_model(
        self,
        adapter_path: str,
        base_model_id: str,
        job_id: str,
        hf_token: Optional[str] = None,
    ) -> str:
        """
        Merge the adapter with the base model and upload to Google Cloud Storage.

        Args:
            adapter_path: Path to the trained adapter
            base_model_id: ID of the base model (e.g., "google/gemma-2-9b")
            job_id: Job ID for tracking and database updates
            hf_token: Hugging Face token for accessing gated models

        Returns:
            str: GCS path to the merged model

        Raises:
            Exception: If merging or upload fails
        """
        try:
            self.logger.info(f"Starting model merge for job {job_id}")
            self.logger.info(f"Adapter path: {adapter_path}")
            self.logger.info(f"Base model: {base_model_id}")

            # Determine backend provider from base_model_id
            backend_provider = self._get_backend_provider(base_model_id)
            self.logger.info(f"Backend provider: {backend_provider}")

            # Login to Hugging Face if token provided
            if hf_token:
                from huggingface_hub import login

                login(token=hf_token)
                self.logger.info("Logged into Hugging Face")

            if backend_provider == "unsloth":
                from unsloth import FastModel

                model, tokenizer = FastModel.from_pretrained(
                    model_name=adapter_path,
                    max_seq_length=2048,
                    load_in_4bit=True,
                )

                model.save_pretrained_merged(
                    "merged_model", tokenizer, save_method="merged_16bit"
                )

            elif backend_provider == "huggingface":
                from transformers import AutoModelForCausalLM
                from peft import PeftModel

                base_model = AutoModelForCausalLM.from_pretrained(base_model_id)
                model = PeftModel.from_pretrained(base_model, adapter_path)
                merged_model = model.merge_and_unload()
                merged_model.save_pretrained("merged_model", safe_serialization=True)

            merged_model_path = "merged_model"

            # Upload merged model to GCS
            self.logger.info("Uploading merged model to Google Cloud Storage...")
            gcs_merged_path = self._upload_merged_model_to_gcs(
                merged_model_path, job_id
            )

            # Clean up temporary merged model directory
            self.logger.info("Cleaning up temporary merged model directory...")
            self._cleanup_temp_directory(merged_model_path)

            self.logger.info(f"Model merging completed. GCS path: {gcs_merged_path}")

            # Update Firestore with merged model path
            self._update_job_merged_path(job_id, gcs_merged_path)

            self.logger.info(f"Successfully merged and uploaded model for job {job_id}")
            return merged_model_path

        except Exception as e:
            self.logger.error(f"Failed to merge model for job {job_id}: {str(e)}")
            raise Exception(f"Model merging failed: {str(e)}")

    def _convert_to_gguf(self, merged_model_path: str, job_id: str) -> str:
        """
        Convert merged model to GGUF format and upload to Google Cloud Storage.

        Args:
            merged_model_path: GCS path to the merged model
            job_id: Job ID for tracking

        Returns:
            str: GCS path to the GGUF model

        Raises:
            Exception: If conversion or upload fails
        """
        try:
            self.logger.info(f"Starting GGUF conversion for job {job_id}")
            self.logger.info(f"Merged model path: {merged_model_path}")

            # TODO: Implement actual GGUF conversion logic
            # This will involve:
            # 1. Downloading the merged model from GCS
            # 2. Converting to GGUF format using appropriate tools
            # 3. Uploading back to GCS

            # For now, we'll create a placeholder GCS path
            gguf_path = f"gs://{self.gcs_export_bucket}/gguf_models/{job_id}/model.gguf"

            # TODO: Replace with actual conversion and upload implementation
            self.logger.info(
                f"GGUF conversion completed. Placeholder path: {gguf_path}"
            )

            # Update Firestore with GGUF path
            self._update_job_gguf_path(job_id, gguf_path)

            self.logger.info(f"Successfully converted to GGUF for job {job_id}")
            return gguf_path

        except Exception as e:
            self.logger.error(f"Failed to convert to GGUF for job {job_id}: {str(e)}")
            raise Exception(f"GGUF conversion failed: {str(e)}")

    def _update_job_merged_path(self, job_id: str, merged_path: str) -> None:
        """
        Update the merged_path field in Firestore for a specific job.

        Args:
            job_id: Job ID to update
            merged_path: Path to the merged model
        """
        try:
            doc_ref = self.db.collection("training_jobs").document(job_id)
            doc_ref.update({"merged_path": merged_path})
            self.logger.info(f"Updated merged_path for job {job_id}: {merged_path}")
        except Exception as e:
            self.logger.error(
                f"Failed to update merged_path for job {job_id}: {str(e)}"
            )
            raise

    def _update_job_gguf_path(self, job_id: str, gguf_path: str) -> None:
        """
        Update the gguf_path field in Firestore for a specific job.

        Args:
            job_id: Job ID to update
            gguf_path: Path to the GGUF model in GCS
        """
        try:
            doc_ref = self.db.collection("training_jobs").document(job_id)
            doc_ref.update({"gguf_path": gguf_path})
            self.logger.info(f"Updated gguf_path for job {job_id}: {gguf_path}")
        except Exception as e:
            self.logger.error(f"Failed to update gguf_path for job {job_id}: {str(e)}")
            raise

    def _upload_merged_model_to_gcs(self, local_model_path: str, job_id: str) -> str:
        """
        Upload the merged model directory to Google Cloud Storage.

        Args:
            local_model_path: Local path to the merged model directory
            job_id: Job ID for organizing the upload

        Returns:
            str: GCS path to the uploaded merged model
        """
        try:
            bucket = self.storage_client.bucket(self.gcs_export_bucket)
            gcs_prefix = f"merged_models/{job_id}"

            self.logger.info(
                f"Uploading merged model from {local_model_path} to gs://{self.gcs_export_bucket}/{gcs_prefix}"
            )

            # Upload all files in the merged model directory
            for root, dirs, files in os.walk(local_model_path):
                for file in files:
                    local_file_path = os.path.join(root, file)
                    # Calculate relative path from the merged model directory
                    relative_path = os.path.relpath(local_file_path, local_model_path)
                    gcs_blob_path = f"{gcs_prefix}/{relative_path}"

                    blob = bucket.blob(gcs_blob_path)
                    blob.upload_from_filename(local_file_path)
                    self.logger.info(
                        f"Uploaded {local_file_path} to gs://{self.gcs_export_bucket}/{gcs_blob_path}"
                    )

            gcs_merged_path = f"gs://{self.gcs_export_bucket}/{gcs_prefix}"
            self.logger.info(f"Successfully uploaded merged model to {gcs_merged_path}")
            return gcs_merged_path

        except Exception as e:
            self.logger.error(f"Failed to upload merged model to GCS: {str(e)}")
            raise Exception(f"GCS upload failed: {str(e)}")

    def _cleanup_temp_directory(self, directory_path: str) -> None:
        """
        Clean up a temporary directory and all its contents.

        Args:
            directory_path: Path to the directory to clean up
        """
        try:
            if os.path.exists(directory_path):
                import shutil

                shutil.rmtree(directory_path)
                self.logger.info(f"Cleaned up temporary directory: {directory_path}")
            else:
                self.logger.warning(
                    f"Directory does not exist for cleanup: {directory_path}"
                )
        except Exception as e:
            self.logger.warning(
                f"Failed to clean up directory {directory_path}: {str(e)}"
            )

    def cleanup_temp_files(self, temp_paths: list) -> None:
        """
        Clean up temporary files created during export operations.

        Args:
            temp_paths: List of temporary file paths to clean up
        """
        try:
            for path in temp_paths:
                if os.path.exists(path):
                    os.remove(path)
                    self.logger.info(f"Cleaned up temporary file: {path}")
        except Exception as e:
            self.logger.warning(f"Failed to clean up some temporary files: {str(e)}")


# Create a global instance for easy access
export_utils = ModelExportUtils()
