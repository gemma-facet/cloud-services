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
        cleanup_temp: bool = True,
    ) -> str:
        """
        Merge the adapter with the base model and upload to Google Cloud Storage.

        Args:
            adapter_path: Path to the trained adapter
            base_model_id: ID of the base model (e.g., "google/gemma-2-9b")
            job_id: Job ID for tracking and database updates
            hf_token: Hugging Face token for accessing gated models
            cleanup_temp: Whether to clean up temporary files after upload

        Returns:
            str: GCS path to the merged model

        Raises:
            Exception: If merging or upload fails
        """
        try:
            self.logger.info(f"Starting model merge for job {job_id}")
            self.logger.info(f"Adapter path: {adapter_path}")
            self.logger.info(f"Base model: {base_model_id}")

            backend_provider = self._get_backend_provider(base_model_id)
            self.logger.info(f"Backend provider: {backend_provider}")

            # Login to Hugging Face if token provided
            if hf_token:
                from huggingface_hub import login

                login(token=hf_token)
                self.logger.info("Logged into Hugging Face")

            # Merge model based on provider
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

            # Upload to GCS and update Firestore
            self.logger.info("Uploading merged model to Google Cloud Storage...")
            gcs_merged_path = self._upload_merged_model_to_gcs(
                merged_model_path, job_id
            )

            # Clean up temp files if requested (skip for GGUF conversion)
            if cleanup_temp:
                self._cleanup_temp_directory(merged_model_path)
            else:
                self.logger.info("Keeping temp files for GGUF conversion")

            self.logger.info(f"Model merging completed. GCS path: {gcs_merged_path}")

            # Update Firestore with merged model path
            self._update_job_merged_path(job_id, gcs_merged_path)

            self.logger.info(f"Successfully merged and uploaded model for job {job_id}")
            return merged_model_path

        except Exception as e:
            self.logger.error(f"Failed to merge model for job {job_id}: {str(e)}")
            raise Exception(f"Model merging failed: {str(e)}")

    def _convert_to_gguf(
        self,
        merged_model_path: str,
        job_id: str,
        local_merged_path: Optional[str] = None,
    ) -> str:
        """
        Convert merged model to GGUF format and upload to Google Cloud Storage.

        Args:
            merged_model_path: GCS path to the merged model
            job_id: Job ID for tracking
            local_merged_path: Local path to merged model (if already available)

        Returns:
            str: GCS path to the GGUF model

        Raises:
            Exception: If conversion or upload fails
        """
        try:
            self.logger.info(f"Starting GGUF conversion for job {job_id}")
            self.logger.info(f"Merged model path: {merged_model_path}")

            # GGUF Conversion Flow:
            # 1. Get merged model (local if available, otherwise download from GCS)
            # 2. Convert to GGUF using llama.cpp
            # 3. Upload GGUF to GCS and update Firestore
            # 4. Clean up all temporary files

            if local_merged_path and os.path.exists(local_merged_path):
                self.logger.info(
                    f"Using existing local merged model: {local_merged_path}"
                )
                model_path_for_conversion = local_merged_path
            else:
                self.logger.info("Downloading merged model from GCS...")
                local_merged_path = f"/tmp/merged_model_{job_id}"
                self._download_merged_model_from_gcs(
                    merged_model_path, local_merged_path
                )
                model_path_for_conversion = local_merged_path

            # Convert to GGUF using llama.cpp
            gguf_file_path = self._run_llama_cpp_conversion(
                model_path_for_conversion, job_id
            )

            # Upload GGUF to GCS and update Firestore
            gcs_gguf_path = self._upload_gguf_to_gcs(gguf_file_path, job_id)
            self._update_job_gguf_path(job_id, gcs_gguf_path)

            # Clean up all temporary files
            self.logger.info("Cleaning up all temporary files...")
            if local_merged_path and local_merged_path.startswith("/tmp/"):
                self._cleanup_temp_directory(local_merged_path)
            if os.path.exists(gguf_file_path):
                os.remove(gguf_file_path)
                self.logger.info(f"Cleaned up temporary GGUF file: {gguf_file_path}")

            self.logger.info(f"Successfully converted to GGUF for job {job_id}")
            return gcs_gguf_path

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

    def _download_merged_model_from_gcs(
        self, gcs_merged_path: str, local_path: str
    ) -> None:
        """
        Download merged model from GCS to local directory.

        Args:
            gcs_merged_path: GCS path to the merged model
            local_path: Local path to download to
        """
        try:
            # Extract bucket and prefix from GCS path
            if gcs_merged_path.startswith("gs://"):
                path_parts = gcs_merged_path[5:].split("/", 1)
                bucket_name = path_parts[0]
                prefix = path_parts[1] if len(path_parts) > 1 else ""
            else:
                raise ValueError(f"Invalid GCS path: {gcs_merged_path}")

            bucket = self.storage_client.bucket(bucket_name)

            # Create local directory
            os.makedirs(local_path, exist_ok=True)

            # Download all blobs with the prefix
            blobs = bucket.list_blobs(prefix=prefix)
            for blob in blobs:
                relative_path = blob.name[len(prefix) :].lstrip("/")
                if relative_path:
                    local_file_path = os.path.join(local_path, relative_path)
                    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                    blob.download_to_filename(local_file_path)
                    self.logger.info(f"Downloaded {blob.name} to {local_file_path}")

            self.logger.info(f"Successfully downloaded merged model to {local_path}")

        except Exception as e:
            self.logger.error(f"Failed to download merged model from GCS: {str(e)}")
            raise Exception(f"GCS download failed: {str(e)}")

    def _run_llama_cpp_conversion(self, local_merged_path: str, job_id: str) -> str:
        """
        Run llama.cpp conversion to convert model to GGUF format.

        Args:
            local_merged_path: Local path to the merged model
            job_id: Job ID for naming the output file

        Returns:
            str: Path to the generated GGUF file
        """
        try:
            import subprocess

            # Create output file path
            output_file = f"/tmp/model_{job_id}.gguf"

            # Run llama.cpp convert command
            cmd = [
                "./llama.cpp/convert.py",
                local_merged_path,
                "--outfile",
                output_file,
                "--outtype",
                "q8_0",  # Default quantization
            ]

            self.logger.info(f"Running llama.cpp conversion: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=".",  # Run from current directory
                timeout=3600,  # 1 hour timeout
            )

            if result.returncode != 0:
                self.logger.error(f"llama.cpp conversion failed: {result.stderr}")
                raise Exception(f"llama.cpp conversion failed: {result.stderr}")

            if not os.path.exists(output_file):
                raise Exception("GGUF file was not created")

            self.logger.info(f"Successfully converted to GGUF: {output_file}")
            return output_file

        except subprocess.TimeoutExpired:
            self.logger.error("llama.cpp conversion timed out")
            raise Exception("llama.cpp conversion timed out")
        except Exception as e:
            self.logger.error(f"Failed to run llama.cpp conversion: {str(e)}")
            raise Exception(f"llama.cpp conversion failed: {str(e)}")

    def _upload_gguf_to_gcs(self, local_gguf_path: str, job_id: str) -> str:
        """
        Upload GGUF file to Google Cloud Storage.

        Args:
            local_gguf_path: Local path to the GGUF file
            job_id: Job ID for organizing the upload

        Returns:
            str: GCS path to the uploaded GGUF file
        """
        try:
            bucket = self.storage_client.bucket(self.gcs_export_bucket)
            gcs_blob_path = f"gguf_models/{job_id}/{os.path.basename(local_gguf_path)}"

            blob = bucket.blob(gcs_blob_path)
            blob.upload_from_filename(local_gguf_path)

            gcs_gguf_path = f"gs://{self.gcs_export_bucket}/{gcs_blob_path}"
            self.logger.info(f"Successfully uploaded GGUF to {gcs_gguf_path}")

            return gcs_gguf_path

        except Exception as e:
            self.logger.error(f"Failed to upload GGUF to GCS: {str(e)}")
            raise Exception(f"GCS upload failed: {str(e)}")

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
