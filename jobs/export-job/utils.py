import logging
from google.cloud import firestore
from schema import (
    ExportSchema,
    JobSchema,
    export_status,
    export_type,
    export_variant,
    ExportArtifact,
)
from typing import Optional
from storage import gcs_storage


class ExportUtils:
    """
    Utility class for export job operations.
    """

    def __init__(
        self,
        db: firestore.Client,
        export_id: str,
        project_id: str,
        hf_token: Optional[str] = None,
    ):
        """
        Initialize ExportUtils with database connection and export configuration.

        Args:
            db: Firestore database client instance
            export_id: Unique identifier for the export job
            project_id: Google Cloud project ID (must be set)
            hf_token: Optional Hugging Face token for model access

        Raises:
            ValueError: If project_id is not provided or empty
        """
        self.project_id = project_id
        if not self.project_id:
            raise ValueError("PROJECT_ID environment variable must be set")

        self.db = db
        self.export_id = export_id
        self.export_ref = self.db.collection("exports").document(export_id)
        self.export_doc = self.export_ref.get().to_dict()
        self.export_doc = ExportSchema(**self.export_doc)
        self.job_id = self.export_doc.job_id
        self.job_ref = self.db.collection("training_jobs").document(self.job_id)
        self.job_doc = self.job_ref.get().to_dict()
        self.job_doc = JobSchema(**self.job_doc)
        self.hf_token = hf_token

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
        local_adapter_path: str,
        base_model_id: str,
        job_id: str,
        hf_token: Optional[str] = None,
        cleanup_temp: bool = True,
    ) -> str:
        """
        Merge the adapter with the base model locally.

        Args:
            local_adapter_path: Local path to the downloaded adapter
            base_model_id: ID of the base model (e.g., "google/gemma-2-9b")
            job_id: Job ID for tracking
            hf_token: Hugging Face token for accessing gated models
            cleanup_temp: Whether to clean up temporary files after merging

        Returns:
            str: Local path to the merged model

        Raises:
            Exception: If merging fails
        """
        try:
            self.logger.info(f"Starting model merge for job {job_id}")
            self.logger.info(f"Local adapter path: {local_adapter_path}")
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

                logging.info("Loading adapter model from Unsloth")

                model, tokenizer = FastModel.from_pretrained(
                    model_name=local_adapter_path,
                    max_seq_length=2048,
                    load_in_4bit=True,
                )

                logging.info("Merging model with Unsloth")

                model.save_pretrained_merged(
                    "merged_model", tokenizer, save_method="merged_16bit"
                )

            elif backend_provider == "huggingface":
                from transformers import AutoModelForCausalLM
                from peft import PeftModel

                base_model = AutoModelForCausalLM.from_pretrained(base_model_id)
                model = PeftModel.from_pretrained(base_model, local_adapter_path)
                merged_model = model.merge_and_unload()
                merged_model.save_pretrained("merged_model", safe_serialization=True)

            merged_model_path = "merged_model"

            # Clean up temp files if requested
            if cleanup_temp:
                gcs_storage._cleanup_local_directory(merged_model_path)
            else:
                self.logger.info("Keeping temp files for further processing")

            self.logger.info(f"Successfully merged model for job {job_id}")
            return merged_model_path

        except Exception as e:
            self.logger.error(f"Failed to merge model for job {job_id}: {str(e)}")
            raise Exception(f"Model merging failed: {str(e)}")

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
            import os

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

    def _update_status(self, status: export_status, message: Optional[str] = None):
        """
        Update the export job status in Firestore.

        Args:
            status: The new status for the export job ("running", "completed", or "failed")
            message: Optional status message to provide additional context
        """
        self.export_ref.update(
            {
                "status": status,
                "message": message,
                "updated_at": firestore.SERVER_TIMESTAMP,
            }
        )

    def _update_export_artifacts(
        self, type: export_type, variant: export_variant, path: str
    ):
        """
        Add an artifact to the export job's artifacts list in Firestore.

        Args:
            type: Type of artifact ("adapter", "merged", or "gguf")
            variant: Variant of artifact ("raw" or "file")
            path: GCS path where the artifact is stored
        """
        artifact = ExportArtifact(type=type, variant=variant, path=path)
        self.export_ref.update({"artifacts": firestore.ArrayUnion([artifact])})

    def _update_job_artifacts(
        self, type: export_type, variant: export_variant, path: str
    ):
        """
        Update the training job's artifacts with the exported artifact path.

        Args:
            type: Type of artifact ("adapter", "merged", or "gguf")
            variant: Variant of artifact ("raw" or "file")
            path: GCS path where the artifact is stored
        """
        self.job_ref.update({f"artifacts.{variant}.{type}": path})

    def export_adapter(self):
        """
        Export the adapter model from the training job.

        Downloads the adapter from GCS, creates a zip file, uploads it to the export bucket,
        and updates both the export job and training job with the artifact information.

        Raises:
            ValueError: If adapter path is not found in the job document
        """
        self.logger.info(f"Exporting adapter for job {self.job_id}")
        self._update_status("running", "Preparing adapter export")

        if self.job_doc.artifacts.file.adapter:
            self.logger.info(f"Adapter already exported for job {self.job_id}")
            self._update_status("completed", "Adapter already present in the database.")
            return

        if not self.job_doc.adapter_path:
            raise ValueError("Adapter path not found in the database.")

        local_adapter_path = None
        try:
            adapter_path = self.job_doc.adapter_path
            local_adapter_path = gcs_storage._download_directory(adapter_path)

            files_destination = f"gs://{gcs_storage.export_files_bucket}/{self.job_id}"
            gcs_zip_path = gcs_storage._zip_upload_file(
                local_adapter_path, files_destination, "adapter"
            )

            self._update_export_artifacts("adapter", "file", gcs_zip_path)
            self._update_job_artifacts("adapter", "file", gcs_zip_path)
            self._update_status("completed", "Adapter exported successfully.")

        except Exception as e:
            self.logger.error(
                f"Failed to export adapter for job {self.job_id}: {str(e)}"
            )
            self._update_status("failed", f"Adapter export failed: {str(e)}")
            raise Exception(f"Adapter export failed: {str(e)}")
        finally:
            # Clean up local files
            if local_adapter_path:
                gcs_storage._cleanup_local_directory(local_adapter_path)

        return

    def export_merged(self):
        """
        Export the merged model from the training job.

        Downloads the merged model from GCS (or creates it from adapter if not available),
        creates a zip file, uploads it to the export bucket, and updates both the export job
        and training job with the artifact information.

        Raises:
            ValueError: If neither merged_path nor adapter_path/base_model_id are available
        """
        self.logger.info(f"Exporting merged model for job {self.job_id}")
        self._update_status("running", "Preparing merged model export")

        # Check if merged model already exported
        if self.job_doc.artifacts.file.merged:
            self.logger.info(f"Merged model already exported for job {self.job_id}")
            self._update_status(
                "completed", "Merged model already present in the database."
            )
            return

        local_merged_path = None
        try:
            # Check if merged model already exists in raw artifacts
            if self.job_doc.artifacts.raw.merged:
                # Download existing merged model
                self.logger.info(
                    f"Downloading existing merged model: {self.job_doc.artifacts.raw.merged}"
                )
                local_merged_path = gcs_storage._download_directory(
                    self.job_doc.artifacts.raw.merged
                )
            else:
                # Create merged model from adapter
                if not self.job_doc.adapter_path or not self.job_doc.base_model_id:
                    raise ValueError(
                        "adapter_path and base_model_id required to create merged model"
                    )

                self.logger.info("Creating merged model from adapter")
                # Download adapter first
                local_adapter_path = gcs_storage._download_directory(
                    self.job_doc.adapter_path
                )

                self._update_status("running", "Merging model with adapter")
                # Merge model locally using the old ModelExportUtils logic
                local_merged_path = self._merge_model(
                    local_adapter_path,
                    self.job_doc.base_model_id,
                    self.job_id,
                    self.hf_token,
                    cleanup_temp=False,
                )

                # Clean up downloaded adapter
                gcs_storage._cleanup_local_directory(local_adapter_path)

                # Upload merged model to export bucket for raw artifacts
                export_destination = (
                    f"gs://{gcs_storage.export_bucket}/merged_models/{self.job_id}"
                )
                merged_gcs_path = gcs_storage._upload_directory(
                    local_merged_path, export_destination
                )

                self._update_job_artifacts("merged", "raw", merged_gcs_path)
                self._update_export_artifacts("merged", "raw", merged_gcs_path)
                self.logger.info(f"Updated raw merged model path: {merged_gcs_path}")

            # Zip and upload to files bucket
            files_destination = f"gs://{gcs_storage.export_files_bucket}/{self.job_id}"
            gcs_zip_path = gcs_storage._zip_upload_file(
                local_merged_path, files_destination, "merged"
            )

            # Update artifacts
            self._update_export_artifacts("merged", "file", gcs_zip_path)
            self._update_job_artifacts("merged", "file", gcs_zip_path)
            self._update_status("completed", "Merged model exported successfully.")

        except Exception as e:
            self.logger.error(
                f"Failed to export merged model for job {self.job_id}: {str(e)}"
            )
            self._update_status("failed", f"Merged model export failed: {str(e)}")
            raise Exception(f"Merged model export failed: {str(e)}")
        finally:
            # Clean up local files
            if local_merged_path:
                gcs_storage._cleanup_local_directory(local_merged_path)

        return

    def export_gguf(self):
        """
        Export the GGUF model from the training job.

        Downloads the merged model from GCS (or creates it from adapter if not available),
        converts it to GGUF format using llama.cpp, uploads it to the export bucket,
        and updates both the export job and training job with the artifact information.

        Raises:
            ValueError: If neither merged_path nor adapter_path/base_model_id are available
        """
        self.logger.info(f"Exporting GGUF model for job {self.job_id}")
        self._update_status("running", "Preparing GGUF export")

        # Check if GGUF model already exported
        if self.job_doc.artifacts.file.gguf:
            self.logger.info(f"GGUF model already exported for job {self.job_id}")
            self._update_status(
                "completed", "GGUF model already present in the database."
            )
            return

        local_merged_path = None
        try:
            # Check if merged model already exists in raw artifacts
            if self.job_doc.artifacts.raw.merged:
                # Download existing merged model
                self.logger.info(
                    f"Downloading existing merged model: {self.job_doc.artifacts.raw.merged}"
                )
                local_merged_path = gcs_storage._download_directory(
                    self.job_doc.artifacts.raw.merged
                )
            else:
                # Create merged model from adapter first
                if not self.job_doc.adapter_path or not self.job_doc.base_model_id:
                    raise ValueError(
                        "adapter_path and base_model_id required to create merged model"
                    )

                self.logger.info(
                    "Creating merged model from adapter for GGUF conversion"
                )
                # Download adapter first
                local_adapter_path = gcs_storage._download_directory(
                    self.job_doc.adapter_path
                )

                self._update_status("running", "Merging model with adapter")
                # Merge model locally
                local_merged_path = self._merge_model(
                    local_adapter_path,
                    self.job_doc.base_model_id,
                    self.job_id,
                    self.hf_token,
                    cleanup_temp=False,
                )

                # Clean up downloaded adapter
                gcs_storage._cleanup_local_directory(local_adapter_path)

                # Upload merged model to export bucket for raw artifacts
                export_destination = (
                    f"gs://{gcs_storage.export_bucket}/merged_models/{self.job_id}"
                )
                merged_gcs_path = gcs_storage._upload_directory(
                    local_merged_path, export_destination
                )

                # Update raw artifacts
                self._update_job_artifacts("merged", "raw", merged_gcs_path)
                self._update_export_artifacts("merged", "raw", merged_gcs_path)
                self.logger.info(f"Updated raw merged model path: {merged_gcs_path}")

            # Convert to GGUF
            self._update_status("running", "Converting model to GGUF format")
            gguf_file_path = self._run_llama_cpp_conversion(
                local_merged_path, self.job_id
            )

            # Upload GGUF to files bucket
            files_destination = f"gs://{gcs_storage.export_files_bucket}/{self.job_id}"
            gcs_gguf_path = gcs_storage._upload_file(
                gguf_file_path, files_destination, "model"
            )

            # Update artifacts
            self._update_export_artifacts("gguf", "file", gcs_gguf_path)
            self._update_job_artifacts("gguf", "file", gcs_gguf_path)
            self._update_status("completed", "GGUF model exported successfully.")

        except Exception as e:
            self.logger.error(
                f"Failed to export GGUF model for job {self.job_id}: {str(e)}"
            )
            self._update_status("failed", f"GGUF model export failed: {str(e)}")
            raise Exception(f"GGUF model export failed: {str(e)}")
        finally:
            # Clean up local files
            if local_merged_path:
                gcs_storage._cleanup_local_directory(local_merged_path)

        return
