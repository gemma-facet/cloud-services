import os
import logging
from typing import Optional
from google.cloud import firestore
from storage import gcs_storage


class ModelExportUtils:
    """
    Utility class for model export operations including merging and GGUF conversion.
    """

    def __init__(self):
        """Initialize the export utilities with Firestore client."""
        self.project_id = os.getenv("PROJECT_ID")
        if not self.project_id:
            raise ValueError("PROJECT_ID environment variable must be set")

        # Initialize Firestore client
        self.db = firestore.Client(project=self.project_id)

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

    def _update_export_path(self, job_id: str, path_type: str, path: str) -> None:
        """
        Update the export path or merged_path in Firestore for a specific job.

        Args:
            job_id: Job ID to update
            path_type: Type of path ('adapter_file', 'merged_file', 'gguf_file', 'merged_path')
            path: Path to the exported file/directory
        """
        try:
            doc_ref = self.db.collection("training_jobs").document(job_id)

            if path_type in ["adapter_file", "merged_file", "gguf_file"]:
                # Remove '_file' suffix to get the export type
                export_type = path_type.replace("_file", "")
                doc_ref.update({f"export.{export_type}": path})
                self.logger.info(
                    f"Updated export.{export_type} for job {job_id}: {path}"
                )
            elif path_type == "merged_path":
                doc_ref.update({"merged_path": path})
                self.logger.info(f"Updated merged_path for job {job_id}: {path}")
            else:
                raise ValueError(f"Invalid path_type: {path_type}")

        except Exception as e:
            self.logger.error(
                f"Failed to update {path_type} for job {job_id}: {str(e)}"
            )
            raise

    def _export_adapter(self, job_id: str, adapter_path: str) -> str:
        """
        Export adapter by downloading, zipping, and uploading to files bucket.

        Args:
            job_id: Job ID for the export
            adapter_path: GCS path to the adapter directory

        Returns:
            str: GCS path to the uploaded adapter zip file
        """
        local_adapter_path = None
        try:
            self.logger.info(f"Starting adapter export for job {job_id}")

            # Download adapter from GCS
            local_adapter_path = gcs_storage._download_directory(adapter_path)

            # Zip and upload to files bucket
            files_destination = f"gs://{gcs_storage.export_files_bucket}/{job_id}"
            gcs_zip_path = gcs_storage._zip_upload_file(
                local_adapter_path, files_destination, "adapter"
            )

            # Update Firestore
            self._update_export_path(job_id, "adapter_file", gcs_zip_path)

            self.logger.info(
                f"Successfully exported adapter for job {job_id}: {gcs_zip_path}"
            )
            return gcs_zip_path

        except Exception as e:
            self.logger.error(f"Failed to export adapter for job {job_id}: {str(e)}")
            raise Exception(f"Adapter export failed: {str(e)}")
        finally:
            # Clean up local files
            if local_adapter_path:
                gcs_storage._cleanup_local_directory(local_adapter_path)

    def _export_merged(
        self,
        job_id: str,
        merged_path: Optional[str] = None,
        adapter_path: Optional[str] = None,
        base_model_id: Optional[str] = None,
        hf_token: Optional[str] = None,
    ) -> str:
        """
        Export merged model by downloading existing or creating from adapter, then zipping and uploading.

        Args:
            job_id: Job ID for the export
            merged_path: Optional GCS path to existing merged model
            adapter_path: Optional GCS path to adapter (needed if merged_path not available)
            base_model_id: Optional base model ID (needed if creating merged model)
            hf_token: Optional Hugging Face token

        Returns:
            str: GCS path to the uploaded merged model zip file
        """
        local_merged_path = None
        try:
            self.logger.info(f"Starting merged model export for job {job_id}")

            if merged_path:
                # Download existing merged model
                self.logger.info(f"Downloading existing merged model: {merged_path}")
                local_merged_path = gcs_storage._download_directory(merged_path)
            else:
                # Create merged model from adapter
                if not adapter_path or not base_model_id:
                    raise Exception(
                        "adapter_path and base_model_id required to create merged model"
                    )

                self.logger.info("Creating merged model from adapter")
                # Download adapter first
                local_adapter_path = gcs_storage._download_directory(adapter_path)

                # Merge model locally
                local_merged_path = self._merge_model(
                    local_adapter_path,
                    base_model_id,
                    job_id,
                    hf_token,
                    cleanup_temp=False,
                )

                # Clean up downloaded adapter
                gcs_storage._cleanup_local_directory(local_adapter_path)

                # Upload merged model to export bucket
                export_destination = (
                    f"gs://{gcs_storage.export_bucket}/merged_models/{job_id}"
                )
                merged_gcs_path = gcs_storage._upload_directory(
                    local_merged_path, export_destination
                )

                # Update merged_path in Firestore
                self._update_export_path(job_id, "merged_path", merged_gcs_path)

            # Zip and upload to files bucket
            files_destination = f"gs://{gcs_storage.export_files_bucket}/{job_id}"
            gcs_zip_path = gcs_storage._zip_upload_file(
                local_merged_path, files_destination, "merged"
            )

            # Update Firestore
            self._update_export_path(job_id, "merged_file", gcs_zip_path)

            self.logger.info(
                f"Successfully exported merged model for job {job_id}: {gcs_zip_path}"
            )
            return gcs_zip_path

        except Exception as e:
            self.logger.error(
                f"Failed to export merged model for job {job_id}: {str(e)}"
            )
            raise Exception(f"Merged model export failed: {str(e)}")
        finally:
            # Clean up local files
            if local_merged_path:
                gcs_storage._cleanup_local_directory(local_merged_path)

    def _export_gguf(
        self,
        job_id: str,
        merged_path: Optional[str] = None,
        adapter_path: Optional[str] = None,
        base_model_id: Optional[str] = None,
        hf_token: Optional[str] = None,
    ) -> str:
        """
        Export GGUF model by converting from existing merged model or creating merged model first.

        Args:
            job_id: Job ID for the export
            merged_path: Optional GCS path to existing merged model
            adapter_path: Optional GCS path to adapter (needed if merged_path not available)
            base_model_id: Optional base model ID (needed if creating merged model)
            hf_token: Optional Hugging Face token

        Returns:
            str: GCS path to the uploaded GGUF file
        """
        local_merged_path = None
        try:
            self.logger.info(f"Starting GGUF export for job {job_id}")

            if merged_path:
                # Download existing merged model
                self.logger.info(f"Downloading existing merged model: {merged_path}")
                local_merged_path = gcs_storage._download_directory(merged_path)
            else:
                # Create merged model from adapter first
                if not adapter_path or not base_model_id:
                    raise Exception(
                        "adapter_path and base_model_id required to create merged model"
                    )

                self.logger.info(
                    "Creating merged model from adapter for GGUF conversion"
                )
                # Download adapter first
                local_adapter_path = gcs_storage._download_directory(adapter_path)

                # Merge model locally
                local_merged_path = self._merge_model(
                    local_adapter_path,
                    base_model_id,
                    job_id,
                    hf_token,
                    cleanup_temp=False,
                )

                # Clean up downloaded adapter
                gcs_storage._cleanup_local_directory(local_adapter_path)

                # Upload merged model to export bucket
                export_destination = (
                    f"gs://{gcs_storage.export_bucket}/merged_models/{job_id}"
                )
                merged_gcs_path = gcs_storage._upload_directory(
                    local_merged_path, export_destination
                )

                # Update merged_path in Firestore
                self._update_export_path(job_id, "merged_path", merged_gcs_path)

            # Convert to GGUF
            gguf_file_path = self._run_llama_cpp_conversion(local_merged_path, job_id)

            # Upload GGUF to files bucket
            files_destination = f"gs://{gcs_storage.export_files_bucket}/{job_id}"
            gcs_gguf_path = gcs_storage._upload_file(
                gguf_file_path, files_destination, "model"
            )

            # Update Firestore
            self._update_export_path(job_id, "gguf_file", gcs_gguf_path)

            self.logger.info(
                f"Successfully exported GGUF model for job {job_id}: {gcs_gguf_path}"
            )
            return gcs_gguf_path

        except Exception as e:
            self.logger.error(f"Failed to export GGUF model for job {job_id}: {str(e)}")
            raise Exception(f"GGUF export failed: {str(e)}")
        finally:
            # Clean up local files
            if local_merged_path:
                gcs_storage._cleanup_local_directory(local_merged_path)


# Create a global instance for easy access
export_utils = ModelExportUtils()
