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
from typing import Any, Optional, Literal, Tuple, Union, TYPE_CHECKING
from storage import gcs_storage


if TYPE_CHECKING:
    from unsloth import FastModel
    from transformers import PreTrainedModel


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


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

    def _get_backend_provider(
        self, base_model_id: str
    ) -> Literal["unsloth", "huggingface"]:
        """
        Determine the backend provider based on the base model ID.

        Args:
            base_model_id: The base model identifier

        Returns:
            Literal["unsloth", "huggingface"]: Backend provider name
        """
        if base_model_id.startswith("unsloth/"):
            return "unsloth"
        else:
            return "huggingface"

    # NOTE: The third return value is the tokenizer. However, we don't know the type of the tokenizer, so kept it as Any.
    # We don't need the tokenizer for the hf merged model, so we can keep it as None.
    def _merge_model(
        self,
        local_adapter_path: str,
        base_model_id: str,
        job_id: str,
        hf_token: Optional[str] = None,
        cleanup_temp: bool = True,
    ) -> Tuple[str, Union["FastModel", "PreTrainedModel"], Union[Any, None]]:
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
            logger.info(f"Starting model merge for job {job_id}")
            logger.info(f"Local adapter path: {local_adapter_path}")
            logger.info(f"Base model: {base_model_id}")

            backend_provider = self._get_backend_provider(base_model_id)
            logger.info(f"Backend provider: {backend_provider}")

            # Login to Hugging Face if token provided
            if hf_token:
                from huggingface_hub import login

                login(token=hf_token)
                logger.info("Logged into Hugging Face")

            return_model = None
            return_tokenizer = None

            # Merge model based on provider
            if backend_provider == "unsloth":
                from unsloth import FastModel

                logger.info("Loading adapter model from Unsloth")

                model, tokenizer = FastModel.from_pretrained(
                    model_name=local_adapter_path,
                    max_seq_length=2048,
                    load_in_4bit=True,
                )

                logger.info("Merging model with Unsloth")

                model.save_pretrained_merged(
                    "merged_model", tokenizer, save_method="merged_16bit"
                )
                return_model = model
                return_tokenizer = tokenizer

            elif backend_provider == "huggingface":
                from transformers import AutoModelForCausalLM
                from peft import PeftModel

                base_model = AutoModelForCausalLM.from_pretrained(base_model_id)
                model = PeftModel.from_pretrained(base_model, local_adapter_path)
                merged_model = model.merge_and_unload()
                merged_model.save_pretrained("merged_model", safe_serialization=True)
                return_model = merged_model
                return_tokenizer = None

            merged_model_path = "merged_model"

            # Clean up temp files if requested
            if cleanup_temp:
                gcs_storage._cleanup_local_directory(merged_model_path)
            else:
                logger.info("Keeping temp files for further processing")

            logger.info(f"Successfully merged model for job {job_id}")
            return merged_model_path, return_model, return_tokenizer

        except Exception as e:
            logger.error(f"Failed to merge model for job {job_id}: {str(e)}")
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
                "/app/llama_venv/bin/python",
                "./llama.cpp/convert_hf_to_gguf.py",
                local_merged_path,
                "--outfile",
                output_file,
                "--outtype",
                "q8_0",  # Default quantization
            ]

            logger.info(f"Running llama.cpp conversion: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=".",  # Run from current directory
                timeout=3600,  # 1 hour timeout
            )

            if result.returncode != 0:
                logger.error(f"llama.cpp conversion failed: {result.stderr}")
                raise Exception(f"llama.cpp conversion failed: {result.stderr}")

            if not os.path.exists(output_file):
                raise Exception("GGUF file was not created")

            logger.info(f"Successfully converted to GGUF: {output_file}")
            return output_file

        except subprocess.TimeoutExpired:
            logger.error("llama.cpp conversion timed out")
            raise Exception("llama.cpp conversion timed out")
        except Exception as e:
            logger.error(f"Failed to run llama.cpp conversion: {str(e)}")
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
            variant: Variant of artifact ("raw" or "file" or "hf")
            path: GCS path where the artifact is stored
        """
        artifact = ExportArtifact(type=type, variant=variant, path=path)
        self.export_ref.update(
            {"artifacts": firestore.ArrayUnion([artifact.model_dump()])}
        )

    def _update_job_artifacts(
        self, type: export_type, variant: export_variant, path: str
    ):
        """
        Update the training job's artifacts with the exported artifact path.

        Args:
            type: Type of artifact ("adapter", "merged", or "gguf")
            variant: Variant of artifact ("raw" or "file" or "hf")
            path: GCS path where the artifact is stored
        """
        self.job_ref.update({f"artifacts.{variant}.{type}": path})

    def _push_adapter_to_hf_hub(self, local_adapter_path: str):
        """
        Push the adapter to HF Hub.

        Args:
            local_adapter_path: Local path to the adapter

        Raises:
            Exception: If pushing to HF Hub fails
        """
        try:
            logger.info(
                f"Starting adapter push to HF Hub: {self.export_doc.hf_repo_id}"
            )
            logger.info(f"Local adapter path: {local_adapter_path}")

            backend_provider = self._get_backend_provider(self.job_doc.base_model_id)
            logger.info(f"Backend provider: {backend_provider}")

            if backend_provider == "unsloth":
                logger.info("Loading adapter model from Unsloth for HF Hub push")
                from unsloth import FastModel

                model, tokenizer = FastModel.from_pretrained(
                    model_name=local_adapter_path,
                    max_seq_length=2048,
                    load_in_4bit=True,
                )

                logger.info("Pushing Unsloth model to HF Hub")
                model.push_to_hub(
                    self.export_doc.hf_repo_id,
                    safe_serialization=True,
                    token=self.hf_token,
                )
                logger.info("Pushing Unsloth tokenizer to HF Hub")
                tokenizer.push_to_hub(self.export_doc.hf_repo_id, token=self.hf_token)

            elif backend_provider == "huggingface":
                logger.info("Loading adapter model from Hugging Face for HF Hub push")
                from peft import PeftModel
                from transformers import AutoTokenizer

                logger.info(f"Loading base model: {self.job_doc.base_model_id}")
                model = PeftModel.from_pretrained(
                    self.job_doc.base_model_id, local_adapter_path
                )
                logger.info("Loading tokenizer for base model")
                tokenizer = AutoTokenizer.from_pretrained(self.job_doc.base_model_id)

                logger.info("Pushing Hugging Face model to HF Hub")
                model.push_to_hub(
                    self.export_doc.hf_repo_id,
                    safe_serialization=True,
                    token=self.hf_token,
                )
                logger.info("Pushing Hugging Face tokenizer to HF Hub")
                tokenizer.push_to_hub(self.export_doc.hf_repo_id, token=self.hf_token)

            logger.info("Updating export artifacts with HF Hub path")
            self._update_export_artifacts("adapter", "hf", self.export_doc.hf_repo_id)
            logger.info("Updating job artifacts with HF Hub path")
            self._update_job_artifacts("adapter", "hf", self.export_doc.hf_repo_id)

            logger.info(
                f"Successfully pushed adapter to HF Hub: {self.export_doc.hf_repo_id}"
            )

        except Exception as e:
            logger.error(f"Failed to push adapter to HF Hub: {str(e)}")
            logger.error(f"Adapter path: {local_adapter_path}")
            logger.error(f"HF repo ID: {self.export_doc.hf_repo_id}")
            raise Exception(f"Failed to push adapter to HF Hub: {str(e)}")

    def _push_merged_to_hf_hub(
        self, model: Union["FastModel", "PreTrainedModel"], tokenizer: Union[Any, None]
    ):
        """
        Push the merged model to HF Hub.

        Args:
            model: The merged model
            tokenizer: The tokenizer

        Raises:
            Exception: If pushing to HF Hub fails
        """
        try:
            logger.info(
                f"Starting merged model push to HF Hub: {self.export_doc.hf_repo_id}"
            )
            logger.info(f"Model type: {type(model)}")
            logger.info(f"Tokenizer type: {type(tokenizer)}")

            backend_provider = self._get_backend_provider(self.job_doc.base_model_id)
            logger.info(f"Backend provider: {backend_provider}")

            if backend_provider == "unsloth":
                logger.info("Pushing Unsloth merged model to HF Hub")
                if tokenizer is None:
                    logger.warning("Tokenizer is None for Unsloth model push")
                model.push_to_hub_merged(
                    self.export_doc.hf_repo_id, tokenizer, token=self.hf_token
                )
                logger.info("Successfully pushed Unsloth merged model to HF Hub")

            elif backend_provider == "huggingface":
                logger.info("Pushing Hugging Face merged model to HF Hub")
                model.push_to_hub(
                    self.export_doc.hf_repo_id,
                    safe_serialization=True,
                    token=self.hf_token,
                )
                logger.info("Successfully pushed Hugging Face merged model to HF Hub")

            logger.info("Updating export artifacts with HF Hub path")
            self._update_export_artifacts("merged", "hf", self.export_doc.hf_repo_id)
            logger.info("Updating job artifacts with HF Hub path")
            self._update_job_artifacts("merged", "hf", self.export_doc.hf_repo_id)

            logger.info(
                f"Successfully pushed merged model to HF Hub: {self.export_doc.hf_repo_id}"
            )

        except Exception as e:
            logger.error(f"Failed to push merged model to HF Hub: {str(e)}")
            logger.error(f"Model type: {type(model)}")
            logger.error(f"Tokenizer type: {type(tokenizer)}")
            logger.error(f"HF repo ID: {self.export_doc.hf_repo_id}")
            logger.error(f"Backend provider: {backend_provider}")
            raise Exception(f"Failed to push merged model to HF Hub: {str(e)}")

    def _push_gguf_to_hf_hub(self, gguf_file_path: str):
        """
        Push the GGUF file to HF Hub.

        Args:
            gguf_file_path: Path to the GGUF file

        Raises:
            Exception: If pushing to HF Hub fails
        """
        try:
            logger.info(
                f"Starting GGUF file push to HF Hub: {self.export_doc.hf_repo_id}"
            )
            logger.info(f"GGUF file path: {gguf_file_path}")

            # Import required modules
            from huggingface_hub import HfApi, login
            import os

            # Verify GGUF file exists
            if not os.path.exists(gguf_file_path):
                raise FileNotFoundError(f"GGUF file not found: {gguf_file_path}")

            file_size = os.path.getsize(gguf_file_path)
            logger.info(f"GGUF file size: {file_size / (1024 * 1024):.2f} MB")

            # Login to Hugging Face
            logger.info("Logging into Hugging Face")
            login(token=self.hf_token)
            logger.info("Successfully logged into Hugging Face")

            # Initialize HF API
            logger.info("Initializing Hugging Face API")
            api = HfApi()

            # Create repository
            logger.info(f"Creating repository: {self.export_doc.hf_repo_id}")
            api.create_repo(repo_id=self.export_doc.hf_repo_id, exist_ok=True)
            logger.info("Repository created or already exists")

            # Upload GGUF file
            logger.info("Uploading GGUF file to HF Hub")
            api.upload_file(
                path_or_fileobj=gguf_file_path,
                path_in_repo="model.gguf",
                repo_id=self.export_doc.hf_repo_id,
                repo_type="model",
            )
            logger.info("Successfully uploaded GGUF file to HF Hub")

            # Update artifacts
            self._update_export_artifacts("gguf", "hf", self.export_doc.hf_repo_id)
            self._update_job_artifacts("gguf", "hf", self.export_doc.hf_repo_id)

            logger.info(
                f"Successfully pushed GGUF file to HF Hub: {self.export_doc.hf_repo_id}"
            )

        except FileNotFoundError as e:
            logger.error(f"GGUF file not found: {str(e)}")
            logger.error(f"Expected file path: {gguf_file_path}")
            raise Exception(f"GGUF file not found: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to push GGUF file to HF Hub: {str(e)}")
            logger.error(f"GGUF file path: {gguf_file_path}")
            logger.error(f"HF repo ID: {self.export_doc.hf_repo_id}")
            raise Exception(f"Failed to push GGUF file to HF Hub: {str(e)}")

    def export_adapter(self):
        """
        Export the adapter model from the training job.

        Downloads the adapter from GCS, creates a zip file, uploads it to the export bucket,
        and updates both the export job and training job with the artifact information.

        Raises:
            ValueError: If adapter path is not found in the job document
        """
        logger.info(f"Exporting adapter for job {self.job_id}")
        self._update_status("running", "Preparing adapter export")

        if self.job_doc.artifacts.file.adapter:
            logger.info(f"Adapter already exported for job {self.job_id}")
            self._update_status("completed", "Adapter already present in the database.")
            return

        if not self.job_doc.adapter_path:
            raise ValueError("Adapter path not found in the database.")

        local_adapter_path = None
        try:
            adapter_path = self.job_doc.adapter_path
            local_adapter_path = gcs_storage._download_directory(adapter_path)

            if "hf_hub" in self.export_doc.destination:
                self._push_adapter_to_hf_hub(local_adapter_path)

            if "gcs" in self.export_doc.destination:
                files_destination = (
                    f"gs://{gcs_storage.export_files_bucket}/{self.job_id}"
                )
                gcs_zip_path = gcs_storage._zip_upload_file(
                    local_adapter_path, files_destination, "adapter"
                )

                self._update_export_artifacts("adapter", "file", gcs_zip_path)
                self._update_job_artifacts("adapter", "file", gcs_zip_path)

            self._update_status("completed", "Adapter exported successfully.")

        except Exception as e:
            logger.error(f"Failed to export adapter for job {self.job_id}: {str(e)}")
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
        logger.info(f"Exporting merged model for job {self.job_id}")
        self._update_status("running", "Preparing merged model export")

        # Check if merged model already exported
        if self.job_doc.artifacts.file.merged:
            logger.info(f"Merged model already exported for job {self.job_id}")
            self._update_status(
                "completed", "Merged model already present in the database."
            )
            return

        local_merged_path = None
        try:
            # Check if merged model already exists in raw artifacts
            if self.job_doc.artifacts.raw.merged:
                # Download existing merged model
                logger.info(
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

                logger.info("Creating merged model from adapter")
                # Download adapter first
                local_adapter_path = gcs_storage._download_directory(
                    self.job_doc.adapter_path
                )

                self._update_status("running", "Merging model with adapter")
                local_merged_path, model, tokenizer = self._merge_model(
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
                logger.info(f"Updated raw merged model path: {merged_gcs_path}")

            if "hf_hub" in self.export_doc.destination:
                self._push_merged_to_hf_hub(model, tokenizer)

            if "gcs" in self.export_doc.destination:
                # Zip and upload to files bucket
                files_destination = (
                    f"gs://{gcs_storage.export_files_bucket}/{self.job_id}"
                )
                gcs_zip_path = gcs_storage._zip_upload_file(
                    local_merged_path, files_destination, "merged"
                )

                # Update artifacts
                self._update_export_artifacts("merged", "file", gcs_zip_path)
                self._update_job_artifacts("merged", "file", gcs_zip_path)

            self._update_status("completed", "Merged model exported successfully.")

        except Exception as e:
            logger.error(
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
        logger.info(f"Exporting GGUF model for job {self.job_id}")
        self._update_status("running", "Preparing GGUF export")

        # Check if GGUF model already exported
        if self.job_doc.artifacts.file.gguf:
            logger.info(f"GGUF model already exported for job {self.job_id}")
            self._update_status(
                "completed", "GGUF model already present in the database."
            )
            return

        local_merged_path = None
        try:
            # Check if merged model already exists in raw artifacts
            if self.job_doc.artifacts.raw.merged:
                # Download existing merged model
                logger.info(
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

                logger.info("Creating merged model from adapter for GGUF conversion")
                # Download adapter first
                local_adapter_path = gcs_storage._download_directory(
                    self.job_doc.adapter_path
                )

                self._update_status("running", "Merging model with adapter")
                # Merge model locally
                local_merged_path, _, _ = self._merge_model(
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
                logger.info(f"Updated raw merged model path: {merged_gcs_path}")

            # Convert to GGUF
            self._update_status("running", "Converting model to GGUF format")
            gguf_file_path = self._run_llama_cpp_conversion(
                local_merged_path, self.job_id
            )

            if "hf_hub" in self.export_doc.destination:
                self._push_gguf_to_hf_hub(gguf_file_path)

            if "gcs" in self.export_doc.destination:
                # Upload GGUF to files bucket
                files_destination = (
                    f"gs://{gcs_storage.export_files_bucket}/{self.job_id}"
                )
                gcs_gguf_path = gcs_storage._upload_file(
                    gguf_file_path, files_destination, "model"
                )

                # Update artifacts
                self._update_export_artifacts("gguf", "file", gcs_gguf_path)
                self._update_job_artifacts("gguf", "file", gcs_gguf_path)

            self._update_status("completed", "GGUF model exported successfully.")

        except Exception as e:
            logger.error(f"Failed to export GGUF model for job {self.job_id}: {str(e)}")
            self._update_status("failed", f"GGUF model export failed: {str(e)}")
            raise Exception(f"GGUF model export failed: {str(e)}")
        finally:
            # Clean up local files
            if local_merged_path:
                gcs_storage._cleanup_local_directory(local_merged_path)

        return
