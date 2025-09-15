import os
import json
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from google.cloud import storage, firestore
from datasets import Dataset
import logging
from abc import ABC, abstractmethod
import shutil
import io
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetTracker:
    """
    Centralized dataset metadata management using Firestore.
    Tracks both raw uploads and processed datasets with user ownership.
    Uses simple dictionaries instead of complex dataclasses.
    """

    def __init__(self, project_id: str):
        """
        Initialize dataset tracker.

        Args:
            project_id: Google Cloud project ID
        """
        self.db = firestore.Client(project=project_id)
        self.processed_collection = self.db.collection("processed_datasets")
        self.logger = logging.getLogger(__name__)

    def get_processed_dataset_metadata(
        self, processed_dataset_id: str
    ) -> Optional[dict]:
        """
        Get processed dataset metadata by ID.

        Args:
            processed_dataset_id: Processed dataset unique ID

        Returns:
            Dict with metadata or None if not found
        """
        try:
            doc = self.processed_collection.document(processed_dataset_id).get()
            if not doc.exists:
                return None
            return doc.to_dict()
        except Exception as e:
            self.logger.error(
                f"Failed to get processed dataset metadata {processed_dataset_id}: {e}"
            )
            return None


class CloudStorageService:
    """
    Service for managing artifacts in Google Cloud Storage (GCS).
    This is used for **BOTH** dataset retrieval and model artifact storage.

    Handles uploading/downloading model adapters and processed datasets to/from GCS buckets.
    Provides a unified interface for cloud storage operations across the training pipeline.

    Args:
        data_bucket: GCS bucket name for storing datasets
        export_bucket: GCS bucket name for storing trained model artifacts
        dataset_tracker: DatasetTracker instance for metadata operations
    """

    def __init__(
        self,
        data_bucket: str,
        export_bucket: str,
        export_files_bucket: str,
        dataset_tracker: Optional[DatasetTracker] = None,
    ):
        self.data_bucket = data_bucket
        self.export_bucket = export_bucket
        self.export_files_bucket = export_files_bucket
        self.storage_client = storage.Client()
        self.dataset_tracker = dataset_tracker

    def upload_model(self, model_dir: str, job_id: str, export_format: str) -> str:
        """
        Upload model artifacts and metadata to GCS, return remote URI

        Uses different folder structures based on export format:
        - trained_adapters/{job_id}/ for adapter-only exports
        - merged_models/{job_id}/ for merged model exports
        - gguf_models/{job_id}/ for GGUF format exports

        Args:
            model_dir (str): Local directory containing model files
            job_id (str): Training job ID
            export_format (str): Format of the exported model

        Returns:
            str: GCS URI where the model artifacts are stored
        """
        try:
            bucket = self.storage_client.bucket(self.export_bucket)

            # Determine folder prefix based on export format
            format_prefix = self._get_format_prefix(export_format)
            prefix = f"{format_prefix}/{job_id}"

            # Upload all files in the model directory
            for root, dirs, files in os.walk(model_dir):
                for fn in files:
                    src = os.path.join(root, fn)
                    rel = os.path.relpath(src, model_dir)
                    blob = bucket.blob(f"{prefix}/{rel}")
                    blob.upload_from_filename(src)

            # This is the location for this export request
            return f"gs://{self.export_bucket}/{prefix}/"
        except Exception as e:
            logger.error(
                f"Error uploading model artifacts to GCS for job {job_id}: {e}",
                exc_info=True,
            )
            raise

    def upload_file(self, local_file_path: str, remote_file_path: str) -> str:
        """
        Upload a single file to GCS.

        Args:
            local_file_path (str): Local path to the file to upload
            remote_file_path (str): Remote path in GCS where the file will be stored
                This should include all the prefixes like gguf_models/file_id_here

        Returns:
            str: GCS URI where the file is stored
        """
        try:
            bucket = self.storage_client.bucket(self.export_files_bucket)

            # Upload the file
            blob = bucket.blob(remote_file_path)
            blob.upload_from_filename(local_file_path)

            return f"gs://{self.export_files_bucket}/{remote_file_path}"
        except Exception as e:
            logger.error(
                f"Error uploading file {local_file_path} to GCS: {e}",
                exc_info=True,
            )
            raise

    def download_model(self, path: str, local_dir: Optional[str] = None) -> str:
        """
        Download model artifacts from cloud storage into a local dir.

        Args:
            path (str): GCS path pointing to the model / adapter, with prefix merged_models or trained_adapters
            local_dir (Optional[str]): Local directory to download artifacts to.
                                       If None, uses a temporary directory.

        Returns:
            str: Local path where the model artifacts were downloaded
        """
        bucket = self.storage_client.bucket(self.export_bucket)
        # Extract prefix from path like "gs://bucket-name/prefix/job_id/"
        # Remove gs:// and bucket name, then remove trailing slash
        path_without_scheme = path.replace("gs://", "")
        path_parts = path_without_scheme.split("/")
        # Skip bucket name (first part) and get the rest as prefix
        prefix = (
            "/".join(path_parts[1:-1])
            if path_parts[-1] == ""
            else "/".join(path_parts[1:])
        )

        # Extract job_id from prefix (e.g., "trained_adapters/job_123" -> "job_123")
        job_id = prefix.split("/")[-1]

        # prepare local directory
        local_dir = local_dir or f"/tmp/inference_{job_id}"
        os.makedirs(local_dir, exist_ok=True)

        # download all artifacts
        for blob in bucket.list_blobs(prefix=prefix):
            rel = blob.name[len(prefix) + 1 :]
            dst = os.path.join(local_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            blob.download_to_filename(dst)

        return local_dir

    def download_processed_dataset(
        self, processed_dataset_id: str
    ) -> Tuple[Dataset, Optional[Dataset]]:
        """
        Download processed dataset files from GCS and return as HuggingFace Datasets.

        Retrieves training and optional evaluation datasets from the configured data bucket.
        The datasets are expected to be stored as Parquet files with metadata retrieved
        from Firestore.

        Args:
            processed_dataset_id (str): Identifier for the processed dataset (processed_dataset_id from preprocessing)

        Returns:
            Tuple[Dataset, Optional[Dataset]]: Train and eval datasets

        Raises:
            FileNotFoundError: If the training dataset is not found in GCS
            ValueError: If no dataset tracker is configured
        """
        if not self.dataset_tracker:
            raise ValueError("DatasetTracker is required but not configured")

        # Get metadata from Firestore instead of metadata.json
        metadata = self.dataset_tracker.get_processed_dataset_metadata(
            processed_dataset_id
        )
        if not metadata:
            raise FileNotFoundError(
                f"Dataset metadata not found in Firestore for {processed_dataset_id}"
            )

        splits = metadata.get("splits", [])

        if not splits:
            raise FileNotFoundError(
                f"No splits found in dataset {processed_dataset_id}"
            )

        bucket = self.storage_client.bucket(self.data_bucket)

        # Find train and test splits
        train_split = None
        eval_split = None

        for split_info in splits:
            split_name = split_info.get("split_name", "").lower()
            if split_name in ["train", "training"]:
                train_split = split_info
            elif split_name in ["test", "validation", "eval", "evaluation"]:
                eval_split = split_info

        # If no explicit train split found, use the first split as train
        if not train_split and splits:
            train_split = splits[0]
            logger.warning(
                f"No explicit train split found for {processed_dataset_id}, using {train_split.get('split_name')}"
            )

        if not train_split:
            raise FileNotFoundError(
                f"No train split found in dataset {processed_dataset_id}"
            )

        # Download train dataset
        train_blob = bucket.blob(
            f"processed_datasets/{processed_dataset_id}/{train_split['split_name']}.parquet"
        )
        if not train_blob.exists():
            raise FileNotFoundError(
                f"Train dataset file not found for {processed_dataset_id}"
            )

        # Download as bytes and load as parquet
        train_data_bytes = train_blob.download_as_bytes()

        train_table = pq.read_table(io.BytesIO(train_data_bytes))
        train_dataset = Dataset(train_table)

        # Download eval dataset if exists
        eval_dataset = None
        if eval_split:
            eval_blob = bucket.blob(
                f"processed_datasets/{processed_dataset_id}/{eval_split['split_name']}.parquet"
            )
            if eval_blob.exists():
                eval_data_bytes = eval_blob.download_as_bytes()
                eval_table = pq.read_table(io.BytesIO(eval_data_bytes))
                eval_dataset = Dataset(eval_table)
            else:
                logger.warning(
                    f"Eval dataset file not found for {processed_dataset_id}, using train only"
                )
        else:
            logger.warning(
                f"No eval split found for {processed_dataset_id}, using train only"
            )

        return train_dataset, eval_dataset

    def _get_format_prefix(self, export_format: Optional[str]) -> str:
        """
        Get the folder prefix based on export format.

        Returns:
            str: Folder prefix for storing the model artifacts

        NOTE: It makes sense to store full models with adapters because:
            1. they use the exact same code for saving and loading
            2. a full model will never have an adapter so no risk of confusion
            3. enables total reuse of logic everywhere -- except that full model cannot be merged!
        """
        format_mapping = {
            "adapter": "trained_adapters",
            "merged": "merged_models",
            "gguf": "gguf_models",
            "full": "trained_adapters",
        }
        return format_mapping.get(
            export_format, "adapters"
        )  # Default to adapters for backwards compatibility


@dataclass
class ModelArtifact:
    """Unified representation of model artifacts regardless of storage backend"""

    base_model_id: str
    job_id: str
    local_path: str
    remote_path: str
    provider: str = "huggingface"  # Training provider: "unsloth" or "huggingface"
    metadata: Optional[Dict[str, Any]] = None


class ModelStorageStrategy(ABC):
    """
    Abstract base strategy for model storage operations.

    Implements the Strategy pattern to support multiple storage backends (GCS, HuggingFace Hub, etc.).
    Each concrete strategy handles the specifics of saving, loading, and cleaning up model artifacts
    for its respective storage system.

    This abstraction allows the training pipeline to work with different storage backends
    without changing the core training logic.
    """

    @abstractmethod
    def save_model(
        self,
        local_path: str,
        job_id: str,
        base_model_id: str,
        export_format: str,
        provider: str = "gcs",
    ) -> ModelArtifact:
        """
        Upload model artifacts from local path to storage backend.

        Args:
            local_path: Local directory containing saved model files
            job_id: Training job ID
            base_model_id: Base model identifier
            export_format: Format of the exported model
            provider: Storage provider

        Returns:
            ModelArtifact: Artifact reference with paths and metadata
        """
        pass

    @abstractmethod
    def save_file(self, local_file_path: str, remote_path_or_repo_id: str) -> str:
        """
        Upload a single file to storage backend.

        Args:
            local_file_path: Local path to the file (e.g., GGUF file)
            remote_path_or_repo_id: remote path in the export bucket or hf repo id

        Returns:
            str: GCS URI or hugging face repo and file path to the uploaded file
        """
        pass

    @abstractmethod
    def load_model_info(self, artifact_id: str) -> ModelArtifact:
        """
        Load model metadata and prepare for inference.

        Retrieves model information from the storage backend without loading the actual
        model weights. This is used to get artifact metadata for inference setup.

        Args:
            artifact_id: Storage-specific identifier (job_id for GCS, repo_id for HF Hub)

        Returns:
            ModelArtifact: Artifact reference with metadata for model loading
        """
        pass

    @abstractmethod
    def cleanup(self, artifact: ModelArtifact) -> None:
        """Clean up local resources"""
        pass


class GCSStorageStrategy(ModelStorageStrategy):
    """
    Google Cloud Storage implementation of model storage strategy.

    Handles saving and loading model artifacts to/from GCS buckets. This strategy
    is used for persistent storage of trained adapters and supports both Unsloth
    and standard HuggingFace models.
    """

    def __init__(self):
        self.storage_service: CloudStorageService = storage_service

    def save_model(
        self,
        local_path: str,
        job_id: str,
        base_model_id: str,
        export_format: str,
        provider: str = "gcs",
    ) -> ModelArtifact:
        """
        Upload model artifacts from local path to GCS.

        Args:
            local_path: Local directory containing saved model files
            job_id: Training job ID
            base_model_id: Base model identifier
            export_format: Format of the exported model
            provider: Storage provider

        Returns:
            ModelArtifact: Reference to the uploaded model with GCS paths
        """
        # Upload to GCS (model is already saved locally by utils.py)
        remote_path = self.storage_service.upload_model(
            local_path, job_id, export_format
        )

        return ModelArtifact(
            base_model_id=base_model_id,
            job_id=job_id,
            local_path=local_path,
            remote_path=remote_path,
            provider=provider,
            metadata={
                "export_format": export_format,
            },
        )

    def save_file(self, local_file_path: str, remote_path_or_repo_id: str) -> str:
        """
        Upload a single file to GCS.

        Args:
            local_file_path: Local path to the file (e.g., GGUF file)
            remote_path_or_repo_id: Remote path in GCS where the file will be stored

        Returns:
            str: GCS URI where the file is stored
        """
        return self.storage_service.upload_file(local_file_path, remote_path_or_repo_id)

    def load_model_info(self, adapter_path: str) -> ModelArtifact:
        """Load model from GCS"""
        meta = self.storage_service.download_model(adapter_path)

        return ModelArtifact(
            base_model_id="",  # Not needed - inference client provides this via base_model_id parameter
            job_id=meta.job_id,
            local_path=meta.local_dir or "",
            remote_path=f"gs://{self.storage_service.export_bucket}/{meta.gcs_prefix}/",
            provider=meta.provider,
            metadata={"gcs_prefix": meta.gcs_prefix},
        )

    def cleanup(self, artifact: ModelArtifact) -> None:
        """Clean up local GCS artifacts"""
        if artifact.local_path and os.path.exists(artifact.local_path):
            shutil.rmtree(artifact.local_path, ignore_errors=True)


class HuggingFaceHubStrategy(ModelStorageStrategy):
    """
    HuggingFace Hub storage implementation of model storage strategy.

    Pushes trained models directly to HuggingFace Hub repositories for public or private
    sharing. This strategy is ideal for models that will be shared or need to be accessible
    via the HuggingFace ecosystem.

    Note: Requires proper HuggingFace authentication and repository permissions.
    """

    def __init__(self):
        pass

    def save_model(
        self,
        local_path: str,
        job_id: str,
        base_model_id: str,
        export_format: str,
        hf_repo_id: str,
        provider: str = "huggingface",
    ) -> ModelArtifact:
        """
        Upload all files in local directory to HuggingFace Hub repository.
        Much more efficient than loading and re-pushing models.
        """
        logging.info(f"Uploading folder {local_path} to HuggingFace Hub: {hf_repo_id}")

        try:
            api = HfApi()
            # Need to create the repo first then upload folder
            api.create_repo(
                repo_id=hf_repo_id,
                repo_type="model",
                private=True,  # Set to True for private repos, False for public
                exist_ok=True,  # Create if not exists
            )
            api.upload_folder(
                folder_path=local_path,
                repo_id=hf_repo_id,
                repo_type="model",
            )
        except Exception as e:
            logging.error(f"Failed to upload folder to HuggingFace Hub: {e}")
            raise

        return ModelArtifact(
            base_model_id=base_model_id,
            job_id=job_id,
            local_path=local_path,
            remote_path=hf_repo_id,
            provider=provider,
            metadata={"hf_repo_id": hf_repo_id},
        )

    def save_file(self, local_file_path: str, remote_path_or_repo_id: str) -> str:
        """
        Upload a single file to HuggingFace Hub repository.

        Args:
            local_file_path: Local path to the file (e.g., GGUF file)
            remote_file_path: HF repo id where this file will be stored

        Returns:
            str: Remote path in HuggingFace Hub where the file is stored
        """
        file_name = os.path.basename(local_file_path)
        logging.info(
            f"Uploading file {file_name} to HuggingFace Hub: {remote_path_or_repo_id}"
        )

        try:
            api = HfApi()
            # Create repo if it doesn't exist
            api.create_repo(
                repo_id=remote_path_or_repo_id,
                repo_type="model",
                private=True,
                exist_ok=True,
            )
            # Upload the single file
            api.upload_file(
                path_or_fileobj=local_file_path,
                path_in_repo=file_name,
                repo_id=remote_path_or_repo_id,
                repo_type="model",
            )
        except Exception as e:
            logging.error(f"Failed to upload file to HuggingFace Hub: {e}")
            raise

        return f"{remote_path_or_repo_id}/{file_name}"

    def load_model_info(self, repo_id: str) -> ModelArtifact:
        """
        Load model info from HuggingFace Hub.

        Attempts to retrieve model configuration from the HuggingFace repository
        to determine the base model and framework used. Falls back gracefully
        if configuration files are not available.

        Args:
            repo_id: HuggingFace repository identifier (e.g., "user/model-name")

        Returns:
            ModelArtifact: Model metadata for inference setup
        """
        try:
            # Try adapter config first
            adapter_config_path = hf_hub_download(
                repo_id=repo_id, filename="adapter_config.json"
            )
            with open(adapter_config_path, "r") as f:
                adapter_config = json.load(f)
            base_model_id = adapter_config.get("base_model_name_or_path", repo_id)
            provider = (
                "unsloth" if base_model_id.startswith("unsloth/") else "huggingface"
            )
        except Exception:
            logging.warning(
                f"Failed to load adapter config for {repo_id}, falling back to repo_id as model_id"
            )
            base_model_id = repo_id
            provider = "huggingface"

        return ModelArtifact(
            base_model_id=base_model_id,
            job_id=repo_id,  # Use repo_id as job_id for HF Hub
            local_path="",  # No local path for HF Hub
            remote_path=repo_id,
            provider=provider,
            metadata={"hf_repo_id": repo_id},
        )

    def cleanup(self, artifact: ModelArtifact) -> None:
        """
        No local artifacts to clean up for HuggingFace Hub.
        This method is required as abstract method but does nothing
        """
        pass  # No local artifacts to clean up for HF Hub


class StorageStrategyFactory:
    """
    Factory for creating storage strategies.

    Implements the Factory pattern to instantiate the appropriate storage strategy
    based on the export type specified in training requests. This allows the training
    pipeline to work with multiple storage backends without tight coupling.

    Example:
    ```python
    storage_strategy = StorageStrategyFactory.create_strategy("gcs", storage_service=storage_service)
    artifact = storage_strategy.save_model(model, tokenizer, local_path, metadata)
    ```

    Supported storage types:
    - "gcs": Google Cloud Storage (requires storage_service parameter)
    - "hfhub": HuggingFace Hub (no additional parameters required)
    """

    @staticmethod
    def create_strategy(storage_type: str, **kwargs) -> ModelStorageStrategy:
        """
        Create appropriate storage strategy based on type.

        Args:
            storage_type: Type of storage ("gcs" or "hfhub")
            **kwargs: Additional parameters required by specific strategies
                     - For "gcs": storage_service (CloudStorageService instance)
                     - For "hfhub": no additional parameters required

        Returns:
            ModelStorageStrategy: Configured storage strategy instance

        Raises:
            ValueError: If storage_type is not supported
        """
        if storage_type == "gcs":
            return GCSStorageStrategy()
        elif storage_type == "hfhub":
            return HuggingFaceHubStrategy()
        else:
            raise ValueError(f"Unknown storage type: {storage_type}")


# default model storage service instance
data_bucket = os.environ.get("GCS_DATA_BUCKET_NAME", "gemma-facet-datasets")
export_bucket = os.environ.get("GCS_EXPORT_BUCKET_NAME", "gemma-facet-models")
export_files_bucket = os.environ.get(
    "GCS_EXPORT_FILES_BUCKET_NAME", "gemma-facet-files"
)
project_id = os.environ.get("PROJECT_ID")

# Initialize dataset tracker if project_id is available
dataset_tracker = DatasetTracker(project_id) if project_id else None

storage_service = CloudStorageService(
    data_bucket, export_bucket, export_files_bucket, dataset_tracker
)
