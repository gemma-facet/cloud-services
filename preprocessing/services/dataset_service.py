import io
import logging
import pyarrow.parquet as pq
import base64
from datasets import Dataset, DatasetDict
from typing import Dict, Literal, Optional
from PIL import Image
from storage.base import StorageInterface
from .dataset_handler import DatasetHandler
from .dataset_loader import DatasetLoader
from .format_converter import FormatConverter
from augmentation import run_augment_pipeline
from schema import (
    DatasetUploadResponse,
    ProcessingResult,
    PreprocessingConfig,
    DatasetsInfoResponse,
    DatasetInfoSample,
    DatasetInfoResponse,
)

logger = logging.getLogger(__name__)


class DatasetService:
    """
    A class that orchestrates all dataset operations and provides a unified interface
    for dataset management and processing.

    This service combines functionality from multiple components:
    - DatasetHandler: Handles file uploads
    - DatasetLoader: Loads datasets from various sources
    - FormatConverter: Converts datasets to ChatML format

    The service provides high-level operations for:
    - Uploading datasets
    - Processing datasets to ChatML format
    - Managing processed datasets

    Attributes:
        storage (StorageInterface): Interface for storage operations
        uploader (DatasetHandler): Handles dataset uploads
        loader (DatasetLoader): Handles dataset loading
        converter (FormatConverter): Handles format conversion

    Example:
        >>> storage = StorageInterface()
        >>> service = DatasetService(storage)
        >>> response = service.upload_dataset(file_data, "dataset.csv")
    """

    def __init__(self, storage: StorageInterface):
        """
        Initialize the DatasetService with required components.

        Args:
            storage (StorageInterface): An interface for storage operations that provides
                methods for storing and retrieving files.
        """
        self.storage = storage
        self.handler = DatasetHandler(storage)
        self.loader = DatasetLoader(storage)
        self.converter = FormatConverter()

    def upload_dataset(
        self, file_data: bytes, filename: str, metadata: Optional[Dict] = None
    ) -> DatasetUploadResponse:
        """
        Upload a dataset file to storage.

        This method handles the complete upload process, including:
        - File validation
        - Secure filename generation
        - Metadata addition
        - Storage upload
        - Response generation

        Args:
            file_data (bytes): The raw file data to upload
            filename (str): The original name of the file
            metadata (Optional[Dict]): Additional metadata to store with the file.
                Common metadata includes:
                - description: Dataset description
                - source: Data source information
                - tags: Dataset tags
                - custom fields

        Returns:
            DatasetUploadResponse: An object containing:
                - dataset_id (str): Unique identifier for the uploaded dataset
                - filename (str): Secure version of the filename
                - gcs_path (str): Path where the file is stored
                - size_bytes (int): Size of the uploaded file in bytes

        Raises:
            ValueError: If the file is invalid or format is not supported
            Exception: For other errors during upload

        Example:
            >>> with open("dataset.csv", "rb") as f:
            ...     file_data = f.read()
            >>> response = service.upload_dataset(
            ...     file_data,
            ...     "dataset.csv",
            ...     metadata={"description": "My dataset"}
            ... )
        """
        return self.handler.upload_dataset(file_data, filename, metadata)

    def process_dataset(
        self,
        dataset_name: str,
        dataset_source: Literal["upload", "huggingface"],
        dataset_id: str,
        processing_mode: str,  # Changed from PreprocessingConfig
        config: PreprocessingConfig,
        dataset_subset: str = "default",
    ) -> ProcessingResult:
        """
        Process a dataset to a conversational format based on the processing mode.

        This method performs the complete dataset processing pipeline:
        1. Loads the dataset from the specified source
        2. Converts it to a conversational format based on the mode
        3. Applies data augmentation if configured
        4. Validates the converted dataset
        5. Saves the processed dataset to storage
        6. Returns processing results

        Args:
            dataset_name (str): The name of the dataset, used for the processed dataset name
            dataset_source (str): The source of the dataset
            dataset_id (str): The identifier for the dataset
            processing_mode (str): The target format mode (e.g., 'language_modeling', 'grpo').
            config (PreprocessingConfig): Configuration for processing, including field mappings.
            dataset_subset (str): The subset of the dataset to use (for Hugging Face datasets).

        Returns:
            ProcessingResult: An object containing processing results and metadata

        Raises:
            Exception: If there's an error during processing
        """
        try:
            # Load dataset with splits
            dataset = self.loader.load_dataset(
                dataset_source, dataset_id, config, dataset_subset
            )

            if not dataset:
                raise ValueError("Dataset is empty or could not be loaded")

            config_dict = config.model_dump()
            # the format converter will detect modality and return
            processed_dataset, modality = (
                self.converter.convert_to_conversational_chatml(
                    dataset, processing_mode, config_dict
                )
            )

            if not processed_dataset:
                raise ValueError("No samples could be converted to ChatML format")

            # Apply data augmentation if the user created a config specification
            augmentation_config = config_dict.get("augmentation_config", {})
            if augmentation_config:
                processed_dataset = self._augment_all_splits(
                    processed_dataset, augmentation_config
                )

            # Save all splits and get the unique processed dataset ID
            dataset_path, processed_dataset_id, metadata = (
                self.handler.upload_processed_dataset(
                    processed_dataset,
                    dataset_name,
                    dataset_id,
                    dataset_subset,
                    config,
                    dataset_source,
                    modality,
                )
            )

            # Extract just the split names for ProcessingResult
            split_names = [split["split_name"] for split in metadata["splits"]]

            # Create the ProcessingResult with an additional metadata field
            result = ProcessingResult(
                dataset_name=dataset_name,
                dataset_subset=dataset_subset,
                dataset_source=dataset_source,
                dataset_id=dataset_id,  # Keep original source dataset ID
                processed_dataset_id=processed_dataset_id,  # Add unique processed dataset ID
                num_examples=sum(split["num_rows"] for split in metadata["splits"]),
                created_at=metadata["created_at"],
                splits=split_names,
                modality=metadata["modality"],
                full_splits=metadata["splits"],
            )

            del dataset
            del processed_dataset
            return result

        except Exception as e:
            import traceback

            logger.error(f"Error processing dataset: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

    def get_datasets_info(self, user_id: str, dataset_tracker) -> DatasetsInfoResponse:
        """
        Get information about all processed datasets owned by a specific user.

        Args:
            user_id: User ID to filter datasets
            dataset_tracker: DatasetTracker instance to get metadata from Firestore

        Returns:
            DatasetsInfoResponse: List of datasets owned by the user
        """
        try:
            if dataset_tracker is None:
                raise ValueError(
                    "dataset_tracker is required for Firestore metadata lookup"
                )

            # Get all processed datasets owned by the user directly from Firestore
            user_dataset_ids = dataset_tracker.get_user_processed_datasets(user_id)

            datasets_info = []
            for processed_dataset_id in user_dataset_ids:
                metadata = dataset_tracker.get_processed_dataset_metadata(
                    processed_dataset_id
                )
                if metadata:
                    # Convert metadata to DatasetInfoSample format
                    dataset_info = DatasetInfoSample(
                        dataset_name=metadata["dataset_name"],
                        dataset_subset=metadata["dataset_subset"],
                        dataset_source=metadata["dataset_source"],
                        dataset_id=metadata["dataset_id"],
                        processed_dataset_id=metadata.get(
                            "processed_dataset_id", "dataset_name"
                        ),
                        num_examples=metadata["num_examples"],
                        created_at=metadata["created_at"],
                        splits=[split["split_name"] for split in metadata["splits"]],
                        modality=metadata.get("modality", "text"),
                    )
                    datasets_info.append(dataset_info)

            return DatasetsInfoResponse(datasets=datasets_info)

        except Exception as e:
            logger.error(f"Error getting datasets info for user {user_id}: {str(e)}")
            raise

    def get_dataset_info(
        self, processed_dataset_id: str, dataset_tracker=None
    ) -> DatasetInfoResponse:
        """
        Get information about a dataset including samples from each split.
        For vision datasets, PIL Images (bytes) are automatically converted to base64 data URLs for API compatibility.

        Args:
            processed_dataset_id: Unique identifier for the processed dataset
            dataset_tracker: DatasetTracker instance to get metadata from Firestore
        """
        try:
            # Get metadata from Firestore instead of metadata.json
            if dataset_tracker is None:
                raise ValueError(
                    "dataset_tracker is required for Firestore metadata lookup"
                )

            metadata_obj = dataset_tracker.get_processed_dataset_metadata(
                processed_dataset_id
            )
            if not metadata_obj:
                raise FileNotFoundError(
                    f"Dataset metadata not found: {processed_dataset_id}"
                )

            # metadata_obj is now a dict, not a dataclass
            metadata = {
                "dataset_name": metadata_obj["dataset_name"],
                "dataset_subset": metadata_obj["dataset_subset"],
                "dataset_source": metadata_obj["dataset_source"],
                "dataset_id": metadata_obj["dataset_id"],
                "upload_date": metadata_obj["created_at"],
                "modality": metadata_obj["modality"],
                "splits": metadata_obj["splits"],
            }

            # Get splits information with samples
            splits_with_samples = []
            for split_info in metadata.get("splits", []):
                split_name = split_info.get("split_name")
                split_path = (
                    f"processed_datasets/{processed_dataset_id}/{split_name}.parquet"
                )

                # Get samples from the split
                samples = []
                try:
                    if self.storage.file_exists(split_path):
                        split_data = self.storage.download_binary_data(split_path)
                        table = pq.read_table(io.BytesIO(split_data))
                        samples = table.slice(0, 5).to_pylist()

                        # For vision datasets, convert images to base64 for API response
                        modality = metadata.get("modality", "text")
                        samples = [
                            self._prepare_sample_for_api(s, modality) for s in samples
                        ]
                        # Filter out any samples that became None during conversion
                        samples = [s for s in samples if s is not None]
                        if modality == "vision":
                            logger.info(
                                "Converted vision samples to base64 for API response"
                            )

                except Exception as e:
                    logger.warning(
                        f"Could not read samples from split {split_name}: {str(e)}"
                    )

                splits_with_samples.append(
                    {
                        "split_name": split_name,
                        "num_rows": split_info.get("num_rows", 0),
                        "path": split_info.get("path", ""),
                        "samples": samples,
                    }
                )

            return DatasetInfoResponse(
                dataset_name=metadata.get("dataset_name"),
                dataset_subset=metadata.get("dataset_subset"),
                dataset_source=metadata.get("dataset_source"),
                dataset_id=metadata.get("dataset_id"),
                processed_dataset_id=processed_dataset_id,
                created_at=metadata.get("upload_date"),
                splits=splits_with_samples,
                modality=metadata.get("modality", "text"),
            )

        except Exception as e:
            logger.error(f"Error getting dataset info: {str(e)}")
            raise

    def _augment_all_splits(
        self, dataset: DatasetDict, augmentation_config: Dict
    ) -> DatasetDict:
        """
        Apply augmentation to a single split or list of data.

        Args:
            dataset (DatasetDict): The dataset to augment
            augmentation_config (Dict): Configuration for the augmentation pipeline

        Returns:
            DatasetDict: The augmented dataset, each split is a Dataset
        """
        try:
            for split in dataset.keys():
                # Apply augmentation pipeline
                augmented_dataset, result = run_augment_pipeline(
                    dataset[split].to_list(), augmentation_config
                )

                # Log results
                if result.errors:
                    for error in result.errors:
                        logger.warning(f"Augmentation error: {error}")

                dataset[split] = Dataset.from_list(augmented_dataset)

            return dataset

        except Exception as e:
            logger.error(f"Split augmentation failed: {e}")
            return dataset

    def _prepare_sample_for_api(self, sample: dict, modality: str) -> dict:
        """
        Prepares a single data sample for API output. If it's a vision sample,
        converts all image representations into base64 data URLs.
        """
        if modality != "vision" or not isinstance(sample, dict):
            return sample

        # New format is the standard, check for it first.
        if "messages" in sample and "images" in sample:
            return self._format_new_vision_sample(sample)

        # Fallback for old formats
        return self._format_old_vision_sample(sample)

    def _format_new_vision_sample(self, sample: dict) -> dict:
        """Formats a new-style vision sample by inlining base64 images."""
        import copy

        base64_images = [
            self._convert_image_to_base64_url(img) for img in sample.get("images", [])
        ]
        base64_images = [b64 for b64 in base64_images if b64]

        # Deep copy to avoid modifying the original sample
        api_sample = {"messages": copy.deepcopy(sample.get("messages", []))}

        image_idx = 0
        for message in api_sample["messages"]:
            if not isinstance(message.get("content"), list):
                continue

            new_content = []
            for content_item in message["content"]:
                if (
                    isinstance(content_item, dict)
                    and content_item.get("type") == "image"
                ):
                    if image_idx < len(base64_images):
                        new_content.append(
                            {"type": "image", "image": base64_images[image_idx]}
                        )
                        image_idx += 1
                else:
                    new_content.append(content_item)
            message["content"] = new_content

        return api_sample

    def _format_old_vision_sample(self, data: any) -> any:
        """Recursively formats an old-style vision sample."""
        if isinstance(data, dict):
            # If the dict itself is an image, convert it.
            b64_url = self._convert_image_to_base64_url(data)
            if b64_url:
                return b64_url

            # Otherwise, recurse on its values.
            return {k: self._format_old_vision_sample(v) for k, v in data.items()}

        if isinstance(data, list):
            return [self._format_old_vision_sample(item) for item in data]

        # If the item is a standalone image (e.g. in a list), convert it.
        b64_url = self._convert_image_to_base64_url(data)
        return b64_url if b64_url else data

    def _convert_image_to_base64_url(self, image_data: any) -> Optional[str]:
        """
        Converts a supported image representation (PIL, HF dict) into a
        base64 data URL. Returns None if the format is not recognized or fails.
        """
        pil_image = None
        if isinstance(image_data, Image.Image):
            pil_image = image_data
        elif isinstance(image_data, dict) and "bytes" in image_data:
            try:
                image_bytes = image_data.get("bytes")
                if isinstance(image_bytes, (bytes, bytearray)):
                    pil_image = Image.open(io.BytesIO(image_bytes))
            except Exception as e:
                logger.warning(f"Failed to create PIL Image from bytes: {e}")

        if not pil_image:
            return None

        try:
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:image/png;base64,{encoded}"
        except Exception as e:
            logger.warning(f"Failed to encode PIL Image to base64: {e}")
            return None
