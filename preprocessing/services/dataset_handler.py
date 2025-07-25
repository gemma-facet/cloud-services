import uuid
import json
from datetime import datetime
import logging
from typing import Optional, Dict
from werkzeug.utils import secure_filename
from storage.base import StorageInterface
from schema import DatasetUploadResponse, PreprocessingConfig
from datasets import DatasetDict
import pandas as pd
import io

logger = logging.getLogger(__name__)


class DatasetHandler:
    """
    A class that handles dataset file uploads and storage operations.

    This class provides functionality to upload dataset files to storage, with support
    for various file formats and metadata handling. It ensures secure file handling
    and generates unique identifiers for uploaded datasets.

    The uploader supports the following file formats:
    - Text files (.txt)
    - CSV files (.csv)
    - JSON files (.json)
    - JSONL files (.jsonl)
    - Excel files (.xlsx, .xls)
    - Parquet files (.parquet)

    Attributes:
        storage (StorageInterface): An interface for storage operations
        supported_formats (Set[str]): Set of supported file extensions

    Example:
        >>> storage = StorageInterface()
        >>> handler = DatasetHandler(storage)
        >>> response = handler.upload_dataset(file_data, "dataset.csv")
    """

    def __init__(self, storage: StorageInterface):
        """
        Initialize the DatasetHandler with a storage interface.

        Args:
            storage (StorageInterface): An interface for storage operations that provides
                methods for storing and retrieving files.
        """
        self.storage = storage
        self.supported_formats = {
            "txt",
            "csv",
            "json",
            "jsonl",
            "xlsx",
            "xls",
            "parquet",
        }

    def _is_allowed_file(self, filename: str) -> bool:
        """
        Check if the file extension is allowed for upload.

        This method verifies if the file's extension is in the list of supported formats.
        The check is case-insensitive.

        Args:
            filename (str): The name of the file to check

        Returns:
            bool: True if the file extension is supported, False otherwise

        Example:
            >>> is_allowed = handler._is_allowed_file("dataset.csv")
            >>> print(is_allowed)
            True
        """
        return (
            "." in filename
            and filename.rsplit(".", 1)[1].lower() in self.supported_formats
        )

    def upload_dataset(
        self, file_data: bytes, filename: str, metadata: Optional[Dict] = None
    ) -> DatasetUploadResponse:
        """
        Upload a dataset file to storage.

        This method handles the complete upload process:
        1. Validates the file format
        2. Generates a unique identifier
        3. Secures the filename
        4. Adds metadata
        5. Uploads to storage
        6. Returns upload response

        Args:
            file_data (bytes): The raw file data to upload
            filename (str): The original name of the file
            metadata (Optional[Dict]): Additional metadata to store with the file.
                Common metadata includes:
                - original_filename: Original name of the file
                - file_id: Unique identifier for the file
                - upload_type: Type of upload (e.g., 'user_upload')
                - custom metadata fields

        Returns:
            DatasetUploadResponse: An object containing:
                - dataset_id (str): Unique identifier for the uploaded dataset
                - filename (str): Secure version of the filename
                - gcs_path (str): Path where the file is stored
                - size_bytes (int): Size of the uploaded file in bytes

        Raises:
            ValueError: If no file is selected or if the file format is not supported
            Exception: For other errors during the upload process

        Example:
            >>> with open("dataset.csv", "rb") as f:
            ...     file_data = f.read()
            >>> response = handler.upload_dataset(
            ...     file_data,
            ...     "dataset.csv",
            ...     metadata={"description": "My dataset"}
            ... )
            >>> print(response.dataset_id)
            '123e4567-e89b-12d3-a456-426614174000'
        """
        try:
            if not filename:
                raise ValueError("No file selected")

            if not self._is_allowed_file(filename):
                raise ValueError(
                    f"File type not allowed. Supported formats: {self.supported_formats}"
                )

            file_id = str(uuid.uuid4())
            secure_name = secure_filename(filename)
            blob_name = f"raw_datasets/{file_id}_{secure_name}"

            upload_metadata = {
                "original_filename": filename,
                "file_id": file_id,
                "upload_type": "user_upload",
                **(metadata or {}),
            }

            storage_path = self.storage.upload_data(
                file_data, blob_name, upload_metadata
            )

            sample = (
                pd.read_csv(io.BytesIO(file_data)).head(5).to_dict(orient="records")
            )

            num_examples = len(pd.read_csv(io.BytesIO(file_data)))

            print(sample)

            return DatasetUploadResponse(
                dataset_id=file_id,
                filename=secure_name,
                gcs_path=storage_path,
                size_bytes=len(file_data),
                sample=sample,
                columns=list(sample[0].keys()),
                num_examples=num_examples,
            )

        except Exception as e:
            logger.error(f"Error uploading dataset: {str(e)}")
            raise

    def upload_processed_dataset(
        self,
        dataset: DatasetDict,
        dataset_name: str,
        dataset_id: str,
        dataset_subset: str,
        config: PreprocessingConfig,
        dataset_source: str,
    ) -> str:
        """
        Upload a processed dataset to the storage.
        The directory structure is as follows:
        - processed_datasets/
            - dataset_name/
                - split_name.parquet (all splits are saved in the same directory)
                - metadata.json (contains the metadata of the dataset)

        Args:
            dataset (DatasetDict): The processed dataset to upload
            dataset_id (str): The unique identifier for the dataset

        Returns:
            str: The file path of the uploaded dataset

        Raises:
            ValueError: If the dataset is empty, if the dataset is not in the correct format,
                       or if all splits are empty
        """
        if not dataset:
            raise ValueError("Dataset is empty")

        if not dataset_name:
            raise ValueError("Dataset name is required")

        # Automatically determine modality: 'vision' if any image field mappings, else 'text'
        modality = (
            "vision"
            if any(fm.type == "image" for fm in config.field_mappings.values())
            else "text"
        )
        metadata = {
            "splits": [],
            "dataset_name": dataset_name,
            "upload_type": "processed_upload",
            "upload_date": datetime.now().isoformat(),
            "config": config.model_dump(),
            "dataset_id": dataset_id,
            "dataset_subset": dataset_subset,
            "dataset_source": dataset_source,
            "modality": modality,
            "created_at": datetime.now().isoformat(),
        }

        base_blob_name = f"processed_datasets/{dataset_name}"

        for split_name, split_dataset in dataset.items():
            if len(split_dataset) == 0:
                logger.warning(f"Split {split_name} is empty, skipping...")
                continue

            buf = io.BytesIO()
            split_dataset.to_parquet(buf)
            blob_name = f"{base_blob_name}/{split_name}.parquet"
            split_path = self.storage.upload_data(buf.getvalue(), blob_name)
            metadata["splits"].append(
                {
                    "split_name": split_name,
                    "num_rows": split_dataset.num_rows,
                    "path": split_path,
                }
            )

        # Check if all splits were empty
        if not metadata["splits"]:
            raise ValueError("Cannot upload dataset: all splits are empty")

        metadata_path = self.storage.upload_data(
            json.dumps(metadata), f"{base_blob_name}/metadata.json"
        )

        dataset_path = metadata_path.replace("metadata.json", "")

        return dataset_path

    def does_dataset_exist(self, dataset_name: str) -> bool:
        """
        Check if a processed dataset exists in storage.

        Args:
            dataset_name (str): Name of the dataset to check

        Returns:
            bool: True if the dataset exists in processed_datasets/, False otherwise
        """
        dataset_path = f"processed_datasets/{dataset_name}"
        exists = self.storage.file_exists(dataset_path)
        return exists
