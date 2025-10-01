import uuid
from datetime import datetime
import logging
from typing import Optional, Dict
from werkzeug.utils import secure_filename
from storage.base import StorageInterface
from schema import DatasetUploadResponse, PreprocessingConfig
from datasets import DatasetDict
import pandas as pd
import io
from .parsers import *
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
    - PDF files (.pdf)
    - Word documents (.docx)
    - PowerPoint presentations (.pptx)
    - HTML files (.html)
    
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
            "pdf",
            "docx",
            "pptx",
            "html",
        }
        self.parsers : dict[str, BaseParser] = {
            "pdf": PDFParser(),
            "docx": DOCXParser(),
            "pptx": PPTParser(),
            "html": HTMLParser(),
        }
        self.pandas_readers : dict[str, callable] = {
            "csv": pd.read_csv,
            "json": pd.read_json,
            "jsonl": lambda f: pd.read_json(f, lines=True),
            "xlsx": pd.read_excel,
            "xls": pd.read_excel,
            "parquet": pd.read_parquet,
            "txt": lambda f: pd.DataFrame(f.readlines(), columns=["text"]), 
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
            file_type = filename.rsplit(".", 1)[1].lower()
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

            sample = []
            num_examples = 0
        
            if file_type in self.parsers:
                parser = self.parsers[file_type]
                dataset = parser.parse(io.BytesIO(file_data))
                num_examples = len(dataset)
                sample_size = min(5, num_examples)
                if sample_size > 0:
                    sample = dataset.select(range(sample_size)).to_list()

            elif file_type in self.pandas_readers:
                reader = self.pandas_readers[file_type]
                df = reader(io.BytesIO(file_data))
                num_examples = len(df)
                if num_examples > 0:
                    sample = df.head(5).to_dict(orient="records")
            

            else:
                raise NotImplementedError(f"Parsing for file type '{file_type}' is not supported.")

            return DatasetUploadResponse(
                dataset_id=file_id,
                filename=secure_name,
                gcs_path=storage_path,
                size_bytes=len(file_data),
                sample=sample,
                columns=list(sample[0].keys()) if sample else [],
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
        modality: str,
    ) -> tuple[str, str, dict]:
        """
        Upload a processed dataset to storage with a unique identifier.

        Args:
            dataset (DatasetDict): The processed dataset splits to upload
            dataset_name (str): Display name for the dataset (stored as metadata)
            dataset_id (str): ID of the source dataset
            dataset_subset (str): Subset of the dataset
            config (PreprocessingConfig): Configuration used for processing
            dataset_source (str): Source of the dataset ("upload" or "huggingface")
            modality (str): Modality of the dataset ("text" or "vision")

        Returns:
            tuple[str, str, dict]: A tuple containing (dataset_path, processed_dataset_id, metadata)
                - dataset_path: Storage path for the processed dataset
                - processed_dataset_id: Unique identifier for the processed dataset
                - metadata: Dict containing dataset metadata for Firestore

        Raises:
            ValueError: If the dataset is empty, if the dataset is not in the correct format,
                       or if all splits are empty
        """
        if not dataset:
            raise ValueError("Dataset is empty")

        if not dataset_name:
            raise ValueError("Dataset name is required")

        # Generate unique ID for processed dataset (8-character UUID hex)
        processed_dataset_id = str(uuid.uuid4()).replace("-", "")[:8]

        metadata = {
            "splits": [],
            "dataset_name": dataset_name,
            "processed_dataset_id": processed_dataset_id,
            "upload_type": "processed_upload",
            "upload_date": datetime.now().isoformat(),
            "config": config.model_dump(),
            "dataset_id": dataset_id,
            "dataset_subset": dataset_subset,
            "dataset_source": dataset_source,
            "modality": modality,
            "created_at": datetime.now().isoformat(),
        }

        base_blob_name = f"processed_datasets/{processed_dataset_id}"

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

        # Return metadata for Firestore instead of creating metadata.json
        dataset_path = f"processed_datasets/{processed_dataset_id}/"

        return dataset_path, processed_dataset_id, metadata

    def does_dataset_exist(self, processed_dataset_id: str) -> bool:
        """
        Check if a processed dataset exists in storage.

        Args:
            processed_dataset_id (str): Unique identifier of the processed dataset to check

        Returns:
            bool: True if the dataset exists in processed_datasets/, False otherwise
        """
        dataset_path = f"processed_datasets/{processed_dataset_id}"
        exists = self.storage.file_exists(dataset_path)
        return exists
