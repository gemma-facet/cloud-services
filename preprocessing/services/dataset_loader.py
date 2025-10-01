import logging
from datasets import load_dataset, DatasetDict, IterableDataset, Dataset
from storage.base import StorageInterface
from schema import (
    PreprocessingConfig,
    ManualSplitConfig,
    NoSplitConfig,
    HFSplitConfig,
)
import io
from .parsers import *

logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    A class that handles loading datasets from uploaded files and Hugging Face datasets.

    This class provides functionality to load datasets from different sources and formats,
    supporting various file types like CSV, JSON, JSONL, Excel, and Parquet files.

    Attributes:
        storage (StorageInterface): An interface for storage operations, used to access
            uploaded files and manage dataset storage.

    Example:
        >>> storage = StorageInterface()
        >>> loader = DatasetLoader(storage)
        >>> dataset = await loader.load_dataset("upload", "my_dataset_id")
    """

    def __init__(self, storage: StorageInterface):
        """
        Initialize the DatasetLoader with a storage interface.

        Args:
            storage (StorageInterface): An interface for storage operations that provides
                methods for accessing and managing stored datasets.
        """
        self.storage = storage
        self.parsers : dict[str, BaseParser] = {
            "pdf": PDFParser(),
            "docx": DOCXParser(),
            "pptx": PPTParser(),
            "html": HTMLParser(),
        }

    def load_dataset(
        self,
        dataset_source: str,
        dataset_id: str,
        config: PreprocessingConfig,
        dataset_subset: str = "default",
    ) -> DatasetDict:
        """
        Load a dataset from the specified source.

        This method supports loading datasets from three different sources:
        1. 'upload': Loads a user-uploaded dataset from storage
        2. 'huggingface': Loads a dataset from Hugging Face's dataset hub

        Args:
            dataset_source (str): The source of the dataset. Must be one of:
                - 'upload': For user-uploaded datasets
                - 'huggingface': For datasets from Hugging Face
            dataset_id (str): The identifier for the dataset:
                - For 'upload': The file ID of the uploaded dataset
                - For 'huggingface': The Hugging Face dataset name
            config (PreprocessingConfig): Configuration for processing, including:
                - field_mappings: Mapping of fields to use for the dataset
                - normalize_whitespace: Whether to normalize whitespace in the dataset
                - augmentation_config: Configuration for augmentation
                - split_config: Configuration for splitting the dataset
            dataset_subset (str): The subset of the dataset to load

        Returns:
            DatasetDict: A dictionary with split names as keys and dataset samples as values.

        Raises:
            ValueError: If an invalid dataset_source is provided.
            FileNotFoundError: If the uploaded dataset is not found.
            Exception: For other errors during dataset loading.

        Example:
            >>> dataset = await loader.load_dataset("huggingface", "squad", dataset_splits=DatasetSplits(train="train", test="test"), config=PreprocessingConfig(split_config=ManualSplitConfig(sample_size=100, test_size=0.2)))
        """
        if dataset_source == "upload" and isinstance(
            config.split_config, HFSplitConfig
        ):
            raise ValueError(
                "HuggingFace split configuration (hf_split) cannot be used with uploaded datasets. Use 'manual_split' or 'no_split' instead."
            )

        try:
            if dataset_source == "upload":
                return self._load_uploaded_dataset(dataset_id, config)
            elif dataset_source == "huggingface":
                return self._load_huggingface_dataset(
                    dataset_id, dataset_subset, config
                )
            else:
                raise ValueError(f"Invalid dataset_source: {dataset_source}")

        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise

    def _load_uploaded_dataset(
        self, dataset_id: str, config: PreprocessingConfig
    ) -> DatasetDict:
        """
        Load a dataset that was previously uploaded by the user.

        This method searches for the uploaded file in storage and parses it based on its
        file extension. Supports various file formats including CSV, JSON, JSONL, Excel,
        and Parquet files. The data is then processed according to the split configuration:
        - NoSplitConfig: Returns all data as train split with optional sampling
        - ManualSplitConfig: Samples the data and splits into train/test sets

        Args:
            dataset_id (str): The unique identifier of the uploaded dataset.
            config (PreprocessingConfig): Configuration for processing, including split_config.

        Returns:
            DatasetDict: A dictionary with split names as keys and dataset samples as values.

        Raises:
            FileNotFoundError: If no file is found with the given dataset_id.
            ValueError: If the file format cannot be parsed.

        Example:
            >>> dataset = await loader._load_uploaded_dataset("123e4567-e89b-12d3-a456-426614174000", config)
        """
        files = self.storage.list_files(prefix=f"raw_datasets/{dataset_id}_")
        if not files:
            raise FileNotFoundError("Uploaded dataset not found")

        file_path = f"{self.storage.base_path}/{files[0]}"
        filename = file_path.split("_", 1)[1]
        # find the type of file
        file_type = filename.rsplit(".", 1)[1].lower()
        if (
            file_type != "csv"
            and file_type != "json"
            and file_type != "jsonl"
            and file_type != "parquet"
            and file_type != "xlsx"
            and file_type != "xls"
            and file_type != "txt"
            and file_type != "pdf"
            and file_type != "html"
            and file_type != "docx"
            and file_type != "pptx"
        ):
            raise ValueError("Invalid file type")

        if file_type in self.parsers:
            parser = self.parsers[file_type]
            file_content = self.storage.download_binary_data(files[0])
            parsed_dataset = parser.parse(io.BytesIO(file_content)) # gives the file object stream to the parser
            dataset = DatasetDict({"train": parsed_dataset})        
        else:
            if file_type == "txt":
                file_type = "text"
                dataset = load_dataset(file_type, data_files=file_path)
            else:    
                dataset = load_dataset(file_type, data_files=file_path)

        # No split configuration - return all data as train split
        if isinstance(config.split_config, NoSplitConfig):
            sample_size = config.split_config.sample_size

            if sample_size and dataset["train"].num_rows > sample_size:
                shuffled_dataset = (
                    dataset["train"].shuffle(seed=42).select(range(sample_size))
                )
                dataset = DatasetDict({"train": shuffled_dataset})

            return dataset

        # Manual split configuration - sample and split into train/test
        elif isinstance(config.split_config, ManualSplitConfig):
            sample_size = config.split_config.sample_size
            test_size = config.split_config.test_size
            train_dataset = dataset["train"]

            # if sample_size and test_size are provided, sample and split into train/test
            if sample_size and train_dataset.num_rows > sample_size and test_size:
                train_size = int(sample_size * (1 - test_size))
                shuffled_dataset = train_dataset.shuffle(seed=42)
                train_split = shuffled_dataset.select(range(train_size))
                test_split = shuffled_dataset.select(range(train_size, sample_size))
                return DatasetDict({"train": train_split, "test": test_split})

            # if sample_size is provided, sample the data
            elif sample_size and train_dataset.num_rows > sample_size:
                sampled_dataset = train_dataset.shuffle(seed=42).select(
                    range(sample_size)
                )
                return DatasetDict({"train": sampled_dataset})

            # if test_size is provided, split into train/test
            elif test_size:
                train_size = int(train_dataset.num_rows * (1 - test_size))
                shuffled_dataset = train_dataset.shuffle(seed=42)
                train_split = shuffled_dataset.select(range(train_size))
                test_split = shuffled_dataset.select(
                    range(train_size, train_dataset.num_rows)
                )
                return DatasetDict({"train": train_split, "test": test_split})

            # if no sampling or splitting is specified, return all data as train
            else:
                return dataset

    def _load_huggingface_dataset(
        self, dataset_name: str, dataset_subset: str, config: PreprocessingConfig
    ) -> DatasetDict:
        """
        Load a dataset from Hugging Face's dataset hub.

        This method downloads and loads a dataset from Hugging Face based on the split_config:
        - NoSplitConfig: Loads train split with sampling
        - ManualSplitConfig: Loads train split, samples it, then splits into train/test
        - HFSplitConfig: Loads all specified splits from Hugging Face

        NOTE: This will use streaming if sample_size is provided to avoid loading large datasets

        Args:
            dataset_name (str): The name of the dataset on Hugging Face (e.g., 'squad', 'glue').
            config (PreprocessingConfig): Configuration for processing, including split_config.

        Returns:
            DatasetDict: A dictionary with split names as keys and dataset samples as values.

        Raises:
            Exception: If there's an error loading the dataset from Hugging Face.

        Example:
            >>> dataset = await loader._load_huggingface_dataset("squad", config)
        """
        try:
            if isinstance(config.split_config, NoSplitConfig):
                if not config.split_config.split:
                    raise ValueError("No split specified")

                sample_size = config.split_config.sample_size
                if sample_size:
                    iter_ds: IterableDataset = load_dataset(
                        dataset_name,
                        dataset_subset,
                        split=config.split_config.split,
                        streaming=True,
                    )
                    # Take a sample of the dataset using streaming, it is still IterableDataset
                    iter_ds = iter_ds.shuffle(seed=42).take(sample_size)
                    # now convert to list and create a Dataset from it, this is where the iterator is executed
                    dataset = Dataset.from_list(list(iter_ds))
                else:
                    dataset: Dataset = load_dataset(
                        dataset_name,
                        dataset_subset,
                        split=config.split_config.split,
                    )

                return DatasetDict({"train": dataset})

            elif isinstance(config.split_config, ManualSplitConfig):
                if not config.split_config.split:
                    raise ValueError("No split specified")

                sample_size = config.split_config.sample_size
                test_size = config.split_config.test_size

                # we use streaming if sample size is provided to handle selecting a subset
                if sample_size:
                    iter_ds: IterableDataset = load_dataset(
                        dataset_name,
                        dataset_subset,
                        split=config.split_config.split,
                        streaming=True,
                    )
                    iter_ds = iter_ds.shuffle(seed=42).take(sample_size)
                    dataset = Dataset.from_list(list(iter_ds))
                else:
                    # Otherwise we load the entire dataset and assume the sample is all rows
                    dataset: Dataset = load_dataset(
                        dataset_name,
                        dataset_subset,
                        split=config.split_config.split,
                    )
                    sample_size = dataset.num_rows

                if test_size:
                    train_size = int(sample_size * (1 - test_size))
                    shuffled_dataset = dataset.shuffle(seed=42)
                    train_split = shuffled_dataset.select(range(train_size))
                    test_split = shuffled_dataset.select(range(train_size, sample_size))
                    return DatasetDict({"train": train_split, "test": test_split})
                else:
                    return DatasetDict({"train": dataset})

            elif isinstance(config.split_config, HFSplitConfig):
                # You cannot select sample_size when using HF splits directly
                if (
                    not config.split_config.train_split
                    or not config.split_config.test_split
                ):
                    raise ValueError("No train or test split specified")

                train_dataset = load_dataset(
                    dataset_name,
                    dataset_subset,
                    split=config.split_config.train_split,
                )

                test_dataset = load_dataset(
                    dataset_name,
                    dataset_subset,
                    split=config.split_config.test_split,
                )

                return DatasetDict({"train": train_dataset, "test": test_dataset})

            else:
                raise ValueError("Invalid split configuration")

        except Exception as e:
            logger.error(f"Error loading Hugging Face dataset: {str(e)}")
            raise
