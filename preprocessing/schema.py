from pydantic import BaseModel, model_validator
from typing import Optional, Dict, List, Any, Literal, Union
from enum import Enum

MIME_TYPES = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "xls": "application/vnd.ms-excel",
    "txt": "text/plain",
    "csv": "text/csv",
    "json": "application/json",
    "jsonl": "application/x-ndjson",
    "html": "text/html",
    "parquet": "application/vnd.apache.parquet",
}


class ProcessingMode(str, Enum):
    """
    Specifies the preprocessing mode to format the dataset for a specific fine-tuning task.
    """

    LANGUAGE_MODELING = "language_modeling"
    PROMPT_ONLY = "prompt_only"
    PREFERENCE = "preference"


class DatasetUploadResponse(BaseModel):
    dataset_id: str
    filename: str
    gcs_path: str
    size_bytes: int
    sample: List[Dict[str, Any]]
    columns: List[str]
    num_examples: int


class FieldMappingConfig(BaseModel):
    type: Literal["column", "template", "image"]
    value: str


class BaseSplitConfig(BaseModel):
    type: Literal["hf_split", "manual_split", "no_split"]


class HFSplitConfig(BaseSplitConfig):
    type: Literal["hf_split"] = "hf_split"
    train_split: str
    test_split: str


class ManualSplitConfig(BaseSplitConfig):
    type: Literal["manual_split"] = "manual_split"
    sample_size: Optional[int] = None
    test_size: Optional[float] = None
    split: Optional[str] = None


class NoSplitConfig(BaseSplitConfig):
    type: Literal["no_split"] = "no_split"
    sample_size: Optional[int] = None
    split: Optional[str] = None


class AugmentationSetupConfig(BaseModel):
    """
    Currently we do not support customising parameters for augmentation methods.
    This is supported in the backend but we do not expose it in the API for now for simplicity.
    """

    augmentation_factor: float = 1.5
    use_eda: Optional[bool] = False
    use_back_translation: Optional[bool] = False
    use_paraphrasing: Optional[bool] = False
    use_synthesis: Optional[bool] = False
    gemini_api_key: Optional[str] = None
    synthesis_ratio: Optional[float] = None
    custom_prompt: Optional[str] = None


class PreprocessingConfig(BaseModel):
    field_mappings: Dict[str, Union[FieldMappingConfig, List[FieldMappingConfig]]] = {}
    normalize_whitespace: bool = True
    augmentation_config: Optional[AugmentationSetupConfig] = None
    split_config: Optional[HFSplitConfig | ManualSplitConfig | NoSplitConfig] = None


class PreprocessingRequest(BaseModel):
    dataset_name: str
    dataset_source: Literal["upload", "huggingface"]
    dataset_id: str
    dataset_subset: str = "default"
    processing_mode: ProcessingMode
    config: PreprocessingConfig

    @model_validator(mode="after")
    def validate_field_mappings(self) -> "PreprocessingRequest":
        if self.processing_mode == ProcessingMode.LANGUAGE_MODELING:
            required_fields = ["user_field", "assistant_field"]
        elif self.processing_mode == ProcessingMode.PROMPT_ONLY:
            required_fields = ["user_field"]
        elif self.processing_mode == ProcessingMode.PREFERENCE:
            required_fields = ["user_field", "chosen_field", "rejected_field"]

        for field in required_fields:
            if field not in self.config.field_mappings:
                raise ValueError(
                    f"'{field}' is required in field_mappings for {self.processing_mode} mode."
                )

        # Validate that image fields only appear within user_field
        for field_name, field_config in self.config.field_mappings.items():
            if field_name != "user_field":
                # non-user fields can only be single config and non-image type
                if isinstance(field_config, list):
                    raise ValueError(
                        f"Field '{field_name}' cannot be a list. Only 'user_field' supports list format."
                    )
                elif (
                    isinstance(field_config, dict)
                    and field_config.get("type") == "image"
                ):
                    raise ValueError(
                        f"Image fields are only allowed within 'user_field', not in '{field_name}'"
                    )

        return self


class DatasetInfoSample(BaseModel):
    dataset_name: str
    dataset_subset: str
    dataset_source: Literal["upload", "huggingface"]
    # Modality of the dataset: 'text' or 'vision'
    modality: Literal["text", "vision"] = "text"
    dataset_id: str
    processed_dataset_id: str
    num_examples: int
    created_at: str
    splits: List[str]


class ProcessingResult(DatasetInfoSample):
    full_splits: List[Dict[str, Any]] = []


class DatasetsInfoResponse(BaseModel):
    datasets: List[DatasetInfoSample]


class DatasetInfoFull(BaseModel):
    dataset_name: str
    dataset_subset: str
    dataset_source: Literal["upload", "huggingface"]
    # Modality of the dataset: 'text' or 'vision'
    modality: Literal["text", "vision"] = "text"
    dataset_id: str
    processed_dataset_id: str
    created_at: str
    splits: List[Dict[str, Any]]


class DatasetInfoResponse(DatasetInfoFull):
    pass


class DatasetDeleteResponse(BaseModel):
    dataset_name: str
    deleted: bool
    message: str
    deleted_files_count: int
    deleted_resources: Optional[List[str]] = None
