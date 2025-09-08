from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional, List, Union


class StringCheckRewardConfig(BaseModel):
    """Config for checking if a string matches a reference."""

    name: str = Field(..., description="Unique name for this reward function instance.")
    type: Literal["string_check"] = "string_check"
    reference_field: str = Field(
        ...,
        description="Field in the dataset for the reference string to check against.",
    )
    operation: Literal["eq", "ne", "like", "ilike"] = Field(
        ..., description="The string comparison operation to perform."
    )


class TextSimilarityRewardConfig(BaseModel):
    """Config for calculating reward based on text similarity."""

    name: str = Field(..., description="Unique name for this reward function instance.")
    type: Literal["text_similarity"] = "text_similarity"
    gemini_api_key: Optional[str] = Field(
        None,
        description="Optional Gemini API key. If not provided, it will fall back to the GOOGLE_API_KEY environment variable.",
    )
    reference_field: str = Field(
        ...,
        description="Field in the dataset for the reference string for similarity comparison.",
    )
    evaluation_metric: Literal[
        "fuzzy_match",
        "bleu",
        "gleu",
        "meteor",
        "cosine",
        "rouge_1",
        "rouge_2",
        "rouge_3",
        "rouge_4",
        "rouge_5",
        "rouge_l",
    ]
    embedding_model: Optional[str] = Field(
        "models/embedding-001",
        description="The Google AI model for embeddings, required for the 'cosine' metric.",
    )


class ScoreModelRewardConfig(BaseModel):
    """Config for using an LLM to score a response numerically."""

    name: str = Field(..., description="Unique name for this reward function instance.")
    type: Literal["score_model"] = "score_model"
    gemini_api_key: Optional[str] = Field(
        None,
        description="Optional Gemini API key. If not provided, it will fall back to the GOOGLE_API_KEY environment variable.",
    )
    model: str = Field(
        "gemini-2.0-flash", description="The Gemini model to use as the judge."
    )
    prompt: str = Field(
        ...,
        description="The prompt/instruction for the judge model, with template support.",
    )
    range: Optional[List[float]] = Field(
        None, description="The output score will be clipped to this [min, max] range."
    )


class LabelModelRewardConfig(BaseModel):
    """Config for using an LLM to classify a response with a label."""

    name: str = Field(..., description="Unique name for this reward function instance.")
    type: Literal["label_model"] = "label_model"
    gemini_api_key: Optional[str] = Field(
        None,
        description="Optional Gemini API key. If not provided, it will fall back to the GOOGLE_API_KEY environment variable.",
    )
    model: str = Field(
        "gemini-2.0-flash", description="The Gemini model to use as the judge."
    )
    prompt: str = Field(
        ...,
        description="The prompt/instruction for the judge model, with template support.",
    )
    labels: List[str] = Field(
        ..., description="The set of all possible labels the judge can output."
    )
    passing_labels: List[str] = Field(
        ...,
        description="The subset of labels that correspond to a positive (1.0) reward.",
    )


class PythonRewardConfig(BaseModel):
    """Config for executing custom Python code as a reward function."""

    name: str = Field(..., description="Unique name for this reward function instance.")
    type: Literal["python"] = "python"
    source: str = Field(
        ...,
        description="A string containing the Python source code for the grader. Must contain a 'grade' function.",
    )


class BuiltInRewardParameters(BaseModel):
    """Parameters for built-in reward functions."""

    think_tag: Optional[str] = "think"
    answer_tag: Optional[str] = "answer"


class BuiltInRewardConfig(BaseModel):
    """Config for using a classic, built-in reward function."""

    name: str = Field(..., description="Unique name for this reward function instance.")
    type: Literal["built_in"] = "built_in"
    function_name: Literal[
        "format_reward",
        "count_xml",
        "expression_accuracy",
        "numerical_accuracy",
    ] = Field(..., description="The name of the built-in, batch-based reward function.")
    parameters: Optional[BuiltInRewardParameters] = Field(
        None, description="Optional parameters for the built-in function."
    )


class RulerRewardConfig(BaseModel):
    """Config for using the RULER model-based grader."""

    name: str = Field(..., description="Unique name for this reward function instance.")
    type: Literal["ruler"] = "ruler"
    gemini_api_key: Optional[str] = Field(
        None,
        description="Optional Gemini API key. If not provided, it will fall back to the GOOGLE_API_KEY environment variable.",
    )
    model: str = Field(
        "gemini-2.0-flash", description="The Gemini model to use as the judge."
    )
    rules: List[str] = Field(
        ..., description="A list of rules or principles for the judge to follow."
    )


AnyGraderConfig = Union[
    StringCheckRewardConfig,
    TextSimilarityRewardConfig,
    ScoreModelRewardConfig,
    LabelModelRewardConfig,
    PythonRewardConfig,
    BuiltInRewardConfig,
    RulerRewardConfig,
]


class HyperparameterConfig(BaseModel):
    """Training hyperparameters configuration"""

    # Basic hyperparameters
    learning_rate: float = 2e-4
    batch_size: int = 2  # batch size > 4 might cause OOM sometimes
    gradient_accumulation_steps: int = 4  # effective batch sz = 2*4 = 8
    epochs: int = 3  # ignored when max_steps provided
    max_steps: Optional[int] = -1  # Default to -1 instead of None to avoid operator err

    # Technical and optimization settings
    packing: bool = False  # whether to pack sequences for training
    padding_free: bool = False  # only works with FA2 or 3
    use_fa2: bool = False  # FA2 is only available when provider is "huggingface"
    max_length: Optional[int] = 1024  # used for truncation for all trainers
    lr_scheduler_type: Optional[str] = "linear"
    save_strategy: Optional[str] = "epoch"
    logging_steps: Optional[int] = 10

    # PEFT Config -- should be present if method is LoRA or QLoRA
    lora_rank: Optional[int] = 16
    lora_alpha: Optional[int] = 16
    lora_dropout: Optional[float] = 0.05

    # GRPO DPO shared
    max_prompt_length: Optional[int] = 512  # Max prompt length

    # GRPO-specific hyperparameters (some are shard with DPO)
    num_generations: Optional[int] = 4  # Number of generations for GRPO
    max_grad_norm: Optional[float] = 0.1  # Gradient clipping for GRPO
    adam_beta1: Optional[float] = 0.9  # Adam beta1 for GRPO
    adam_beta2: Optional[float] = 0.99  # Adam beta2 for GRPO
    warmup_ratio: Optional[float] = 0.1  # Warmup ratio for GRPO

    # DPO-specific hyperparameters
    beta: Optional[float] = 0.1  # DPO/ORPO regularization parameter


class EvaluationConfig(BaseModel):
    """Evaluation configuration during training"""

    # NOTE: Only specify eval_strategy if you actually provide eval_dataset
    eval_strategy: Optional[str] = "no"  # "no", "steps", "epoch"
    eval_steps: Optional[int] = 50  # Required if eval_strategy="steps"

    # Metrics configuration, otherwise eval only returns eval loss etc.
    # if true returns computed metrics ["accuracy", "perplexity"]
    compute_eval_metrics: Optional[bool] = False
    # Set to True to enable batch evaluation mode for metrics computation
    batch_eval_metrics: Optional[bool] = False


class WandbConfig(BaseModel):
    """Configuration for wandb monitoring, will extend to be a subclass of MonitoringConfig later"""

    api_key: str
    project: Optional[str] = None  # defaulted to "huggingface" if not provided
    log_model: Optional[Literal["false", "checkpoint", "end"]] = "end"


class ExportConfig(BaseModel):
    """Configuration for model export"""

    format: Literal["adapter", "merged", "full"] = "adapter"
    destination: Literal["gcs", "hfhub"] = "gcs"
    hf_repo_id: Optional[str] = None
    # Whether to also export a GGUF version alongside the main format
    include_gguf: Optional[bool] = False
    gguf_quantization: Optional[
        Literal[
            "none",
            "f16",
            "bf16",
            "q8_0",
            "q4_k_m",
        ]
    ] = None


class TrainingConfig(BaseModel):
    """Unified config structure for training, all customizations should be included here and ONLY here"""

    # Core configurations
    base_model_id: str
    provider: Literal["unsloth", "huggingface"] = "huggingface"
    method: Literal["Full", "LoRA", "QLoRA"] = "QLoRA"
    trainer_type: Literal["sft", "dpo", "grpo", "orpo"] = "sft"
    modality: Literal["text", "vision"] = "text"

    # Grouped configurations
    hyperparameters: HyperparameterConfig = HyperparameterConfig()
    export_config: ExportConfig = ExportConfig()

    # Optional configurations
    eval_config: Optional[EvaluationConfig] = None
    wandb_config: Optional[WandbConfig] = None

    # GRPO-specific reward config
    reward_config: Optional[List[AnyGraderConfig]] = None

    @field_validator("trainer_type")
    @classmethod
    def validate_trainer_compatibility(cls, v, info):
        """Validate trainer type compatibility with other config"""
        # Validation for trainer compatibility will be added later...
        return v


# NOTE: This struct is shared between the API and the backend service
class TrainRequest(BaseModel):
    """Request schema for training job, only TrainingConfig will be accessible in backend"""

    processed_dataset_id: str
    hf_token: str
    job_name: str = "unnamed job"
    training_config: TrainingConfig


class JobSubmitResponse(BaseModel):
    job_id: str


class EvaluationMetrics(BaseModel):
    """
    Evaluation metrics structure to hold results after training.
    This is used to store metrics like accuracy, loss, etc.
    """

    accuracy: Optional[float] = None
    perplexity: Optional[float] = None
    eval_loss: Optional[float] = None
    eval_runtime: Optional[float] = None


class JobStatusResponse(BaseModel):
    job_name: str
    status: Literal["queued", "preparing", "training", "completed", "failed"]
    modality: Optional[Literal["text", "vision"]] = "text"
    wandb_url: Optional[str] = None
    processed_dataset_id: Optional[str] = None
    adapter_path: Optional[str] = None
    base_model_id: Optional[str] = None
    # Path to GGUF file if it was exported alongside the main model
    gguf_path: Optional[str] = None
    # Evaluation metrics recorded after training
    metrics: Optional[EvaluationMetrics] = None
    error: Optional[str] = None


class JobListEntry(BaseModel):
    job_id: str
    job_name: str = "unnamed job"
    base_model_id: Optional[str] = None
    modality: Optional[Literal["text", "vision"]] = "text"
    # "unknown" is a fallback for jobs that don't have a status but are listed
    status: Literal[
        "queued", "preparing", "training", "completed", "failed", "unknown"
    ] = "unknown"


class JobListResponse(BaseModel):
    jobs: List[JobListEntry]


class DownloadUrlResponse(BaseModel):
    download_url: str


class JobDeleteResponse(BaseModel):
    job_id: str
    deleted: bool
    message: str
    deleted_resources: Optional[List[str]] = None
