from pydantic import BaseModel, field_validator
from typing import Literal, Optional, List


# Define the available built-in reward functions for validation and auto-completion
RewardFunctionName = Literal[
    "think_format",
    "expression_accuracy",
    "numerical_accuracy",
    "correctness",
    "is_integer",
    "strict_format",
    "soft_format",
    "xml_count",
]


class RewardConfig(BaseModel):
    """
    Configures the reward functions to be used during training.
    Allows combining multiple weighted reward functions.
    """

    functions: List[RewardFunctionName]

    @field_validator("functions")
    @classmethod
    def no_duplicate_functions(cls, v: List[RewardFunctionName]):
        """Ensures that each reward function is only specified once."""
        if len(v) != len(set(v)):
            raise ValueError("Duplicate reward function names are not allowed.")
        return v


class HyperparameterConfig(BaseModel):
    """Training hyperparameters configuration"""

    # Basic hyperparameters
    learning_rate: float = 2e-4
    batch_size: int = 2  # batch size > 4 might cause OOM sometimes
    gradient_accumulation_steps: int = 4  # effective batch sz = 2*4 = 8
    epochs: int = 3  # ignored when max_steps provided
    max_steps: Optional[int] = -1  # Default to -1 instead of None to avoid operator err

    # Technical and optimization settings
    packing: bool = False  # whether to pack sequences for training, only works with FA2
    use_fa2: bool = False  # FA2 is only available when provider is "huggingface"
    max_seq_length: Optional[int] = None  # used to load pretrained models
    lr_scheduler_type: Optional[str] = "linear"
    save_strategy: Optional[str] = "epoch"
    logging_steps: Optional[int] = 10

    # PEFT Config -- should be present if method is LoRA or QLoRA
    lora_rank: Optional[int] = 16
    lora_alpha: Optional[int] = 16
    lora_dropout: Optional[float] = 0.05

    # GRPO DPO shared
    max_prompt_length: Optional[int] = 512  # Max prompt length
    max_length: Optional[int] = 1024  # Max length for DPO

    # GRPO-specific hyperparameters (some are shard with DPO)
    num_generations: Optional[int] = 2  # Number of generations for GRPO
    max_grad_norm: Optional[float] = 0.1  # Gradient clipping for GRPO
    adam_beta1: Optional[float] = 0.9  # Adam beta1 for GRPO
    adam_beta2: Optional[float] = 0.99  # Adam beta2 for GRPO
    warmup_ratio: Optional[float] = 0.1  # Warmup ratio for GRPO

    # DPO-specific hyperparameters
    beta: Optional[float] = 0.1  # DPO regularization parameter


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

    format: Literal["adapter", "merged"] = "adapter"
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
    trainer_type: Literal["sft", "dpo", "grpo"] = "sft"
    modality: Literal["text", "vision"] = "text"

    # Grouped configurations
    hyperparameters: HyperparameterConfig = HyperparameterConfig()
    export_config: ExportConfig = ExportConfig()

    # Optional configurations
    eval_config: Optional[EvaluationConfig] = None
    wandb_config: Optional[WandbConfig] = None

    # GRPO-specific: reward functions configuration
    reward_config: Optional[RewardConfig] = None

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
