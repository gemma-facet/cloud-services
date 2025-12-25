from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Literal, Union

ModelType = Literal["adapter", "merged", "base"]


class InferenceRequest(BaseModel):
    hf_token: Optional[str] = None  # HF Token (optional, uses HF_TOKEN env var if not provided)
    # Storage path or identifier (GCS path, HF Hub repo ID, or local path)
    model_source: str
    # Type of model: adapter, merged, or base model
    model_type: ModelType
    # Base model ID to use for tokenizer and model class selection
    base_model_id: str
    # A single text message
    prompt: str
    # Whether to use vLLM for inference (overrides default provider selection)
    use_vllm: Optional[bool] = False


class InferenceResponse(BaseModel):
    result: str


class BatchInferenceRequest(BaseModel):
    hf_token: Optional[str] = None  # HF Token (optional, uses HF_TOKEN env var if not provided)
    # Storage path or identifier (GCS path, HF Hub repo ID, or local path)
    model_source: str
    # Type of model: adapter, merged, or base model
    model_type: ModelType
    # Base model ID to use for tokenizer and model class selection
    base_model_id: str
    # A list of conversations, where each conversation is a list of messages
    messages: List[List[Dict[str, Any]]]
    # Whether to use vLLM for inference (overrides default provider selection)
    use_vllm: Optional[bool] = False


class BatchInferenceResponse(BaseModel):
    results: list[str]


TaskType = Literal[
    "conversation", "qa", "summarization", "translation", "classification", "general"
]

MetricType = Literal[
    "rouge",
    "bertscore",
    "accuracy",
    "exact_match",
    "bleu",
    "meteor",
    "recall",
    "precision",
    "f1",
]


class EvaluationRequest(BaseModel):
    hf_token: Optional[str] = None  # HF Token (optional, uses HF_TOKEN env var if not provided)
    # Storage path or identifier (GCS path, HF Hub repo ID, or local path)
    model_source: str
    # Type of model: adapter, merged, or base model
    model_type: ModelType
    # Base model ID to use for tokenizer and model class selection
    base_model_id: str
    # Dataset ID to evaluate on (must have an eval split)
    dataset_id: str
    # Task type for predefined metric suite (mutually exclusive with metrics)
    task_type: Optional[TaskType] = None
    # Specific list of metrics to compute (mutually exclusive with task_type)
    metrics: Optional[List[MetricType]] = None
    # Maximum number of samples to evaluate (optional, for faster evaluation)
    max_samples: Optional[int] = None
    # Number of sample results to include in response (default: 3)
    num_sample_results: Optional[int] = 3
    # Whether to use vLLM for inference (overrides default provider selection)
    use_vllm: Optional[bool] = False


class SampleResult(BaseModel):
    # Model's generated prediction
    prediction: str
    # Ground truth reference
    reference: str
    # Index of the sample in the evaluation dataset
    sample_index: int
    # Input messages/question (with images converted to base64 for API compatibility)
    input: Optional[List[Dict[str, Any]]] = None


class EvaluationResponse(BaseModel):
    # Computed metrics results (can be simple floats or nested dicts for complex metrics like bertscore)
    metrics: Dict[str, Union[float, Dict[str, float]]]
    # Number of samples evaluated
    num_samples: int
    # Dataset ID that was evaluated
    dataset_id: str
    # Sample results for inspection
    samples: List[SampleResult]
