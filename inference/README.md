# Inference Service

FastAPI service for running inference on fine-tuned Gemma models.

## Structure

- **`app.py`** - FastAPI application with endpoints
- **`base.py`** - Core inference orchestration logic
- **`providers.py`** - Inference provider implementations (HuggingFace, Unsloth, vLLM)
- **`storage.py`** - Model loading from GCS/HuggingFace Hub
- **`schema.py`** - Request/response models
- **`evaluation.py`** - Evaluation logic for fine-tuned models using `evaluate`
- **`utils.py`** - Utility functions for modality detection and storage type inference

## Deployment

The `cloudbuild.yaml` handles the build, push to artifact, and deploying / updating service.

```bash
cd inference
gcloud builds submit --config cloudbuild.yaml --ignore-file=.gcloudignore
```

## Endpoints

> [!NOTE]
> Whenever `model_source` is referred to, it can be an adapter path, merged model path, or base model path. It can be either a GCS path or a HuggingFace Hub path. The `model_type` field explicitly specifies whether this is an "adapter", "merged", or "base" model. These are handled by 1. the storage logic and 2. the provider logic.

### POST `/inference`

Single inference request. The service automatically detects storage type from the model source. Both `model_source` and `base_model_id` are accessible on the job object in the frontend because the training service returns these fields.

> This is for text only!!! You should use `/inference/batch` for more complex structure or vision fields, the messages can be obtained from the preprocessing service.

**Request:**

```json
{
  "hf_token": "hf_your_token_here",
  "model_source": "/path/to/model",
  "model_type": "adapter",
  "base_model_id": "google/gemma-3-2b-pt",
  "prompt": "Your input text here"
}
```

**Model Source Examples:**

- **GCS path**: `gs://bucket/trained_adapters/job_123/adapter`
- **HuggingFace Hub**: `username/model-name`

**Model Type Values:**

- **`"adapter"`**: LoRA/QLoRA adapter that needs to be loaded with base model
- **`"merged"`**: Fully merged model with adapter weights integrated
- **`"base"`**: Base model without any fine-tuning

**Response:**

```json
{
  "result": "Generated text"
}
```

### POST `/inference/batch`

Batch inference for multiple conversations.

**Request:**

```json
{
  "hf_token": "hf_your_token_here",
  "model_source": "username/model-name",
  "model_type": "merged",
  "base_model_id": "google/gemma-3-2b-pt",
  "messages": [
    [{ "role": "user", "content": "What is the capital of France?" }],
    [{ "role": "user", "content": "Explain quantum computing." }]
  ],
  "use_vllm": true
}
```

**Response:**

```json
{
  "results": ["Paris is the capital of France.", "Quantum computing is..."]
}
```

#### Vision Batch Inference

Image field value should be a base64-encoded string, which can be obtained from the `get_dataset_preview` endpoint from preprocessing service (it returns the entire `messages` structure but contains assistant field).

```json
{
  "hf_token": "hf_your_token_here",
  "model_source": "username/model-name",
  "model_type": "adapter",
  "base_model_id": "google/gemma-3-2b-pt",
  "messages": [
    [
      {
        "content": [{ "type": "text", "text": "..." }],
        "role": "system"
      },
      {
        "content": [
          { "type": "text", "text": "..." },
          {
            "type": "image",
            "image": "data:image/png;base64,<base64_encoded_image>"
          }
        ],
        "role": "user"
      }
    ]
  ]
}
```

### POST `/evaluation`

Evaluate a fine-tuned model on a dataset. You can use either **task types** (recommended) or specific metrics.

#### Using Task Types (Recommended)

```json
{
  "hf_token": "hf_your_token_here",
  "model_source": "username/model-name",
  "model_type": "merged",
  "base_model_id": "google/gemma-3-2b-pt",
  "dataset_id": "processed_dataset_123",
  "task_type": "conversation",
  "num_sample_results": 3
}
```

**Available Task Types:**

- `"conversation"` → bertscore, rouge
- `"qa"` → exact_match, bertscore
- `"summarization"` → rouge, bertscore
- `"translation"` → bleu, meteor
- `"classification"` → accuracy, recall, precision, f1, exact_match
- `"general"` → bertscore, rouge

#### Using Specific Metrics

```json
{
  "hf_token": "hf_your_token_here",
  "model_source": "username/model-name",
  "model_type": "adapter",
  "base_model_id": "google/gemma-3-2b-pt",
  "dataset_id": "processed_dataset_123",
  "metrics": ["bertscore", "rouge"],
  "num_sample_results": 3
}
```

**Available Metrics:**

- `bertscore`: Semantic similarity (⭐ **Recommended for LLMs**)
- `rouge`: Text overlap and summarization quality (⭐ **Recommended for LLMs**)
- `exact_match`: Exact string matching (good for QA)
- `accuracy`: Classification accuracy
- `recall`, `precision`, `f1`: Classification metrics
- `bleu`, `meteor`: Translation metrics

> More metrics will be added soon

**Response:**

```json
{
  "metrics": {
    "bertscore_f1": 0.85,
    "rouge_l": 0.82,
    "rouge_1": 0.85,
    "rouge_2": 0.78
  },
  "samples": [
    {
      "prediction": "Model's generated text",
      "reference": "Ground truth text",
      "sample_index": 42
    },
    {
      "prediction": "Another prediction",
      "reference": "Another reference",
      "sample_index": 156
    }
  ],
  "num_samples": 1000,
  "dataset_id": "processed_dataset_123",
  "task_type": "conversation"
}
```

### GET `/health`

Health check endpoint.

**Response:**

```json
{
  "status": "healthy",
  "service": "inference"
}
```

## Inference Providers

The service supports multiple inference providers for different use cases:

### HuggingFace Transformers (`huggingface`)

- **Use case**: Standard inference with HuggingFace models
- **Supports**: Text and vision models, adapters and merged models
- **Best for**: Most general use cases, well-tested compatibility

### Unsloth (`unsloth`)

- **Use case**: Optimized inference for Unsloth-trained models
- **Supports**: Text and vision models, adapters and merged models
- **Best for**: Models trained with Unsloth framework

### vLLM (`vllm`)

- **Use case**: High-performance inference with vLLM
- **Supports**: Text and vision models, adapters (via LoRA) and merged models
- **Best for**: High-throughput production inference
- **Requirements**:
  - For adapters: Uses LoRA requests with base model
  - For merged/base: Direct model loading

## Environment

- Cloud Run service
- Environment variable: `PROJECT_ID` required for Firestore
- Port: 8000 (default)
