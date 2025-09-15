# Training Service

FastAPI service for managing fine-tuning jobs on Gemma models.

## Structure

- **`app.py`** - FastAPI application with job management endpoints
- **`job_manager.py`** - Job state management with Firestore
- **`schema.py`** - Request/response models

## Deployment

```bash
gcloud builds submit --config cloudbuild.yaml --ignore-file=.gcloudignore
```

## Endpoints

### POST `/train`

> NOTE: You should definitely read through the [training job README](jobs/training-job/README.md) first to understand how the training job works and what configurations are required in the `training_config` field.

Start a new training job.

**Request:**

```json
{
  "processed_dataset_id": "dataset_abc123",
  "hf_token": "hf_...",
  "job_name": "my-job-name",
  "training_config": {
    "base_model_id": "google/gemma-2b",
    "provider": "huggingface",
    "method": "LoRA",
    "trainer_type": "sft",
    "modality": "text",
    "hyperparameters": {
      "learning_rate": 0.0001,
      "batch_size": 4,
      "gradient_accumulation_steps": 4,
      "epochs": 3,
      "max_steps": -1,
      "packing": false,
      "use_fa2": false,
      "max_length": 2048,
      "lr_scheduler_type": "linear",
      "save_strategy": "epoch",
      "logging_steps": 10,
      "lora_rank": 16,
      "lora_alpha": 16,
      "lora_dropout": 0.05
    },
    "export_config": {
      "format": "merged",
      "destination": "gcs",
      "hf_repo_id": null,
      "include_gguf": false,
      "gguf_quantization": null
    },
    "eval_config": {
      "eval_strategy": "epoch",
      "eval_steps": 50,
      "compute_eval_metrics": true,
      "batch_eval_metrics": false
    },
    "reward_config": [
      {
        "name": "builtin_format",
        "type": "built_in",
        "function_name": "format_reward",
        "parameters": {
          "think_tag": "reasoning",
          "answer_tag": "answer"
        }
      }
    ],
    "wandb_config": null
  }
}
```

**Response:**

```json
{
  "job_id": "training_abc123_gemma-2b_def456"
}
```

### GET `/jobs`

List all jobs.

**Response:**

```json
{
  "jobs": [
    {
      "job_id": "training_abc123_gemma-2b_def456",
      "job_name": "My Training Job",
      "base_model_id": "google/gemma-1b-it",
      "status": "queued" | "preparing" | "training" | "completed" | "failed" | "unknown",
      "modality": "text" | "vision",
    }
  ]
}
```

### GET `/jobs/{job_id}`

Get training job status.

**Response:**

```json
{
  "job_name": "My Training Job",
  "status": "queued" | "preparing" | "training" | "completed" | "failed",
  "modality": "text" | "vision",
  "wandb_url": "https://wandb.ai/...",
  "adapter_path": "gs://bucket/trained_adapters/job123/ or gs://bucket/merged_models/job123/",
  "base_model_id": "google/gemma-2b",
  "gguf_path": "gs://bucket/gguf_models/job123/model-q8_0.gguf",
  "metrics": {
    "accuracy": 0.95,
    "perplexity": 1.23,
    "eval_loss": 0.156,
    "eval_runtime": 12.34
  },
  "error": "Error message if failed"
}
```

### GET `/jobs/download/{job_id}`

Get pre-signed URL from GCS to download GGUF file.

> [!CAUTION]
> This does not yet work because the service account has some issue with permissions (private key needed to sign bucket??). For now just download from the public URL public access is enabled on this bucket because all it has are files to download.

**Response:**

```json
{
  "download_url": "https://storage.googleapis.com/bucket/gguf_models/job123/model-q8_0.gguf?..."
}
```

### DELETE `/jobs/{job_id}`

Delete a training job and all associated files (at firestore and GCS).

**Response:**

```json
{
  "job_id": "training_abc123_gemma-2b_def456",
  "deleted": true,
  "message": "Job and all associated files deleted successfully.",
  "deleted_resources": [
    "gs://bucket/trained_adapters/job123/",
    "gs://bucket/merged_models/job123/",
    "gs://bucket/gguf_models/job123/"
  ]
}
```

### GET `/health`

Health check endpoint.

## Training Jobs Document Schema

This is the single-most authoritative source of truth for the training job document stored in Firestore. The following files use this exact identical schema, and if out of sync here is where it should be consulted:

```json
{
  "job_id": "job_12345",
  "job_name": "My Training Job",
  "user_id": "user_abcde",
  "base_model_id": "google/gemma-2b",
  "processed_dataset_id": "processed_67890",
  "modality": "text",
  "status": "pending",
  "dataset_id": "dataset_67890",
  "artifacts": {
    "raw": {
      "adapter": "gs://bucket/trained_adapters/job_12345/adapter",
      "merged": null,
    },
    "file": {
      "adapter": "https://storage.googleapis.com/gemma-facet-files/.../adapter.zip",
      "merged": null,
      "gguf": "https://storage.googleapis.com/gemma-facet-files/.../model.gguf"
    },
    "hf": {
      "adapter": "https://huggingface.co/username/model-name/adapter",
      "merged": null,
      "gguf": null
    }
  },
  "created_at": "2024-10-01T12:00:00Z",
  "updated_at": "2024-10-01T12:30:00Z"
  "wandb_url": "https://wandb.ai/...",
  "metrics": {
    "accuracy": 0.95,
    "perplexity": 1.23,
    "eval_loss": 0.156,
    "eval_runtime": 12.34
  },
  "error": "Error message if failed"
}
```

This is used in:

- `training/schema.py` and `training/job_manager.py`
- `jobs/training-job/job_manager.py`
- `export/schema.py`
- `jobs/export-job/utils.py`

> IMPORTANT! If the model is a full fine tuned model, we reuse the `artifacts.raw.adapter` field because a FFT model will NEVER have an adapter. This avoid replicating all the logic with storage + export.

## Job Lifecycle

1. **Submit** → Job queued in Firestore
2. **Start** → Cloud Run job triggered
3. **Track** → Status updates via Firestore
4. **Complete** → Model exported to GCS/HF Hub

## Configuration

- **Environment**: `PROJECT_ID`, `REGION`, `GCS_CONFIG_BUCKET_NAME`
- **Storage**: Training configs stored in GCS
- **Monitoring**: Weights & Biases integration
- **Port**: 8080 (default)
