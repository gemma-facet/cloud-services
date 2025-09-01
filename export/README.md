# Export Service

FastAPI service for exporting fine-tuned Gemma models in various formats.

## Structure

- **`main.py`** - FastAPI application with export endpoints
- **`utils.py`** - Model export utilities for merging and uploading
- **`schema.py`** - Request/response models

## Endpoints

### POST `/export`

Export a completed training job in the specified format.

**Request:**

```json
{
  "job_id": "training_abc123_gemma-2b_def456",
  "export_type": "adapter" | "merged" | "gguf",
  "hf_token": "hf_your_token_here"
}
```

**Export Types:**

- **`adapter`** - Returns the adapter path (if already available)
- **`merged`** - Merges adapter with base model and uploads to GCS (requires `hf_token` for gated models)
- **`gguf`** - ⚠️ **Not yet implemented** - Will convert to GGUF format

**Note:** The `hf_token` is required for accessing gated models like Gemma models from Hugging Face Hub.

**Response:**

```json
{
  "success": true,
  "job_id": "training_abc123_gemma-2b_def456",
  "export_type": "merged",
  "merged_path": "gs://gemma-export-bucket/merged_models/training_abc123_gemma-2b_def456/"
}
```

### GET `/health`

Health check endpoint.

**Response:**

```json
{
  "status": "healthy",
  "service": "export"
}
```

## Features

### Model Merging

The service supports merging adapters with base models for both training providers:

- **Unsloth**: Uses `save_pretrained_merged` with "merged_16bit" method
- **HuggingFace**: Uses PEFT's `merge_and_unload()` method

### Storage

- **GCS Integration**: Uploads merged models to `gs://gemma-export-bucket/merged_models/{job_id}/`
- **Automatic Cleanup**: Removes temporary files after successful upload
- **Firestore Updates**: Tracks export paths in the training jobs collection

### Security

- **Authentication**: Requires API Gateway authentication
- **Authorization**: Validates job ownership before export
- **Job Validation**: Only allows export of completed training jobs

## Environment Variables

- **`PROJECT_ID`** - Google Cloud project ID (required)
- **`GCS_EXPORT_BUCKET_NAME`** - GCS bucket for exports (default: "gemma-export-bucket")

## Job Lifecycle Integration

1. **Training Complete** → Job marked as "completed" in Firestore
2. **Export Request** → Service validates job ownership and status
3. **Model Processing** → Merges adapter with base model (if needed)
4. **Upload** → Saves to GCS and updates Firestore
5. **Cleanup** → Removes temporary files

## Limitations

- **GGUF Export**: Not yet implemented (requires llama.cpp conversion tools)
- **Provider Support**: Currently supports Unsloth and HuggingFace providers
- **Model Size**: Large models may require significant memory and processing time

## Error Handling

- **404**: Job not found or adapter not available
- **403**: Job ownership validation failed
- **400**: Job not completed or invalid export type
- **501**: GGUF export not implemented
- **500**: Internal server error during processing


