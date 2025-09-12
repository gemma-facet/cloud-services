# Export Job Service

Cloud Run job that exports fine-tuned models in various formats and destinations.

## Structure

- **`main.py`** - Job entry point, loads export configuration and executes export
- **`utils.py`** - Core export logic with provider support and format conversion
- **`storage.py`** - Google Cloud Storage operations for downloading/uploading models
- **`schema.py`** - Export configuration models and data structures

## Execution Flow

1. **Start**: Job triggered by export service with `EXPORT_ID`
2. **Config**: Load export configuration from Firestore
3. **Validate**: Check required parameters and permissions
4. **Export**: Execute export based on type (adapter/merged/gguf)
5. **Upload**: Save artifacts to GCS and/or push to Hugging Face Hub
6. **Update**: Update job status and artifact paths in Firestore

## Required Environment Variables

- **`EXPORT_ID`** - Unique identifier for the export job (required)
- **`PROJECT_ID`** - Google Cloud project ID (required)
- **`HF_TOKEN`** - Hugging Face token for model access (required for HF Hub exports)
- **`GCS_EXPORT_BUCKET_NAME`** - GCS bucket for raw model storage (default: "gemma-facet-models")
- **`GCS_EXPORT_FILES_BUCKET_NAME`** - GCS bucket for zip files (default: "gemma-facet-files")

## Export Types

### Adapter Export

Exports the fine-tuned adapter weights (LoRA/QLoRA) from a training job.

**Supported Destinations:**
- **GCS**: Downloads adapter from training job, creates zip file, uploads to export bucket
- **HF Hub**: Downloads adapter, loads with appropriate provider, pushes to Hugging Face Hub

**Optimizations:**
- Short-circuits if GCS zip already exists and only GCS export is requested
- Reuses existing artifacts when possible

### Merged Export

Exports the full model with adapter weights merged into the base model.

**Supported Destinations:**
- **GCS**: Downloads/creates merged model, creates zip file, uploads to export bucket
- **HF Hub**: Downloads/creates merged model, loads with appropriate provider, pushes to Hugging Face Hub

**Optimizations:**
- Reuses existing raw merged artifacts when available
- Avoids loading models into memory unless HF Hub export is requested
- Creates and stores raw merged artifacts for future reuse

### GGUF Export

Exports the model in GGUF format for CPU inference using llama.cpp.

**Supported Destinations:**
- **GCS**: Downloads/creates merged model, converts to GGUF, uploads to export bucket
- **HF Hub**: Downloads/creates merged model, converts to GGUF, pushes to Hugging Face Hub

**Optimizations:**
- Reuses existing GGUF files when available
- Reuses existing merged models for conversion
- Short-circuits if GGUF zip already exists and only GCS export is requested

## Backend Providers

The export service supports two backend providers based on the base model used in training:

### Unsloth Provider

- **Detection**: Models with `unsloth/` prefix
- **Adapter Export**: Uses `FastModel.from_pretrained()` and `push_to_hub()`
- **Merged Export**: Uses `save_pretrained_merged()` and `push_to_hub_merged()`
- **Tokenizer**: Includes tokenizer in HF Hub exports

### Hugging Face Provider

- **Detection**: All other model IDs
- **Adapter Export**: Uses `PeftModel.from_pretrained()` and `push_to_hub()`
- **Merged Export**: Uses `merge_and_unload()` and `push_to_hub()`
- **Tokenizer**: Loads tokenizer separately for adapter exports

## Export Destinations

### Google Cloud Storage (GCS)

**File Variants:**
- **Raw**: Uncompressed model directories (for merged models)
- **File**: Compressed zip files for easy download

**Storage Structure:**
```
gs://export-bucket/
├── merged_models/{job_id}/          # Raw merged models
└── ...

gs://export-files-bucket/
├── {job_id}/
│   ├── adapter.zip                  # Adapter zip file
│   ├── merged.zip                   # Merged model zip file
│   └── model.gguf                   # GGUF file
└── ...
```

### Hugging Face Hub

**Repository Structure:**
- **Adapter**: PEFT adapter weights + tokenizer
- **Merged**: Full merged model + tokenizer (for Unsloth) or model only (for HF)
- **GGUF**: Single `model.gguf` file

**Requirements:**
- Valid `HF_TOKEN` environment variable
- `hf_repo_id` specified in export configuration
- Repository will be created automatically if it doesn't exist

## Artifact Management

The export service maintains three types of artifacts:

### Raw Artifacts
- **Purpose**: Uncompressed model directories for reuse
- **Storage**: GCS export bucket
- **Types**: `adapter`, `merged`
- **Usage**: Avoids re-downloading/merging for subsequent exports

### File Artifacts
- **Purpose**: Compressed files for easy download
- **Storage**: GCS files bucket
- **Types**: `adapter`, `merged`, `gguf`
- **Format**: ZIP files (except GGUF which is a single file)

### HF Hub Artifacts
- **Purpose**: Models published to Hugging Face Hub
- **Storage**: Hugging Face Hub
- **Types**: `adapter`, `merged`, `gguf`
- **Format**: Repository URLs

## GGUF Conversion

The service uses llama.cpp for GGUF conversion with the following configuration:

**Conversion Process:**
1. Downloads merged model from GCS or creates from adapter
2. Runs `convert_hf_to_gguf.py` with q8_0 quantization (default)
3. Uploads resulting GGUF file to specified destinations

**Requirements:**
- llama.cpp installed in `/app/llama_venv/`
- Python environment accessible at `/app/llama_venv/bin/python`
- Conversion script at `./llama.cpp/convert_hf_to_gguf.py`

**Timeout:** 1 hour maximum for conversion process

## Error Handling

**Validation Errors:**
- Missing required environment variables
- Invalid export configuration
- Missing training job or adapter path

**Processing Errors:**
- Model loading failures
- Conversion errors (GGUF)
- Upload failures (GCS/HF Hub)
- Network timeouts

**Recovery:**
- All errors are logged with detailed context
- Export status updated to "failed" with error message
- Local files cleaned up on failure
- Partial artifacts may be preserved for debugging

## Performance Optimizations

### Memory Management
- Models only loaded into memory when HF Hub export required
- Local files cleaned up immediately after use
- Streaming downloads for large files

### Caching Strategy
- Reuses existing raw artifacts when available
- Short-circuits exports when target artifacts already exist
- Stores intermediate artifacts for future reuse

### Parallel Processing
- Multiple export types can be processed in sequence
- GCS and HF Hub exports can be combined in single job

## Example Export Configurations

### Adapter Export (GCS Only)
```json
{
  "export_id": "exp_123",
  "job_id": "job_456",
  "type": "adapter",
  "destination": ["gcs"],
  "status": "running"
}
```

### Merged Export (Both Destinations)
```json
{
  "export_id": "exp_124",
  "job_id": "job_457",
  "type": "merged",
  "destination": ["gcs", "hf_hub"],
  "hf_repo_id": "username/my-fine-tuned-model",
  "status": "running"
}
```

### GGUF Export (HF Hub Only)
```json
{
  "export_id": "exp_125",
  "job_id": "job_458",
  "type": "gguf",
  "destination": ["hf_hub"],
  "hf_repo_id": "username/my-model-gguf",
  "status": "running"
}
```

## Deployment

```bash
gcloud builds submit --config cloudbuild.yaml --ignore-file=.gcloudignore
```

**Recommended Configuration:**
- **Memory**: 16Gi (for model loading and merging)
- **CPU**: 4 cores
- **Timeout**: 2 hours (for GGUF conversion)
- **Environment**: Python 3.12 with GPU support

## Dependencies

**Core Libraries:**
- `google-cloud-firestore` - Firestore database operations
- `google-cloud-storage` - GCS file operations
- `huggingface-hub` - HF Hub integration
- `transformers` - Model loading and manipulation
- `peft` - Parameter-efficient fine-tuning support
- `unsloth` - Optimized training and inference

**Optional Dependencies:**
- `llama.cpp` - GGUF conversion (installed separately)

## Known Limitations

- **GGUF Conversion**: Requires llama.cpp to be pre-installed in the container
- **Memory Usage**: Large models may require significant memory for merging
- **Network Dependencies**: Relies on stable internet for HF Hub uploads
- **Provider Compatibility**: Some advanced features may not work across all providers

## Troubleshooting

**Common Issues:**

1. **"Adapter path not found"**
   - Ensure training job completed successfully
   - Check that adapter was saved during training

2. **"HF token not found"**
   - Set `HF_TOKEN` environment variable
   - Ensure token has write permissions

3. **"llama.cpp conversion failed"**
   - Verify llama.cpp installation
   - Check model compatibility with GGUF format
   - Increase timeout for large models

4. **"Model merging failed"**
   - Ensure base model is accessible
   - Check adapter compatibility with base model
   - Verify sufficient memory available

**Debug Information:**
- All operations are logged with detailed context
- Export status and error messages stored in Firestore
- Local files preserved on failure for inspection
