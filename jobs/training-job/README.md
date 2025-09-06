# Training Job Service

Cloud Run job that executes fine-tuning on Gemma models.

## Structure

- **`main.py`** - Job entry point, loads config and starts training
- **`base.py`** - Base class for training jobs with common workflow / functionality
- **`providers.py`** - Core training logic with provider support
- **`job_manager.py`** - Job state tracking and Firestore integration
- **`storage.py`** - Model and dataset saving/loading from GCS/HF Hub
- **`schema.py`** - Training configuration models

## Deployment

```bash
gcloud builds submit --config cloudbuild.yaml --ignore-file=.gcloudignore
```

- Cloud Run job with GPU support (L4)
- Memory: 32Gi, CPU: 8 cores
- GPU: 1x NVIDIA L4

## Execution Flow

1. **Start**: Job triggered by training service with `JOB_ID`
2. **Config**: Load training config from GCS bucket
3. **Login**: Authenticate with HuggingFace using token
4. **Train**: Execute fine-tuning with selected provider
5. **Evaluate**: Run evaluation if configured
6. **Track**: Update job status in Firestore
7. **Export**: Save model to GCS or push to HF Hub

## Required Configurations

To create a training job, the following information must be provided or a default option will be used.

### Modaltiy

- **text**: Text-based models (e.g., Gemma3 1B)
- **vision**: Multimodal models (e.g., Gemma3 4B)
- **audio**: Audio-based models (e.g., Gemma3n, THIS IS NOT YET SUPPORTED!)

### Providers

- **unsloth**: Optimized training with Unsloth library _<-- default_
- **huggingface**: Standard HuggingFace training

### PEFT + Quantisation

- **Full**: Full fine-tuning (requires more memory)
- **LoRA**: Low-rank adaptation (memory efficient)
- **QLoRA**: Quantized LoRA (4-bit precision) _<-- default_

### Trainer

> This only supports trainers in the TRL (transformers reinforcement learning) library. Dataset checks are not enforced for now but will be later!! Please use your judgement for now since it will just break otherwise...

- **SFTTrainer**: require `language-modelling` or `prompt-completion` type dataset _<-- default_
- **GRPOTrainer**: Used for reasoning tasks, require `prompt-only` type dataset
- **DPOTrainer**: Used for preference tuning, require `preference` type dataset

### Hyperparameters

All hyperparameters are grouped under the `hyperparameters` section of the config. Example fields:

```json
"hyperparameters": {
  "learning_rate": 0.0002,
  "batch_size": 4,
  "gradient_accumulation_steps": 4,
  "epochs": 3,
  "max_steps": -1,
  "packing": false,
  "use_fa2": false,
  "max_seq_length": 2048,
  "lr_scheduler_type": "linear",
  "save_strategy": "epoch",
  "logging_steps": 10,
  "lora_rank": 16,
  "lora_alpha": 16,
  "lora_dropout": 0.05
}
```

**Field descriptions:**

- `learning_rate`: Learning rate for optimizer (default: 2e-4)
- `batch_size`: Per-device batch size (default: 2)
- `gradient_accumulation_steps`: Number of steps to accumulate gradients (default: 4)
- `epochs`: Number of epochs to train (default: 3)
- `max_steps`: Maximum number of steps to train (default: -1 for unlimited)
- `packing`: Whether to use sequence packing (default: false, only works with FA2)
- `use_fa2`: Use FlashAttention-2 (default: false, only available for HuggingFace provider)
- `max_seq_length`: Maximum sequence length for model/tokenizer
- `lr_scheduler_type`: Learning rate scheduler type (default: "linear")
- `save_strategy`: When to save checkpoints (default: "epoch")
- `logging_steps`: Logging frequency (default: 10)
- `lora_rank`, `lora_alpha`, `lora_dropout`: LoRA/QLoRA PEFT parameters (used if method is LoRA/QLoRA)

### Export

All export options are grouped under the `export_config` section of the config. Example fields:

```json
"export_config": {
  "format": "adapter",           // "adapter" or "merged"
  "destination": "gcs",          // "gcs" or "hfhub"
  "hf_repo_id": null,             // HuggingFace repo ID (if using hfhub)
  "include_gguf": false,          // Whether to also export GGUF format
  "gguf_quantization": null       // GGUF quantization type (see below)
}
```

**Field descriptions:**

- `format`: Export format, either `adapter` (LoRA/QLoRA weights only, default) or `merged` (full model, usually f16 for vLLM)
- `destination`: Where to export the model (`gcs` for Google Cloud Storage, `hfhub` for HuggingFace Hub)
- `hf_repo_id`: (Optional) HuggingFace repo ID for upload
- `include_gguf`: (Optional) Also export a GGUF file for CPU inference (default: false)
- `gguf_quantization`: (Optional) GGUF quantization type, one of: `none`, `f16`, `bf16`, `q8_0`, `q4_k_m` (default: `q8_0`)

## Optional Configurations

The following features will be turned off by default if not provided.

### Evaluation

Configure evaluation during training:

- **eval_strategy**: When to run evaluation (`no`, `steps`, `epoch`)
- **eval_steps**: Evaluation frequency when using `steps` strategy
- **evaluation_metrics**: `true/false` whether to compute evaluation metrics (accuracy and perplexity, token level)
- **batch_eval_metrics**: Enable batch evaluation mode for better performance

> If you want task-specific metrics, currently they are only supported in the inference service with the `evaluation` endpoint

### Monitoring

- **wandb**: Provide your wandb API key and other variables to set up cloud logging
- **streaming logs**: Stream logs to frontend or a CLI whatever toolkit (NOT SUPPORTED, PLANNED)

### Reward Functions (for GRPO)

When using the `GRPOTrainer`, you must provide a `reward_config` list in your training configuration. This list defines one or more reward functions, or "graders," that score the model's generated responses. The final reward is the sum of the scores from all configured graders.

There are three main categories of graders available:

#### 1. Built-in Functions

These are simple, non-LLM-based functions for common checks.

- **`expression_accuracy`**: Verifies mathematical expressions.
- **`numerical_accuracy`**: Verifies a numerical answer against a solution.
- **`format_reward`**: Checks if the output strictly follows a `<think>...</think><answer>...</answer>` format. The tags are configurable via the `parameters` field (e.g., `think_tag: "reasoning"`).

#### 2. Reference-Based Graders

These graders compare the model's output to a reference column in your dataset.

- **`string_check`**: Performs simple string comparisons (e.g., `eq`, `like`).
- **`text_similarity`**: Calculates a score based on text similarity metrics (e.g., `bleu`, `rouge_l`, `cosine`).

#### 3. Model-Based Graders (Reference-Free)

These are the most powerful graders. They use a judge LLM (e.g., Gemini 2.0 Flash) to evaluate responses without needing a ground-truth solution.

- **`score_model`**: Assigns a numerical score (0.0 to 1.0) to a response based on a custom instruction you provide in the `prompt` field.
- **`label_model`**: Classifies a response with a specific label from a list you provide. Useful for categorical evaluation.
- **`ruler`**: A sophisticated grader that evaluates a batch of responses relative to each other based on a set of `rules` you define. This is highly effective for GRPO as it focuses on relative quality rather than absolute scores.

**Configuration Note:** For all model-based graders (`text_similarity` with cosine, `score_model`, `label_model`, and `ruler`), you can provide a `gemini_api_key` directly in the configuration. If omitted, the system will fall back to using the `GOOGLE_API_KEY` environment variable.

#### Example `reward_config`

```json
"reward_config": [
  {
    "name": "builtin_format",
    "type": "built_in",
    "function_name": "format_reward",
    "parameters": {
      "think_tag": "reasoning",
      "answer_tag": "answer"
    }
  },
  {
    "name": "similarity_check",
    "type": "text_similarity",
    "reference_field": "reference_answer",
    "evaluation_metric": "bleu"
  },
  {
    "name": "ruler_grader",
    "type": "ruler",
    "model": "gemini-2.0-flash",
    "rules": [
      "The response should show clear reasoning before the answer.",
      "The final answer should be a number at the end of the response."
    ],
    "gemini_api_key": "YOUR_GEMINI_API_KEY"
  }
]
```

You can combine any number of graders in the list. Each grader will contribute to the final reward used for

## Example Training Config

I will update this later...

**Training Config:**

> NOTE: This is completely identically shared with `TrainingConfig` from `training/schema.py`. This describes the `training_config` field, not the entire request.

```json
{
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
    "max_seq_length": 2048,
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
```

## Known Issues

- GGUF export is currently not supported by HuggingFace but you can do it manually using `llama.cpp` or `ollama` tools.

- Evaluation during training is not supported for Unsloth, this might be worked on by unsloth team (use eval in inference service instead)

- Evaluation metrics might not work on hugging face due to OOM depending on your config...
