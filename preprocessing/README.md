# Preprocessing Service

FastAPI service for preprocessing datasets for Gemma fine-tuning. Handles uploading, processing, and storing datasets in Google Cloud Storage or the local file system.

> [!NOTE]
> This service chooses to use conversational format with the ChatML variation. It covers multiple use cases by supporting different types.
> [!CAUTION]
> This service is not responsible for applying chat template and tokenizing / applying processors. This is a huge difference with some documentation that you will see on TRL, HF, or Unsloth. We have a specialized vision collator to do these **during training** (because it is meaningless to save these tokens in the dataset).

## Structure

- **`app.py`** – FastAPI application with endpoints
- **`services/`** – Core logic for dataset handling, loading, conversion
- **`storage/`** – Storage backends for GCS and local file system
- **`schema.py`** – Request/response models

## Deployment

The `cloudbuild.yaml` handles the build, push to artifact, and deploying / updating service.

```bash
cd preprocessing
gcloud builds submit --config cloudbuild.yaml --ignore-file=.gcloudignore
```

- Cloud Run service (CPU only)
- Environment: `GCS_DATA_BUCKET_NAME` required for GCS storage
- Port: 8080 (default)

## Endpoints

### POST `/datasets/upload`

Upload a dataset file.

**Request:**
Upload a file using multipart/form-data. See API docs for details.

**Response:**

```json
{
  "dataset_name": "your_dataset_name",
  "status": "uploaded"
}
```

### POST `/datasets/process`

Start preprocessing job (supports text and vision datasets).

**Request:**

```json
{
  "dataset_name": "your_dataset_name",
  "dataset_source": "huggingface",
  "dataset_id": "org/repo",
  "dataset_subset": "default",
  "processing_mode": "language_modeling" | "prompt_only",
  "config": {
    /* field mappings and options, see below */
  }
}
```

**Response:**

```json
{
  "dataset_name": "your_dataset_name"
  // OTHER METADATA FIELDS NOT USED BY THE FRONTEND
  // the frontend redirects to the `/datasets/{processed_dataset_id}` endpoint directly
}
```

### GET `/datasets`

List all datasets.

**Response:**

```json
["dataset1", "dataset2"]
```

### GET `/datasets/{processed_dataset_id}`

Get dataset information.

**Response:**

```json
{
  "dataset_name": "your_dataset_name",
  "processed_dataset_id": "uuid",
  "num_rows": 1234,
  "columns": ["col1", "col2", ...],
  "info": {
    "dataset_subset": "default",
    "dataset_source": "upload",
    "dataset_id": "abc123",
    "created_at": "2025-07-25T12:34:56",
    "modality": "text", // or "vision"
    "splits": [
      {
        "split_name": "train",
        "num_rows": 1000,
        "path": "processed_datasets/your_dataset_name/train.parquet",
        "samples": [
          // For vision datasets:
          {
            "messages": [
              { "role": "system", "content": [{ "type": "text", "text": "Describe the images." }] },
              {
                "role": "user",
                "content": [
                  { "type": "text", "text": "Compare these images." },
                  { "type": "image", "image": "data:image/png;base64,..." },
                  { "type": "image", "image": "data:image/png;base64,..." }
                ]
              },
              { "role": "assistant", "content": [{ "type": "text", "text": "The first image shows..." }] }
            ]
          }
          // up to five samples per split...
        ]
      },
      // ... more splits (e.g., "test")
    ]
  }
}
```

### Note

- Modality is returned when you fetch info for a dataset and it is determined by the service and saved to metadata during processing. It is NOT set by the user.

- Samples dict is structured according to the ChatML format and the specific type of the dataset (as well as modality). See the [Conversion Examples](#conversion-examples) section for details.

- **IMPORTANT**: For vision datasets, we use the old format `{"type": "image", "image": "data:image/png;base64,..."}` in the sample response because this is easier for the frontend to parse. The conversion is handled by the backend.

### DELETE `/datasets/{processed_dataset_id}`

Delete a dataset and all associated files.

**Response:**

```json
{
  "dataset_name": "your_dataset_name",
  "deleted": true,
  "message": "Dataset and all associated files deleted successfully.",
  "deleted_files_count": 3,
  "deleted_resources": [
    "processed_datasets/your_dataset_name_processed/train.parquet",
    "processed_datasets/your_dataset_name_processed/test.parquet",
    "raw_datasets/your_dataset_name.csv"
  ]
}
```

### GET `/health`

Health check endpoint.

## Conversion Examples

The only format used in this project is conversation format as opposed to standard format defined in TRL docs. The specific variation of this format used in ChatML. The reason is that this format supports both text-only and multimodal (vision) conversations.

We have standardized to use list format (containing `type`) in `content` field for consistency between vision and text datasets, **even if the dataset is text-only**. The handler is backward compatible but this format is always preferred for consistency.

### Supported Formats & Modes

This service can ingest standard tabular formats (JSON, JSONL, CSV, Parquet, Excel) and HuggingFace datasets. It then preprocesses them into specific conversational formats tailored for different fine-tuning tasks. The desired output is controlled via a `processing_mode` parameter in the process request.

- Language modeling: `system`, `user`, `assistant` roles in a single-turn conversation for `SFTTrainer`

- Prompt-only: `user` role with a prompt, used for policy optimization with trainers like `GRPOTrainer`

- Preference tuning: `user`, `chosen`, `rejected` roles for preference-based algorithms like `DPOTrainer`

> We don't currently support implicitly prompt in preferences which is recommended for Reward Modelling, will do that soon!

The following examples illustrate each row of the processed dataset after conversion. This refers to `dataset[i]` which is `Dict[str, Any]`, but some trainer will index like `dataset["images"]` which is `List[List[PIL.Image]]` for vision datasets.

> IMPORTANT!!! We have now migrated to the new TRL vision dataset format here. Images are stored in a separate `images` field as a list of PIL Image objects. The `content` field in the user message contains placeholders `{"type": "image"}`. This ensures better compatibility with native collators. In the past versions, you will see this as `{"type": "image", "data": "<PIL.Image>"}` instead. It is still backward compatible as for now.

### Language Modeling

```json
{
  "messages": [
    {
      "role": "system",
      "content": [{ "type": "text", "text": "Describe the images." }]
    },
    {
      "role": "user",
      "content": [
        { "type": "text", "text": "Compare these images." },
        { "type": "image" },
        { "type": "image" }
      ]
    },
    {
      "role": "assistant",
      "content": [{ "type": "text", "text": "The first image shows..." }]
    }
  ],
  "images": ["<PIL.Image object>", "<PIL.Image object>"]
}
```

> **Note:** Images are always included in the user message as a list of content items. See below for vision configuration.

### Prompt-only

```json
{
  "prompt": [
    {
      "role": "system",
      "content": [{ "type": "text", "text": "REASONING_PROMPT" }]
    },
    {
      "role": "user",
      "content": [
        { "type": "text", "text": "What color is the sky?" },
        { "type": "image" }
      ]
    }
  ],
  "images": ["<PIL.Image object>"],
  "answer": "10",
  "reasoning": "<think>Some math logic</think>",
  "any_other_additional_fields": "additional_info"
}
```

### Preference

```json
{
  "prompt": [
    {
      "role": "user",
      "content": [
        { "type": "text", "text": "What color is the sky?" },
        { "type": "image" }
      ]
    }
  ],
  "chosen": [
    {
      "role": "assistant",
      "content": [{ "type": "text", "text": "It is blue." }]
    }
  ],
  "rejected": [
    {
      "role": "assistant",
      "content": [{ "type": "text", "text": "It is green." }]
    }
  ],
  "images": ["<PIL.Image object>"]
}
```

## Conversion Configuration

### Language Modeling

Vision processing is automatically enabled when image field mappings are detected. Simply add image field mappings to your configuration (you can include multiple images in a single user message).

```json
{
  "config": {
    "vision_enabled": true,
    "field_mappings": {
      // user_field is either List[Dict] or just Dict
      "user_field": [
        {
          "type": "template",
          "value": "Compare these images and tell me the differences."
        },
        {
          "type": "image",
          "value": "image1"
        },
        {
          "type": "image",
          "value": "image2"
        }
      ],
      "assistant_field": {
        "type": "column",
        "value": "comparison"
      }
    }
  }
}
```

- Images are **always added to user messages only**
- Images are processed in the order they appear in the field_mappings
- Supported image formats: PIL Image objects, base64 strings, file paths, HuggingFace dataset format with `bytes` field

### Prompt-only

```json
{
  "config": {
    "vision_enabled": true,
    "field_mappings": {
      "system_field": {
        "type": "template",
        "value": "This is how you should reason... put stuff in this tag... put answer in..."
      },
      "user_field": [
        {
          "type": "template",
          "value": "What color is the sky?"
        },
        {
          "type": "image",
          "value": "image1"
        }
      ],
      // these additional fields are TEXT-ONLY-FIELDS!
      "answer": {
        "type": "column",
        "value": "answer"
      },
      "reasoning": {
        "type": "column",
        "value": "reasoning"
      },
      "any_other_additional_fields": {
        "type": "column",
        "value": "additional_info"
      }
    }
  }
}
```

### Preference

```json
{
  "config": {
    "vision_enabled": true,
    "field_mappings": {
      "user_field": [
        {
          "type": "template",
          "value": "What color is the sky?"
        },
        {
          "type": "image",
          "value": "image1"
        }
      ],
      "chosen_field": {
        "type": "column",
        "value": "chosen"
      },
      "rejected_field": {
        "type": "column",
        "value": "rejected"
      }
    }
  }
}
```

- `user_field` and `system_field` work the same way as in language modeling, but the `assistant_field` is not used.
- Unlike language modeling, current research indicates that ONLY ONE image is allowed, but the conversion works the same

## Metadata Management

The preprocessing service uses a hybrid storage approach:

- **Dataset Files**: Stored in Google Cloud Storage (or local filesystem) as parquet files
- **Metadata**: Centrally managed in Firestore for consistency and user ownership tracking

Each preprocessed dataset is identified by a unique 8-char UUID based identifier, this field is called `processed_dataset_id`. It is used as the ID for the document and firestore and the folder in GCS. This is different from `dataset_id` which refers to the ID of the **source** dataset, e.g. ID at hugging face or uploaded files.
