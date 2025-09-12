## Export Service

FastAPI service that triggers Cloud Run Jobs to export fine‑tuned Gemma models (adapter, merged, GGUF) and tracks progress in Firestore.

### Highlights
- **AuthN/AuthZ**: Verifies Firebase ID tokens and enforces job ownership.
- **State tracking**: Creates `exports/{export_id}` documents with status and artifacts.
- **Async processing**: Triggers a Cloud Run Job (`EXPORT_JOB_NAME`) to perform the heavy export work.
- **Destinations**: Google Cloud Storage and/or Hugging Face Hub.


## Architecture

- **API**: `FastAPI` app in `app.py`.
- **Auth**: Firebase Admin SDK validates bearer tokens.
- **DB**: Firestore collections
  - `training_jobs/{job_id}`: produced by the Training Service when jobs run/finish
  - `exports/{export_id}`: created by this service to track each export
- **Work executor**: Google Cloud Run Job launched per export with environment overrides
- **Schemas**: Pydantic models in `schema.py`


## Repository layout

- `app.py`: FastAPI application and endpoints
- `schema.py`: Request/response models shared with the job runner
- `Dockerfile`: Container image for the API
- `pyproject.toml`: Python dependencies (managed with `uv`)


## Data model

- `training_jobs` (input reference)
  - Important fields used here: `job_id`, `user_id`, `status`, `adapter_path`, `base_model_id`, `artifacts`
- `exports` (created by this service)
  - `export_id`: UUID string
  - `job_id`: source training job
  - `type`: `adapter | merged | gguf`
  - `destination`: `["gcs" | "hf_hub"]`
  - `hf_repo_id`: optional
  - `status`: `running | completed | failed`
  - `message`: status message
  - `artifacts`: array of `{ type, path, variant }`
  - `started_at`, `updated_at`, `finished_at`


## Authentication and Authorization

- The API requires a bearer token containing a Firebase ID token.
- `get_current_user_id` validates the token via Firebase Admin (`auth.verify_id_token`) and extracts the `uid`.
- Ownership checks: the `user_id` in `training_jobs/{job_id}` must match the caller’s `uid`.


## Environment configuration

- `PROJECT_ID` (required): Google Cloud project for Firestore and Cloud Run
- `REGION` (default: `us-central1`): Cloud Run Jobs region
- `EXPORT_JOB_NAME` (default: `export-job`): Name of the Cloud Run Job to run

Optional (used by the job executor via overrides):
- `HF_TOKEN` (per‑request): Provided to the job when exporting from gated models/HF Hub

Local development can use a `.env` file; environment is loaded via `python-dotenv`.


## API Endpoints

### GET `/health`
Simple liveness check.

Response:
```json
{ "status": "healthy", "service": "export" }
```

### GET `/exports`
List completed training jobs owned by the caller. Useful to choose which job to export next.

Auth: Firebase bearer token required.

Response body: `List[JobSchema]`

Example curl:
```bash
curl -s -H "Authorization: Bearer $FIREBASE_ID_TOKEN" \
  https://<host>/exports | jq
```

### POST `/exports`
Create an export request for a completed job. Validates job ownership and status, then:
1) creates `exports/{export_id}` with `status=running`, 2) triggers Cloud Run Job, 3) returns an acknowledgement.

Auth: Firebase bearer token required.

Request body (ExportRequest):
```json
{
  "job_id": "training_abc123_gemma-2b_def456",
  "export_type": "adapter",
  "destination": ["gcs"],
  "hf_token": null,
  "hf_repo_id": null
}
```

Rules:
- If destination includes `hf_hub`, `hf_repo_id` and `hf_token` are required.
- Only jobs with `status = completed` can be exported.
- Only the job owner can export.

Response (ExportAck):
```json
{
  "success": true,
  "message": "Export job <export_id> started successfully",
  "export_id": "2c72b1fe-..."
}
```

Example curl:
```bash
curl -s -X POST -H "Content-Type: application/json" \
  -H "Authorization: Bearer $FIREBASE_ID_TOKEN" \
  -d '{
        "job_id": "training_abc123_gemma-2b_def456",
        "export_type": "merged",
        "destination": ["gcs"],
        "hf_token": "'$HF_TOKEN'"
      }' \
  https://<host>/exports | jq
```

### GET `/exports/{job_id}`
Fetches job details (ownership enforced) and the latest export record for that `job_id` (if any).

Auth: Firebase bearer token required.

Response (GetExportResponse):
```json
{
  "job_id": "training_abc123_gemma-2b_def456",
  "user_id": "uid_123",
  "adapter_path": "gs://...",
  "base_model_id": "google/gemma-2b",
  "modality": "text",
  "artifacts": { "file": {"adapter": null, "merged": null, "gguf": null}, "raw": {"adapter": null, "merged": null}, "hf": {"adapter": null, "merged": null, "gguf": null} },
  "latest_export": {
    "export_id": "...",
    "job_id": "...",
    "type": "merged",
    "destination": ["gcs"],
    "hf_repo_id": null,
    "status": "running",
    "message": "Export job started",
    "artifacts": [],
    "started_at": "2024-01-01T00:00:00Z",
    "finished_at": null,
    "updated_at": "2024-01-01T00:00:00Z"
  }
}
```


## Cloud Run Job integration

- On POST `/exports`, the service calls Cloud Run Jobs API with overrides:
  - `EXPORT_ID`: the new export document ID
  - `PROJECT_ID`: project to write updates to Firestore
  - `HF_TOKEN`: optional, only if provided in request
- The job identified by `EXPORT_JOB_NAME` must exist in `REGION`.
- The job implementation is responsible for:
  - Performing the export (`adapter`, `merged`, `gguf`)
  - Writing progress and final status to `exports/{export_id}`
  - Uploading artifacts (e.g., to GCS) and recording artifact paths


## Relationship to Training Service

- Training Service produces `training_jobs` and marks jobs `completed` when done.
- This Export Service reads those records to validate ownership and completion, and to gather metadata (e.g., `adapter_path`, `base_model_id`).
- Export outputs may be referenced back by clients alongside training job details.


## Local development

### Prerequisites
- Python 3.12
- Google Cloud credentials with Firestore access
- Firebase Admin SDK credentials (application default credentials or service account)

### Setup
```bash
# From repository root or export/ directory
cd export
uv venv
uv sync --frozen

# .env (example)
cat > .env << 'EOF'
PROJECT_ID=your-gcp-project
REGION=us-central1
EXPORT_JOB_NAME=export-job
EOF

# Run
uv run app.py
# or
uv run uvicorn app:app --host 0.0.0.0 --port 8080
```

### Docker
```bash
cd export
docker build -t export-service:latest .
docker run --rm -p 8080:8080 \
  -e PROJECT_ID=your-gcp-project \
  -e REGION=us-central1 \
  -e EXPORT_JOB_NAME=export-job \
  export-service:latest
```


## Deployment (Google Cloud)

1. Ensure the Cloud Run Job (`EXPORT_JOB_NAME`) exists and can access required resources.
2. Deploy the API to Cloud Run (or another compute target) with env vars: `PROJECT_ID`, `REGION`, `EXPORT_JOB_NAME`.
3. Grant the API service account permissions:
   - Firestore: `datastore.user`
   - Cloud Run Jobs: `run.jobsRunner`
   - If using secret tokens from requests, no additional secret manager is required (token passed as override).

If using Cloud Build, a typical command is:
```bash
gcloud builds submit --project $PROJECT_ID
```


## Permissions

- API service account needs:
  - `roles/datastore.user` (Firestore access)
  - `roles/run.invoker` or `roles/run.admin`/`roles/run.jobsRunner` (to run jobs)
- The job’s service account needs access to:
  - Read Firestore `exports/{export_id}` and write updates
  - Read `training_jobs/{job_id}`
  - Write to GCS bucket for artifacts
  - Optionally access Hugging Face with provided `HF_TOKEN`


## Error handling

- 400: invalid request (e.g., export to `hf_hub` without `hf_repo_id`/`hf_token`; job not `completed`)
- 401: invalid or expired Firebase token
- 403: job does not belong to caller
- 404: job not found
- 500: internal error (auth, Firestore access, job trigger errors)


## Troubleshooting

- "PROJECT_ID environment variable must be set": define `PROJECT_ID` in env or `.env`.
- 401 errors: ensure you pass a valid Firebase ID token in `Authorization: Bearer ...`.
- 403 on export: verify the `training_jobs/{job_id}` belongs to the caller’s `uid`.
- 400 on export: ensure the training job `status` is `completed` and HF params are provided when exporting to `hf_hub`.
- Cloud Run job not found: verify `EXPORT_JOB_NAME` and `REGION` are correct and the job exists.
- Firestore permission denied: ensure the API service account has `roles/datastore.user`.


## FAQ

- What formats are supported?
  - `adapter`, `merged`, and `gguf`.
- Where are artifacts stored?
  - Typically GCS under a project‑defined bucket; HF Hub when destination includes `hf_hub`.
- Does the API block until export completes?
  - No. It returns immediately after scheduling the Cloud Run Job. Poll `GET /exports/{job_id}` for the latest export record.
