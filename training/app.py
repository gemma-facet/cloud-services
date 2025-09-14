import os
import logging
import json
import base64
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Depends, Request
from google.cloud import run_v2
from google.cloud import storage
from datetime import timedelta, datetime, timezone
from schema import (
    TrainRequest,
    JobSubmitResponse,
    JobStatusResponse,
    JobListResponse,
    JobListEntry,
    DownloadUrlResponse,
    JobDeleteResponse,
)
import uvicorn
import hashlib
from job_manager import JobStateManager, JobSchema, JobStatus

app = FastAPI(
    title="Gemma Training Service",
    version="1.0.0",
    description="Training backend (Cloud Run GPU) for fine-tuning LLMs with various methods",
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logging.info("✅ Training service ready")


project_id = os.getenv("PROJECT_ID")
if not project_id:
    raise ValueError("PROJECT_ID environment variable must be set for Firestore client")
job_manager = JobStateManager(project_id)

# These will not be configured in the os envvars because they are pretty much fixed to these two values
REGION = os.getenv("REGION", "us-central1")
JOB_NAME = os.getenv("TRAINING_JOB_NAME", "training-job")
GCS_CONFIG_BUCKET = os.getenv("GCS_CONFIG_BUCKET_NAME", "gemma-facet-configs")
GCS_EXPORT_BUCKET = os.getenv("GCS_EXPORT_BUCKET_NAME", "gemma-facet-models")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "training"}


# Auth dependency for API Gateway
def get_current_user_id(request: Request) -> str:
    """
    Extract user ID from X-Apigateway-Api-Userinfo header set by API Gateway.
    The gateway requires the JWT to contain iss (issuer), sub (subject), aud (audience), iat (issued at), exp (expiration time) claims
    API Gateway will send the authentication result in the X-Apigateway-Api-Userinfo to the backend API whcih contains the base64url encoded content of the JWT payload.
    In this case, the gateway will override the original Authorization header with this X-Apigateway-Api-Userinfo header.

    Args:
        request: FastAPI Request object containing headers

    Returns:
        str: User ID extracted from JWT claims

    Raises:
        HTTPException: 401 if userinfo header is missing or invalid
    """
    userinfo_header = request.headers.get("X-Apigateway-Api-Userinfo")
    if not userinfo_header:
        raise HTTPException(
            status_code=401,
            detail="Missing authentication userinfo. Ensure requests go through API Gateway.",
        )

    try:
        # Decode base64url encoded JWT payload
        # Add padding if needed for proper base64 decoding
        missing_padding = len(userinfo_header) % 4
        if missing_padding:
            userinfo_header += "=" * (4 - missing_padding)

        decoded_bytes = base64.urlsafe_b64decode(userinfo_header)
        claims = json.loads(decoded_bytes.decode("utf-8"))

        user_id = claims.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=401, detail="User ID not found in authentication claims"
            )
        return user_id
    except (json.JSONDecodeError, base64.binascii.Error, UnicodeDecodeError) as e:
        raise HTTPException(
            status_code=401, detail=f"Invalid authentication userinfo format: {str(e)}"
        )


def make_job_id(
    processed_dataset_id: str, base_model_id: str, request: TrainRequest
) -> str:
    """
    Creates a job ID for a training job.

    This has the following properties:
    - IDs are only the same if the processed dataset ID, base model ID, and request are the same
    - IDs are deterministic, i.e. same request will always produce the same ID
    - IDs are short because we truncate the hash to 8 characters

    This is desired because the exact same job will overwrite the previous one and avoid confusion and resource waste.
    This also makes tracking in firestore consistent.

    Args:
        processed_dataset_id (str): processed_dataset_id, this might contain spaces we need to replace them first
        base_model_id (str): The ID of the base model
        request (TrainRequest): The request object

    Returns:
        str: The job ID
    """
    base = f"training_{processed_dataset_id.replace(' ', '-')}_{base_model_id.split('/')[-1]}"
    request_str = json.dumps(request.model_dump(), sort_keys=True)
    short_hash = hashlib.sha256(request_str.encode()).hexdigest()[:8]
    return f"{base}_{short_hash}"


@app.get("/jobs", response_model=JobListResponse)
async def list_jobs(current_user_id: str = Depends(get_current_user_id)):
    """List all jobs with job_id and job_name."""
    try:
        # Query Firestore for jobs owned by user
        docs = job_manager.collection.where("user_id", "==", current_user_id).stream()
        entries = []
        for doc in docs:
            data = doc.to_dict() or {}
            entries.append(
                JobListEntry(
                    job_id=data.get("job_id"),
                    job_name=data.get("job_name"),
                    base_model_id=data.get("base_model_id"),
                    status=data.get("status", "unknown"),
                    modality=data.get("modality", "text"),
                )
            )
        return JobListResponse(jobs=entries)
    except Exception as e:
        logging.error(f"Failed to list jobs: {e}")
        raise HTTPException(status_code=500, detail="Failed to list jobs")


@app.post("/train", response_model=JobSubmitResponse)
async def start_training(
    request: TrainRequest,
    current_user_id: str = Depends(get_current_user_id),
):
    processed_dataset_id = request.processed_dataset_id

    # Verify ownership of the processed dataset before starting the job
    if not job_manager.verify_processed_dataset_ownership(
        processed_dataset_id, current_user_id
    ):
        raise HTTPException(status_code=404, detail="Processed dataset not found")

    base_model_id = request.training_config.base_model_id
    job_id = make_job_id(processed_dataset_id, base_model_id, request)

    if (
        request.training_config.export_config.destination == "hfhub"
        and not request.training_config.export_config.hf_repo_id
    ):
        raise HTTPException(
            status_code=400,
            detail="hf_repo_id is required when export is hfhub",
        )

    # Immediately create job record in Firestore
    job_schema = JobSchema(
        job_id=job_id,
        status=JobStatus.QUEUED.value,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        processed_dataset_id=request.processed_dataset_id,
        base_model_id=request.training_config.base_model_id,
        job_name=request.job_name,
        modality=request.training_config.modality,
        user_id=current_user_id,
    )
    job_manager.ensure_job_document_exists(job_id, job_schema)

    # Upload config to GCS
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_CONFIG_BUCKET)
        blob = bucket.blob(f"{job_id}.json")
        blob.upload_from_string(
            json.dumps(request.model_dump()), content_type="application/json"
        )
        logging.info(
            f"Uploaded training config to gs://{GCS_CONFIG_BUCKET}/{job_id}.json"
        )
    except Exception as e:
        logging.error(f"Failed to upload config to GCS: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to upload config to GCS: {str(e)}"
        )

    try:
        client = run_v2.JobsClient()
        job_name = f"projects/{project_id}/locations/{REGION}/jobs/{JOB_NAME}"
        run_request = run_v2.RunJobRequest(
            name=job_name,
            overrides=run_v2.RunJobRequest.Overrides(
                container_overrides=[
                    run_v2.RunJobRequest.Overrides.ContainerOverride(
                        env=[run_v2.EnvVar(name="JOB_ID", value=job_id)]
                    )
                ]
            ),
        )
        _ = client.run_job(request=run_request)

        # NOTE: We don't need to wait for the job to complete
        # because the job will be tracked in firestore
        # If needed you can do response = operation.result()

        return JobSubmitResponse(job_id=job_id)
    except Exception as e:
        logging.error(f"Failed to start training job: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to start training job: {str(e)}"
        )


@app.get("/jobs/download/{job_id}", response_model=DownloadUrlResponse)
async def download_gguf_file(
    job_id: str,
    current_user_id: str = Depends(get_current_user_id),
):
    """
    Generate a signed URL for downloading the GGUF file of a specific job.
    This is a convenience endpoint that automatically finds the GGUF file path from job status.
    """
    try:
        # Verify ownership
        if not job_manager.verify_job_ownership(job_id, current_user_id):
            raise HTTPException(status_code=404, detail="Job not found")
        # Get job status to find GGUF path
        job_data = job_manager.get_job_status_dict(job_id)
        if not job_data:
            raise HTTPException(status_code=404, detail="Job not found")

        gguf_path = job_data.get("gguf_path")
        if not gguf_path:
            raise HTTPException(
                status_code=404,
                detail="No GGUF file available for this job. Check if include_gguf was enabled during training.",
            )

        # Extract the blob path from the GCS URL
        # gguf_path format: gs://bucket/gguf_models/job_123/model.gguf
        if not gguf_path.startswith("gs://"):
            raise HTTPException(status_code=500, detail="Invalid GGUF path format")

        # Remove gs://bucket/ prefix to get blob path
        path_parts = gguf_path.replace("gs://", "").split("/", 1)
        if len(path_parts) != 2:
            raise HTTPException(status_code=500, detail="Invalid GGUF path format")

        bucket_name, blob_path = path_parts

        # Verify it's the expected bucket
        if bucket_name != GCS_EXPORT_BUCKET:
            raise HTTPException(
                status_code=500, detail="GGUF file in unexpected bucket"
            )

        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_EXPORT_BUCKET)
        blob = bucket.blob(blob_path)

        if not blob.exists():
            raise HTTPException(
                status_code=404, detail="GGUF file not found in storage"
            )

        # Generate a signed URL that is valid for 1 hour
        signed_url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=30),
            method="GET",
        )

        return DownloadUrlResponse(download_url=signed_url)

    except HTTPException:
        raise
    except Exception as e:
        logging.error(
            f"Failed to generate GGUF download URL for job {job_id}: {str(e)}"
        )
        raise HTTPException(status_code=500, detail=str(e))


# NOTE: In the future we might want to add /jobs/{job_id}/download/model endpoint for adapter and merged models


def delete_gcs_resources(
    job_id: str, adapter_path: Optional[str], gguf_path: Optional[str]
) -> List[str]:
    """
    Delete GCS resources associated with a job.

    Args:
        job_id: Job identifier
        adapter_path: GCS path to adapter/merged model
        gguf_path: GCS path to GGUF model

    Returns:
        List of deleted resource paths
    """
    deleted_resources = []
    storage_client = storage.Client()

    # Delete config file
    try:
        config_bucket = storage_client.bucket(GCS_CONFIG_BUCKET)
        config_blob = config_bucket.blob(f"{job_id}.json")
        if config_blob.exists():
            config_blob.delete()
            deleted_resources.append(f"gs://{GCS_CONFIG_BUCKET}/{job_id}.json")
            logging.info(f"Deleted config file: gs://{GCS_CONFIG_BUCKET}/{job_id}.json")
    except Exception as e:
        logging.warning(f"Failed to delete config file for job {job_id}: {e}")

    # Delete adapter/merged model
    if adapter_path:
        try:
            if adapter_path.startswith("gs://"):
                # Parse GCS path
                path_parts = adapter_path.replace("gs://", "").split("/", 1)
                if len(path_parts) == 2:
                    bucket_name, blob_prefix = path_parts
                    bucket = storage_client.bucket(bucket_name)

                    # Delete all blobs with this prefix (handles directories)
                    blobs = bucket.list_blobs(prefix=blob_prefix)
                    deleted_count = 0
                    for blob in blobs:
                        blob.delete()
                        deleted_count += 1

                    if deleted_count > 0:
                        deleted_resources.append(
                            f"gs://{bucket_name}/{blob_prefix} ({deleted_count} files)"
                        )
                        logging.info(
                            f"Deleted adapter resources: gs://{bucket_name}/{blob_prefix} ({deleted_count} files)"
                        )
        except Exception as e:
            logging.warning(f"Failed to delete adapter resources for job {job_id}: {e}")

    # Delete GGUF model
    if gguf_path:
        try:
            if gguf_path.startswith("gs://"):
                # Parse GCS path
                path_parts = gguf_path.replace("gs://", "").split("/", 1)
                if len(path_parts) == 2:
                    bucket_name, blob_path = path_parts
                    bucket = storage_client.bucket(bucket_name)
                    blob = bucket.blob(blob_path)

                    if blob.exists():
                        blob.delete()
                        deleted_resources.append(gguf_path)
                        logging.info(f"Deleted GGUF file: {gguf_path}")
        except Exception as e:
            logging.warning(f"Failed to delete GGUF file for job {job_id}: {e}")

    return deleted_resources


@app.delete("/jobs/{job_id}", response_model=JobDeleteResponse)
async def delete_job(
    job_id: str,
    current_user_id: str = Depends(get_current_user_id),
):
    """
    Delete a job and all associated GCS resources.
    This includes the job metadata, config file, adapter/merged model, and GGUF file if available.
    """
    try:
        # Verify ownership
        if not job_manager.verify_job_ownership(job_id, current_user_id):
            raise HTTPException(status_code=404, detail="Job not found")

        # First, get job data to find associated resources
        job_data = job_manager.get_job_status_dict(job_id)
        if not job_data:
            raise HTTPException(status_code=404, detail="Job not found")

        adapter_path = job_data.get("adapter_path")
        gguf_path = job_data.get("gguf_path")

        # Delete GCS resources
        deleted_resources = delete_gcs_resources(job_id, adapter_path, gguf_path)

        # Delete job metadata from Firestore
        job_deleted = job_manager.delete_job(job_id)

        if job_deleted:
            message = f"Successfully deleted job {job_id} and associated resources"
            logging.info(message)
        else:
            message = f"Job {job_id} metadata was not found, but cleaned up any existing resources"
            logging.warning(message)

        return JobDeleteResponse(
            job_id=job_id,
            deleted=True,
            message=message,
            deleted_resources=deleted_resources,
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to delete job {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete job: {str(e)}")


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    current_user_id: str = Depends(get_current_user_id),
):
    # Verify ownership
    if not job_manager.verify_job_ownership(job_id, current_user_id):
        raise HTTPException(status_code=404, detail="Job not found")

    job_data = job_manager.get_job_status_dict(job_id)
    if not job_data:
        logging.error(f"Job not found: {job_id}")
        raise HTTPException(status_code=404, detail="Job not found")

    return job_data


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
