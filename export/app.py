import logging
import os
from typing import List
import uuid
import firebase_admin
from firebase_admin import auth
from google.cloud import firestore
from google.cloud import run_v2
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from schema import ExportRequest, ExportAck, JobSchema

app = FastAPI(
    title="Gemma Export Service",
    version="1.0.0",
    description="Export service for exporting models",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

try:
    firebase_admin.initialize_app()
    logging.info("✅ Firebase initialized")
except Exception as e:
    logging.error(f"Failed to initialize Firebase: {e}")
    raise

# Initialize Firestore client
project_id = os.getenv("PROJECT_ID")
if not project_id:
    raise ValueError("PROJECT_ID environment variable must be set for Firestore client")
db = firestore.Client(project=project_id)

# Cloud Run Job configuration
REGION = os.getenv("REGION", "us-central1")
EXPORT_JOB_NAME = "export-job"

logging.info("✅ Export service ready")


bearer_scheme = HTTPBearer()


def get_current_user_id(
    token: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> str:
    """
    Verify Firebase ID token and extract use ID (uid).

    Args:
        token: HTTPAuthorizationCredentials object containing the bearer token

    Returns:
        str: User ID extracted from the token

    Raises:
        HTTPException: 401 if the token is invalid or expired
    """
    try:
        decoded_token = auth.verify_id_token(token.credentials)
        user_id = decoded_token.get("uid")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    except auth.ExpiredIdTokenError:
        logging.error("Failed to verify token: Expired token")
        raise HTTPException(status_code=401, detail="Expired token")
    except auth.InvalidIdTokenError as e:
        logging.error(f"Failed to verify token: Invalid token: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        logging.error(f"Failed to verify token: Unknown error: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error during authentication."
        )


def create_export_document(job_id: str, export_type: str, user_id: str) -> str:
    """
    Create an export document in Firestore and return the export_id.

    Args:
        job_id: The training job ID
        export_type: Type of export (adapter, merged, gguf)
        user_id: User ID who requested the export

    Returns:
        str: The export document ID
    """
    export_id = str(uuid.uuid4())

    export_data = {
        "export_id": export_id,
        "job_id": job_id,
        "type": export_type,
        "status": "running",
        "message": "Export job started",
        "artifacts": [],
        "started_at": firestore.SERVER_TIMESTAMP,
        "updated_at": firestore.SERVER_TIMESTAMP,
    }

    db.collection("exports").document(export_id).set(export_data)
    logging.info(f"Created export document {export_id} for job {job_id}")

    return export_id


@app.get("/health", name="Health Check")
async def health_check():
    return {"status": "healthy", "service": "export"}


@app.get("/exports", response_model=List[JobSchema])
async def get_exports(current_user_id: str = Depends(get_current_user_id)):
    try:
        docs = (
            db.collection("training_jobs")
            .where(filter=firestore.FieldFilter("user_id", "==", current_user_id))
            .where(filter=firestore.FieldFilter("status", "==", "completed"))
            .stream()
        )
        entries = []
        for doc in docs:
            data = doc.to_dict()
            entries.append(JobSchema(**data))
        return entries
    except Exception as e:
        logging.error(f"Failed to get exports: {e}")
        raise HTTPException(status_code=500, detail="Failed to get exports")


@app.post("/exports", response_model=ExportAck, name="Export Model")
async def export(
    request: ExportRequest, current_user_id: str = Depends(get_current_user_id)
):
    # Validate job ownership and completion status
    try:
        doc = db.collection("training_jobs").document(request.job_id).get()
        if not doc.exists:
            logging.error(f"Job {request.job_id} not found")
            raise HTTPException(status_code=404, detail="Job not found")

        job_data = doc.to_dict()
        if not job_data:
            logging.error(f"Job {request.job_id} data is empty")
            raise HTTPException(status_code=404, detail="Job data not found")

        # Check job ownership
        job_user_id = job_data.get("user_id")
        if job_user_id != current_user_id:
            logging.error(
                f"User {current_user_id} attempted to access job {request.job_id} owned by {job_user_id}"
            )
            raise HTTPException(
                status_code=403,
                detail="Access denied: Job does not belong to current user",
            )

        # Check job completion status
        job_status = job_data.get("status")
        if job_status != "completed":
            logging.error(
                f"Job {request.job_id} is not completed (status: {job_status})"
            )
            raise HTTPException(
                status_code=400, detail="Only completed jobs can be exported"
            )

        logging.info(
            f"Export request validated for job {request.job_id} by user {current_user_id}"
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to verify job ownership: {e}")
        raise HTTPException(status_code=500, detail="Failed to verify job ownership")

    # Create export document and trigger job
    try:
        # Create export document in Firestore
        export_id = create_export_document(
            request.job_id, request.export_type, current_user_id
        )

        # Trigger Cloud Run Job
        client = run_v2.JobsClient()
        job_name = f"projects/{project_id}/locations/{REGION}/jobs/{EXPORT_JOB_NAME}"
        run_request = run_v2.RunJobRequest(
            name=job_name,
            overrides=run_v2.RunJobRequest.Overrides(
                container_overrides=[
                    run_v2.RunJobRequest.Overrides.ContainerOverride(
                        env=[
                            run_v2.EnvVar(name="EXPORT_ID", value=export_id),
                            run_v2.EnvVar(name="PROJECT_ID", value=project_id),
                        ]
                        + (
                            [run_v2.EnvVar(name="HF_TOKEN", value=request.hf_token)]
                            if request.hf_token
                            else []
                        )
                    )
                ]
            ),
        )

        _ = client.run_job(request=run_request)
        logging.info(f"Triggered export job {export_id} for job {request.job_id}")

        return ExportAck(
            success=True,
            message=f"Export job {export_id} started successfully",
            export_id=export_id,
        )

    except Exception as e:
        logging.error(f"Failed to start export job for {request.job_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to start export job: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
