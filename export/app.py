import logging
import os
import firebase_admin
from firebase_admin import auth
from google.cloud import firestore
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.concurrency import run_in_threadpool
from schema import ExportRequest, ExportResponse, ExportInfo
from huggingface_hub import login
from typing import Optional
from utils import export_utils

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


def login_hf(hf_token: Optional[str]):
    """
    Login to Hugging Face.
    Login is required for pushing and pulling models since Gemma models are gated.
    """
    if hf_token:
        login(token=hf_token)
        logging.info("Logged into Hugging Face")
    else:
        logging.warning("HF Token not provided. Hugging Face login skipped.")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "export"}


@app.post("/export", response_model=ExportResponse)
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

    # Handle export based on type
    try:
        # Get export data from Firestore
        export_data = job_data.get("export", {})

        # Check if export already exists
        existing_export_path = export_data.get(request.export_type)
        if existing_export_path:
            logging.info(
                f"{request.export_type.title()} export: returning existing path for job {request.job_id}"
            )
            return ExportResponse(
                success=True,
                job_id=request.job_id,
                export=ExportInfo(
                    type=request.export_type,
                    path=existing_export_path,
                ),
            )

        # Export doesn't exist, need to create it
        logging.info(f"Creating {request.export_type} export for job {request.job_id}")

        # Get required data from job document
        adapter_path = job_data.get("adapter_path")
        merged_path = job_data.get("merged_path")
        base_model_id = job_data.get("base_model_id")

        # Login to Hugging Face if token provided
        login_hf(request.hf_token)

        # Handle different export types
        if request.export_type == "adapter":
            if not adapter_path:
                raise HTTPException(
                    status_code=404,
                    detail="Adapter not found for this job. The training may not have completed successfully.",
                )

            export_path = await run_in_threadpool(
                export_utils._export_adapter, request.job_id, adapter_path
            )

        elif request.export_type == "merged":
            if not adapter_path:
                raise HTTPException(
                    status_code=404,
                    detail="Adapter not found for this job. Cannot create merged model.",
                )

            export_path = await run_in_threadpool(
                export_utils._export_merged,
                request.job_id,
                merged_path,
                adapter_path,
                base_model_id,
                request.hf_token,
            )

        elif request.export_type == "gguf":
            if not adapter_path:
                raise HTTPException(
                    status_code=404,
                    detail="Adapter not found for this job. Cannot create GGUF model.",
                )

            export_path = await run_in_threadpool(
                export_utils._export_gguf,
                request.job_id,
                merged_path,
                adapter_path,
                base_model_id,
                request.hf_token,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported export type: {request.export_type}",
            )

        return ExportResponse(
            success=True,
            job_id=request.job_id,
            export=ExportInfo(
                type=request.export_type,
                path=export_path,
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(
            f"Failed to export {request.export_type} for job {request.job_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to export {request.export_type}: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
