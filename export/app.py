import logging
import os
import firebase_admin
from firebase_admin import auth
from google.cloud import firestore
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.concurrency import run_in_threadpool
from schema import ExportRequest, ExportResponse
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

    # Handle adapter export
    if request.export_type == "adapter":
        adapter_path = job_data.get("adapter_path")
        if adapter_path:
            logging.info(
                f"Adapter export: returning existing path for job {request.job_id}"
            )
            return ExportResponse(
                success=True,
                job_id=request.job_id,
                export_type=request.export_type,
                adapter_path=adapter_path,
            )
        else:
            logging.warning(f"Adapter path not found for job {request.job_id}")
            raise HTTPException(
                status_code=404,
                detail="Adapter not found for this job. The training may not have completed successfully.",
            )

    elif request.export_type == "merged":
        merged_path = job_data.get("merged_path")
        if merged_path:
            logging.info(
                f"Merged export: returning existing path for job {request.job_id}"
            )
            return ExportResponse(
                success=True,
                job_id=request.job_id,
                export_type=request.export_type,
                merged_path=merged_path,
            )
        else:
            logging.info(f"Creating merged model for job {request.job_id}")
            # Create merged model from adapter
            try:
                adapter_path = job_data.get("adapter_path")
                if not adapter_path:
                    raise HTTPException(
                        status_code=404,
                        detail="Adapter not found for this job. Cannot create merged model.",
                    )

                base_model_id = job_data.get("base_model_id")
                if not base_model_id:
                    raise HTTPException(
                        status_code=400, detail="Base model ID not found for this job."
                    )

                # Login to Hugging Face first
                login_hf(request.hf_token)

                # Merge the model
                merged_path = await run_in_threadpool(
                    export_utils._merge_model,
                    adapter_path,
                    base_model_id,
                    request.job_id,
                    request.hf_token,
                )

                return ExportResponse(
                    success=True,
                    job_id=request.job_id,
                    export_type=request.export_type,
                    merged_path=merged_path,
                )

            except Exception as e:
                logging.error(
                    f"Failed to merge model for job {request.job_id}: {str(e)}"
                )
                raise HTTPException(
                    status_code=500, detail=f"Failed to merge model: {str(e)}"
                )

    elif request.export_type == "gguf":
        gguf_path = job_data.get("gguf_path")
        if gguf_path:
            logging.info(
                f"GGUF export: returning existing path for job {request.job_id}"
            )
            return ExportResponse(
                success=True,
                job_id=request.job_id,
                export_type=request.export_type,
                gguf_path=gguf_path,
            )
        else:
            logging.info(f"Starting GGUF conversion for job {request.job_id}")
            # GGUF Export Flow:
            # 1. Check if merged model exists in Firestore
            # 2. If yes: download from GCS and convert to GGUF
            # 3. If no: check if adapter exists, merge model, then convert to GGUF
            try:
                # First check if we have a merged model, if not create one
                merged_path = job_data.get("merged_path")

                if merged_path:
                    logging.info(
                        f"Downloading existing merged model for GGUF conversion: {merged_path}"
                    )
                    # Download existing merged model and convert to GGUF
                    gguf_path = await run_in_threadpool(
                        export_utils._convert_to_gguf,
                        merged_path,
                        request.job_id,
                        None,  # No local path, will download from GCS
                    )
                else:
                    logging.info("Creating merged model first for GGUF conversion")
                    # Create merged model first, then convert to GGUF
                    adapter_path = job_data.get("adapter_path")
                    if not adapter_path:
                        raise HTTPException(
                            status_code=404,
                            detail="Adapter not found for this job. Cannot create GGUF model.",
                        )

                    base_model_id = job_data.get("base_model_id")
                    if not base_model_id:
                        raise HTTPException(
                            status_code=400,
                            detail="Base model ID not found for this job.",
                        )

                    # Login to Hugging Face first
                    login_hf(request.hf_token)

                    # Merge the model first (don't clean up temp files for GGUF conversion)
                    merged_path = await run_in_threadpool(
                        export_utils._merge_model,
                        adapter_path,
                        base_model_id,
                        request.job_id,
                        request.hf_token,
                        False,  # cleanup_temp=False
                    )

                    logging.info("Converting local merged model to GGUF")
                    # Convert to GGUF using the local merged model
                    gguf_path = await run_in_threadpool(
                        export_utils._convert_to_gguf,
                        merged_path,
                        request.job_id,
                        "merged_model",  # Local path to the merged model we just created
                    )

                return ExportResponse(
                    success=True,
                    job_id=request.job_id,
                    export_type=request.export_type,
                    gguf_path=gguf_path,
                )

            except Exception as e:
                logging.error(
                    f"Failed to convert to GGUF for job {request.job_id}: {str(e)}"
                )
                raise HTTPException(
                    status_code=500, detail=f"Failed to convert to GGUF: {str(e)}"
                )

    # If we reach here, it means no specific export type was handled
    # This shouldn't happen due to the Literal type constraint, but let's handle it gracefully
    raise HTTPException(
        status_code=400,
        detail=f"Unsupported export type: {request.export_type}",
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
