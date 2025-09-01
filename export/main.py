import logging
import os
import json
import base64
from google.cloud import firestore
from fastapi import FastAPI, HTTPException, Depends, Request
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

# Initialize Firestore client
project_id = os.getenv("PROJECT_ID")
if not project_id:
    raise ValueError("PROJECT_ID environment variable must be set for Firestore client")
db = firestore.Client(project=project_id)

logging.info("✅ Export service ready")


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
    # First check if the job_id exists in the database and belongs to the current user
    try:
        doc = db.collection("training_jobs").document(request.job_id).get()
        if not doc.exists:
            logging.error(f"Job {request.job_id} not found")
            raise HTTPException(status_code=404, detail="Job not found")

        job_data = doc.to_dict()
        if not job_data:
            logging.error(f"Job {request.job_id} data is empty")
            raise HTTPException(status_code=404, detail="Job data not found")

        # Check if the job belongs to the current user
        job_user_id = job_data.get("user_id")
        if job_user_id != current_user_id:
            logging.error(
                f"User {current_user_id} attempted to access job {request.job_id} owned by {job_user_id}"
            )
            raise HTTPException(
                status_code=403,
                detail="Access denied: Job does not belong to current user",
            )

        # Check if the job is completed (only completed jobs should be exportable)
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

    # NOW IT IS CONFIRMED THAT THE JOB EXISTS AND BELONGS TO THE CURRENT USER AND IS COMPLETED
    # SO WE CAN EXPORT THE MODEL

    # If user only wants adapters, check if it's already available in the database
    if request.export_type == "adapter":
        adapter_path = job_data.get("adapter_path")
        if adapter_path:
            logging.info(
                f"Adapter path found in database for job {request.job_id}: {adapter_path}"
            )
            return ExportResponse(
                success=True,
                job_id=request.job_id,
                export_type=request.export_type,
                adapter_path=adapter_path,
            )
        else:
            logging.warning(
                f"Adapter path not found in database for job {request.job_id}"
            )
            raise HTTPException(
                status_code=404,
                detail="Adapter not found for this job. The training may not have completed successfully.",
            )

    elif request.export_type == "merged":
        merged_path = job_data.get("merged_path")
        if merged_path:
            logging.info(
                f"Merged path found in database for job {request.job_id}: {merged_path}"
            )
            return ExportResponse(
                success=True,
                job_id=request.job_id,
                export_type=request.export_type,
                merged_path=merged_path,
            )
        else:
            logging.info(f"Merged path not found in database for job {request.job_id}")
            # Need to merge the model
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

                # Merge the model using utility function
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
                f"GGUF path found in database for job {request.job_id}: {gguf_path}"
            )
            return ExportResponse(
                success=True,
                job_id=request.job_id,
                export_type=request.export_type,
                gguf_path=gguf_path,
            )
        else:
            logging.info(f"GGUF path not found in database for job {request.job_id}")
            # GGUF conversion is not yet fully implemented
            raise HTTPException(
                status_code=501,
                detail="GGUF conversion is not yet implemented. Please use 'merged' export type instead.",
            )

    # If we reach here, it means no specific export type was handled
    # This shouldn't happen due to the Literal type constraint, but let's handle it gracefully
    raise HTTPException(
        status_code=400,
        detail=f"Unsupported export type: {request.export_type}",
    )
