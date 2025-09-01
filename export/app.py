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
