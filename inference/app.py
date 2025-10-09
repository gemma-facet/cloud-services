import os
import logging
import firebase_admin
from firebase_admin import auth
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.concurrency import run_in_threadpool
from google.cloud import firestore
from huggingface_hub import login
from schema import (
    InferenceRequest,
    InferenceResponse,
    BatchInferenceRequest,
    BatchInferenceResponse,
    EvaluationRequest,
    EvaluationResponse,
)
from base import run_inference, run_batch_inference, run_evaluation
from typing import Optional

app = FastAPI(
    title="Gemma Inference Service",
    version="1.0.0",
    description="Inference service for running inference on trained models",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Initialize Firebase Admin SDK
# This will use the GOOGLE_APPLICATION_CREDENTIALS environment variable
# or the default service account when running in a Google Cloud environment.
try:
    firebase_admin.initialize_app()
    logging.info("✅ Firebase Admin SDK initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize Firebase Admin SDK: {e}")

# Initialize Firestore client
project_id = os.getenv("PROJECT_ID")
if not project_id:
    raise ValueError("PROJECT_ID environment variable must be set for Firestore client")
# By default uses (default) database, but can specify in env var e.g. staging
database_name = os.getenv("FIRESTORE_DB")
db = firestore.Client(project=project_id, database=database_name)


logging.info("✅ Inference service ready")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "inference"}


# Security scheme for Bearer token
bearer_scheme = HTTPBearer()


def get_current_user_id(
    token: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> str:
    """
    Verify Firebase ID token and extract user ID (uid).
    This function is a FastAPI dependency that can be used to protect endpoints.

    Args:
        token: The HTTPAuthorizationCredentials containing the bearer token.

    Returns:
        str: The user's unique ID (uid) from the verified Firebase token.

    Raises:
        HTTPException: 401 if the token is missing, invalid, or expired.
    """
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Bearer token not provided",
        )
    try:
        # Verify the token against the Firebase Auth API.
        decoded_token = auth.verify_id_token(token.credentials)
        user_id = decoded_token.get("uid")
        if not user_id:
            raise HTTPException(status_code=401, detail="User ID not found in token")
        return user_id
    except auth.ExpiredIdTokenError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except auth.InvalidIdTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during token verification: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error during authentication"
        )


def login_hf(hf_token: Optional[str]):
    """
    Login to Hugging Face.
    For now we support env variable for dev but in prod we will just raise an error.
    Login is required for pushing and pulling models since Gemma3 is a gated model.
    """
    token = hf_token
    if token:
        login(token=token)
        logging.info("Logged into Hugging Face")
    else:
        logging.warning("HF Token not provided. Hugging Face login skipped.")


@app.post("/inference", response_model=InferenceResponse)
async def inference(
    request: InferenceRequest,
    current_user_id: str = Depends(get_current_user_id),
):
    """Run inference using a trained model"""
    try:
        login_hf(request.hf_token)
        output = await run_in_threadpool(
            run_inference,
            request.model_source,
            request.model_type,
            request.base_model_id,
            request.prompt,
            request.use_vllm,
        )
        return {"result": output}
    except FileNotFoundError:
        logging.error(f"Model {request.model_source} not found")
        raise HTTPException(status_code=404, detail="Model not found")
    except Exception as e:
        logging.error(f"Inference failed with error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inference/batch", response_model=BatchInferenceResponse)
async def batch_inference(
    request: BatchInferenceRequest,
    current_user_id: str = Depends(get_current_user_id),
):
    """Run batch inference using a trained model"""
    messages = request.messages
    if not messages or not isinstance(messages, list) or len(messages) == 0:
        raise HTTPException(status_code=400, detail="messages (list) is required")
    try:
        login_hf(request.hf_token)
        outputs = await run_in_threadpool(
            run_batch_inference,
            request.model_source,
            request.model_type,
            request.base_model_id,
            messages,
            request.use_vllm,
        )
        return {"results": outputs}
    except FileNotFoundError:
        logging.error(f"Model {request.model_source} not found")
        raise HTTPException(status_code=404, detail="Model not found")
    except Exception as e:
        logging.error(f"Batch inference failed with error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluation", response_model=EvaluationResponse)
async def evaluation(
    request: EvaluationRequest,
    current_user_id: str = Depends(get_current_user_id),
):
    """Run evaluation of a fine-tuned model on a dataset"""
    try:
        # Verify dataset ownership before evaluating
        try:
            doc = db.collection("processed_datasets").document(request.dataset_id).get()
            if not doc.exists or doc.to_dict().get("user_id") != current_user_id:
                raise HTTPException(status_code=404, detail="Dataset not found")
        except HTTPException:
            raise
        except Exception as e:
            logging.error(f"Failed to verify dataset ownership: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to verify dataset ownership"
            )

        login_hf(request.hf_token)
        result = await run_in_threadpool(
            run_evaluation,
            request.model_source,
            request.model_type,
            request.base_model_id,
            request.dataset_id,
            request.task_type,
            request.metrics,
            request.max_samples,
            request.num_sample_results or 3,
            request.use_vllm,
        )
        return {
            "metrics": result["metrics"],
            "samples": result["samples"],
            "num_samples": result["num_samples"],
            "dataset_id": result["dataset_id"],
        }
    except FileNotFoundError as e:
        logging.error(f"Resource not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logging.error(f"Invalid request: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Evaluation failed with error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
