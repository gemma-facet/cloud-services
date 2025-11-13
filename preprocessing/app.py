import os
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from storage import GCSStorageManager, LocalStorageManager
from services.dataset_service import DatasetService
from dataset_tracker import DatasetTracker
import base64
import json
from schema import (
    DatasetUploadResponse,
    PreprocessingRequest,
    ProcessingResult,
    DatasetsInfoResponse,
    DatasetInfoResponse,
    DatasetDeleteResponse,
    DatasetSynthesizeResponse,
    MIME_TYPES,
)

project_id = os.getenv("PROJECT_ID")
database_name = os.getenv("FIRESTORE_DB", None)
dataset_tracker = DatasetTracker(project_id, database_name=database_name)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Gemma Dataset Preprocessing Service",
    version="2.0.0",
    description="A modular service for preprocessing datasets into ChatML format",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

storage_type = os.getenv("STORAGE_TYPE", "gcs")  # "gcs" or "local"

if storage_type == "gcs":
    bucket_name = os.getenv("GCS_DATA_BUCKET_NAME", "gemma-dataset-bucket")
    storage_manager = GCSStorageManager(bucket_name)
    logger.info(f"Using GCS storage with bucket: {bucket_name}")
else:
    data_path = os.getenv("LOCAL_DATA_PATH", "./data")
    storage_manager = LocalStorageManager(data_path)
    logger.info(f"Using local storage at: {data_path}")

dataset_service = DatasetService(storage_manager)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "preprocessing"}


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


@app.post("/datasets/upload", response_model=DatasetUploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    current_user_id: str = Depends(get_current_user_id),
):
    """Upload a dataset file to storage"""
    try:
        file_content = await file.read()
        filename = file.filename or ""
        filetype = filename.rsplit(".", 1)[1].lower() if "." in filename else ""
        content_type = MIME_TYPES.get(filetype, "application/octet-stream")
        result = dataset_service.upload_dataset(
            file_data=file_content,
            filename=file.filename or "unknown",
            metadata={"content_type": content_type, "user_id": current_user_id},
        )
        # Track raw dataset metadata
        raw_metadata = {
            "dataset_id": result.dataset_id,
            "gcs_path": result.gcs_path,
            "user_id": current_user_id,
            "filename": result.filename,
            "content_type": content_type or "unknown",
            "size_bytes": result.size_bytes,
        }
        dataset_tracker.track_raw_dataset(raw_metadata)

        return result

    except Exception as e:
        logger.error(f"Error uploading dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/datasets/synthesize", response_model=DatasetSynthesizeResponse)
def synthesize_dataset(
    file: UploadFile = File(...),
    current_user_id: str = Depends(get_current_user_id),
):
    """Synthesize a dataset using the dataset synthesizer service"""
    try:
        file_content = file.file.read()
        filename = file.filename or ""
        dataset,storage_path = dataset_service.synthesize_dataset(
            file_data=file_content,
            filename=filename,
            metadata={"user_id": current_user_id},
            )
        return {
            "filename": filename,
            "gcs_path": storage_path,
        }

    except Exception as e:
        logger.error(f"Error synthesizing dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")

@app.post("/datasets/process", response_model=ProcessingResult)
def process_dataset(
    request: PreprocessingRequest,
    current_user_id: str = Depends(get_current_user_id),
):
    """Process a dataset into ChatML format

    The processing converts the dataset to ChatML format using the provided configuration.
    The configuration can include field mappings that specify either direct column mappings or template strings
    with column references.

    Example field mappings:
    ```python
    {
        "system_field": {"type": "template", "value": "You are a helpful assistant."},
        "user_field": {"type": "column", "value": "question"},
        "assistant_field": {"type": "template", "value": "Answer: {answer}"}
    }
    ```
    """
    try:
        # For uploaded datasets, verify ownership of the raw dataset
        if request.dataset_source == "upload":
            if not dataset_tracker.verify_raw_dataset_ownership(
                request.dataset_id, current_user_id
            ):
                raise HTTPException(
                    status_code=404,
                    detail="Source dataset not found or not owned by user",
                )

        result = dataset_service.process_dataset(
            dataset_name=request.dataset_name,
            dataset_source=request.dataset_source,
            dataset_id=request.dataset_id,
            dataset_subset=request.dataset_subset,
            processing_mode=request.processing_mode,
            config=request.config,
        )

        # Use the complete metadata from the result for Firestore storage
        # Note: result.full_metadata contains the complete splits info as List[Dict]
        processed_metadata = {
            "processed_dataset_id": result.processed_dataset_id,
            "dataset_name": request.dataset_name,
            "user_id": current_user_id,
            "dataset_id": request.dataset_id,
            "dataset_source": request.dataset_source,
            "dataset_subset": request.dataset_subset,
            "created_at": result.created_at,
            "num_examples": result.num_examples,
            "splits": result.full_splits,
            "modality": result.modality,
        }
        dataset_tracker.track_processed_dataset(processed_metadata)

        return result

    except Exception as e:
        logger.error(f"Error processing dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/datasets", response_model=DatasetsInfoResponse)
def get_datasets_info(
    current_user_id: str = Depends(get_current_user_id),
):
    """
    Get information about all the processed datasets owned by the user.
    """
    try:
        return dataset_service.get_datasets_info(current_user_id, dataset_tracker)
    except Exception as e:
        logger.error(f"Error getting datasets info: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get datasets info: {str(e)}"
        )


@app.get("/datasets/{processed_dataset_id}", response_model=DatasetInfoResponse)
def get_dataset_info(
    processed_dataset_id: str,
    current_user_id: str = Depends(get_current_user_id),
):
    """
    Get information about a dataset using its unique processed dataset ID.
    """
    try:
        # verify ownership of processed dataset
        if not dataset_tracker.verify_processed_dataset_ownership(
            processed_dataset_id, current_user_id
        ):
            raise HTTPException(status_code=404, detail="Dataset not found")
        return dataset_service.get_dataset_info(processed_dataset_id, dataset_tracker)
    except Exception as e:
        logger.error(f"Error getting dataset info: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get dataset info: {str(e)}"
        )


@app.delete("/datasets/{processed_dataset_id}", response_model=DatasetDeleteResponse)
def delete_dataset(
    processed_dataset_id: str,
    current_user_id: str = Depends(get_current_user_id),
):
    """
    Delete a dataset and all associated files using its unique processed dataset ID.
    NOTE: This only deletes preprocessed dataset, NOT raw dataset files!
    """
    try:
        deleted_resources = []
        total_deleted_files = 0

        # verify ownership of processed dataset
        if not dataset_tracker.verify_processed_dataset_ownership(
            processed_dataset_id, current_user_id
        ):
            raise HTTPException(
                status_code=404, detail="Dataset not found or not owned by user"
            )

        # Get dataset name from Firestore metadata for response
        try:
            processed_metadata = dataset_tracker.get_processed_dataset_metadata(
                processed_dataset_id
            )
            dataset_name = (
                processed_metadata["dataset_name"]
                if processed_metadata
                else processed_dataset_id
            )
        except Exception:
            dataset_name = (
                processed_dataset_id  # fallback to ID if metadata unavailable
            )

        # Delete processed dataset (parquet files only, metadata is in Firestore)
        processed_prefix = f"processed_datasets/{processed_dataset_id}"
        processed_files = storage_manager.list_files(prefix=processed_prefix)
        if processed_files:
            deleted_count = storage_manager.delete_directory(processed_prefix)
            total_deleted_files += deleted_count
            deleted_resources.append(
                f"Processed dataset: {processed_prefix} ({deleted_count} files)"
            )

        if total_deleted_files > 0:
            message = f"Successfully deleted dataset '{dataset_name}' and all associated files"
            logger.info(message)
        else:
            message = f"Dataset '{dataset_name}' not found or already deleted"
            logger.warning(message)

        # remove metadata for processed dataset
        if total_deleted_files > 0:
            dataset_tracker.delete_processed_dataset_metadata(processed_dataset_id)
        return DatasetDeleteResponse(
            dataset_name=dataset_name,
            deleted=total_deleted_files > 0,
            message=message,
            deleted_files_count=total_deleted_files,
            deleted_resources=deleted_resources,
        )

    except Exception as e:
        logger.error(f"Failed to delete dataset {dataset_name}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to delete dataset: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
