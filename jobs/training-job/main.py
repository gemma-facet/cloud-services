import os
import json
import sys
from google.cloud import storage
from providers import TrainingService
from job_manager import JobStateManager, JobTracker
from schema import TrainRequest
from huggingface_hub import login
import logging

GCS_CONFIG_BUCKET = os.getenv("GCS_CONFIG_BUCKET_NAME", "gemma-facet-configs")


def load_training_config(job_id):
    """
    Load the training configuration JSON from GCS for a given job_id.
    """
    gcs_path = os.getenv("TRAINING_CONFIG_GCS_PATH")
    if gcs_path:
        try:
            client = storage.Client()
            if gcs_path.startswith("gs://"):
                bucket_name, blob_name = gcs_path[5:].split("/", 1)
            else:
                raise ValueError("GCS path must start with gs://")
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            config_json = blob.download_as_text()
            return config_json
        except Exception as e:
            logging.error(f"Error downloading config from GCS: {e}", exc_info=True)
            sys.exit(1)
    else:
        default_gcs_path = f"gs://{GCS_CONFIG_BUCKET}/{job_id}.json"
        try:
            client = storage.Client()
            bucket_name, blob_name = default_gcs_path[5:].split("/", 1)
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            if blob.exists():
                config_json = blob.download_as_text()
                return config_json
            else:
                logging.error(f"Config blob does not exist: {default_gcs_path}")
                sys.exit(1)
        except Exception as e:
            logging.error(
                f"Error downloading default config from GCS: {e}", exc_info=True
            )
            sys.exit(1)


def delete_training_config(job_id: str) -> None:
    """
    Delete the training config JSON from GCS for given job_id.
    """
    gcs_path = (
        os.getenv("TRAINING_CONFIG_GCS_PATH")
        or f"gs://{GCS_CONFIG_BUCKET}/{job_id}.json"
    )
    try:
        client = storage.Client()
        if gcs_path.startswith("gs://"):
            bucket_name, blob_name = gcs_path[5:].split("/", 1)
        else:
            logging.warning(f"Invalid GCS path for deletion: {gcs_path}")
            return
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        if blob.exists():
            blob.delete()
            logging.info(f"Deleted config blob: {gcs_path}")
        else:
            logging.warning(f"Config blob not found for deletion: {gcs_path}")
    except Exception as e:
        logging.error(f"Error deleting config from GCS: {e}", exc_info=True)


def main():
    job_id = os.getenv("JOB_ID")
    if not job_id:
        logging.error("Error: JOB_ID environment variable is required")
        sys.exit(1)

    train_config_json = load_training_config(job_id)

    if train_config_json:
        train_request = TrainRequest.model_validate(json.loads(train_config_json))
    else:
        logging.error("Training config not found")
        sys.exit(1)

    if train_request.hf_token:
        login(token=train_request.hf_token)
    else:
        raise ValueError("Huggingface token is required for training")

    project_id = os.environ.get("PROJECT_ID")
    # NOTE: Project ID should always be provided else will fail
    if not project_id:
        raise ValueError(
            "PROJECT_ID environment variable must be set for Firestore client"
        )
    job_manager = JobStateManager(project_id=project_id)

    try:
        # Use JobTracker context manager for automatic status transitions
        with JobTracker(job_manager, job_id) as tracker:
            # Execute training with job tracker (WandB and status handled in training service)
            train_req = TrainRequest.model_validate(train_request)
            service = TrainingService.from_provider(train_req.training_config.provider)
            # We don't need to return anything because they are saved by the tracker internally to firestore already
            _ = service.run_training(
                train_req.processed_dataset_id, train_req.training_config, tracker
            )
    except Exception as e:
        logging.error(f"Training job {job_id} failed: {str(e)}", exc_info=True)
        # Exception handling is done by JobTracker context manager
    finally:
        # Clean up the training config file from GCS
        delete_training_config(job_id)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Uncaught exception in training job: {e}", exc_info=True)
        sys.exit(1)
