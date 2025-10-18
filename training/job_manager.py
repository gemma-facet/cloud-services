import logging
import math
from enum import Enum
from typing import Optional, Dict, Any
from google.cloud import firestore
from datetime import datetime, timezone
from schema import JobSchema, JobStatusResponse


class JobStatus(Enum):
    """Enum for job status values"""

    QUEUED = "queued"
    PREPARING = "preparing"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"


class JobStateManager:
    """
    Centralized job state management using Firestore using repository pattern.
    Provides clean API for tracking training job progress.
    """

    def __init__(
        self,
        project_id: str,
        collection_name: str = "training_jobs",
        database_name: Optional[str] = None,
    ):
        """
        Initialize job state manager.

        Args:
            project_id: Google Cloud project ID
            collection_name: Name of the Firestore collection for jobs
        """
        self.db = firestore.Client(project=project_id, database=database_name)
        self.collection = self.db.collection(collection_name)
        self.logger = logging.getLogger(__name__)

    def get_job(self, job_id: str) -> Optional[JobSchema]:
        """
        Retrieve job metadata by ID.

        Args:
            job_id: Job identifier

        Returns:
            JobSchema or None if job not found
        """
        try:
            doc = self.collection.document(job_id).get()
            if not doc.exists:
                return None
            data = doc.to_dict()
            if not data:
                return None

            # Pad missing fields for backward compatibility
            if "artifacts" not in data:
                data["artifacts"] = {}

            return JobSchema(**data)
        except Exception as e:
            self.logger.error(f"Failed to get job {job_id}: {e}")
            raise

    def get_job_status_dict(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job status as dictionary for API responses.

        Args:
            job_id: Job identifier

        Returns:
            Dictionary with job status info or None if not found
        """
        job = self.get_job(job_id)
        if not job:
            return None

        # Metrics contain numerical values and they might be nan but JSON response does not support these values
        metrics = (
            self._sanitize_metrics(job.metrics.model_dump()) if job.metrics else None
        )

        response = JobStatusResponse(
            job_name=job.job_name,
            status=job.status,
            modality=job.modality,
            wandb_url=job.wandb_url,
            processed_dataset_id=job.processed_dataset_id,
            base_model_id=job.base_model_id,
            artifacts=job.artifacts,
            metrics=metrics,
            error=job.error,
        )
        return response.model_dump(exclude_none=True)

    def ensure_job_document_exists(
        self, job_id: str, job_metadata: Optional[JobSchema] = None
    ):
        """
        Ensure a job document exists in Firestore. If not, create it with the provided metadata or minimal info.
        """
        doc_ref = self.collection.document(job_id)
        doc = doc_ref.get()
        if not doc.exists:
            if job_metadata is None:
                # Create with minimal info
                job_metadata = JobSchema(
                    job_id=job_id,
                    job_name="unnamed job",
                    status="queued",
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                    processed_dataset_id="",
                    base_model_id="",
                    user_id="",
                )
            doc_ref.set(job_metadata.model_dump(exclude_none=True))

    def list_jobs(self) -> list:
        """
        List all jobs with job_id and job_name.
        """
        jobs = []
        try:
            docs = self.collection.stream()
            for doc in docs:
                data = doc.to_dict()
                jobs.append(
                    {
                        "job_id": data.get("job_id"),
                        "job_name": data.get("job_name"),
                        "base_model_id": data.get("base_model_id"),
                        "status": data.get("status"),
                        "modality": data.get("modality", "text"),
                    }
                )
            return jobs
        except Exception as e:
            self.logger.error(f"Failed to list jobs: {e}")
            raise

    def delete_job(self, job_id: str) -> bool:
        """
        Delete job metadata from Firestore.

        Args:
            job_id: Job identifier

        Returns:
            bool: True if job was deleted, False if job was not found

        Raises:
            Exception: If deletion fails for other reasons
        """
        try:
            doc_ref = self.collection.document(job_id)
            doc = doc_ref.get()
            if not doc.exists:
                self.logger.warning(f"Job {job_id} not found for deletion")
                return False

            doc_ref.delete()
            self.logger.info(f"Successfully deleted job {job_id} from Firestore")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete job {job_id}: {e}")
            raise

    def verify_processed_dataset_ownership(self, dataset_id: str, user_id: str) -> bool:
        """
        Verify that a processed dataset is owned by the specified user.

        Args:
            dataset_id: The processed dataset ID to check
            user_id: The user ID to verify ownership against

        Returns:
            bool: True if the user owns the dataset, False otherwise
        """
        try:
            doc = self.db.collection("processed_datasets").document(dataset_id).get()
            if not doc.exists:
                return False
            data = doc.to_dict()
            return data and data.get("user_id") == user_id
        except Exception as e:
            self.logger.error(
                f"Failed to verify processed dataset ownership for {dataset_id}: {e}"
            )
            return False

    def verify_job_ownership(self, job_id: str, user_id: str) -> bool:
        """
        Verify that a training job is owned by the specified user.

        Args:
            job_id: The job ID to check
            user_id: The user ID to verify ownership against

        Returns:
            bool: True if the user owns the job, False otherwise
        """
        try:
            doc = self.collection.document(job_id).get()
            if not doc.exists:
                return False
            data = doc.to_dict()
            return data and data.get("user_id") == user_id
        except Exception as e:
            self.logger.error(f"Failed to verify job ownership for {job_id}: {e}")
            return False

    def _sanitize_metrics(self, metrics):
        """
        Recursively sanitize metrics dict, replacing NaN/Infinity with None.
        """
        if isinstance(metrics, dict):
            return {k: self._sanitize_metrics(v) for k, v in metrics.items()}
        elif isinstance(metrics, list):
            return [self._sanitize_metrics(v) for v in metrics]
        elif isinstance(metrics, float):
            if math.isnan(metrics) or math.isinf(metrics):
                return None
            return metrics
        else:
            return metrics
