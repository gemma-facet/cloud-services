import logging
from enum import Enum
from typing import Optional, Dict, Any
from google.cloud import firestore
from datetime import datetime, timezone


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

    def mark_preparing(self, job_id: str) -> None:
        self.collection.document(job_id).update(
            {
                "status": JobStatus.PREPARING.value,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        self.logger.info(f"Marked job {job_id} as preparing")

    def mark_training(self, job_id: str, wandb_url: Optional[str] = None) -> None:
        update_data = {
            "status": JobStatus.TRAINING.value,
            "updated_at": datetime.now(timezone.utc),
        }
        if wandb_url:
            update_data["wandb_url"] = wandb_url
        self.collection.document(job_id).update(update_data)
        self.logger.info(f"Marked job {job_id} as training")

    def mark_completed(
        self,
        job_id: str,
        adapter_path: str,
        base_model_id: str,
        export_format: str,
        metrics: Optional[Dict[str, Any]] = None,
        gguf_path: Optional[str] = None,
    ) -> None:
        """
        Mark job as completed and optionally store evaluation metrics.
        """
        update_data = {
            "status": JobStatus.COMPLETED.value,
            "updated_at": datetime.now(timezone.utc),
            "base_model_id": base_model_id,
        }

        # The training job saves the raw model, not a file (zip)
        # The export job will create the file artifact (zip)
        # NOTE: If export_format is "full", it means full fine-tune, so we save under raw.adapter as well
        if export_format in ["adapter", "merged", "full"]:
            if export_format == "full":
                export_format = "adapter"
            update_data[f"artifacts.raw.{export_format}"] = adapter_path

        if metrics is not None:
            update_data["metrics"] = metrics
        if gguf_path is not None:
            update_data["artifacts.file.gguf"] = gguf_path
        self.collection.document(job_id).update(update_data)
        self.logger.info(f"Marked job {job_id} as completed")

    def mark_failed(self, job_id: str, error: str) -> None:
        self.collection.document(job_id).update(
            {
                "status": JobStatus.FAILED.value,
                "updated_at": datetime.now(timezone.utc),
                "error": error,
            }
        )
        self.logger.info(f"Marked job {job_id} as failed: {error}")


# Context manager for automatic status updates
class JobTracker:
    """
    Context manager that automatically handles job status transitions.
    Usage:
        with JobTracker(job_manager, job_id) as tracker:
            tracker.preparing("Loading model...")
            # do preparation work
            tracker.training(wandb_url="https://...")
            # do training work
            tracker.completed(adapter_path="/path/to/adapter")
    """

    def __init__(
        self,
        job_manager: JobStateManager,
        job_id: str,
    ):
        self.job_manager = job_manager
        self.job_id = job_id
        self._error_occurred = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Automatically marks the job as failed if an error occurs."""
        if exc_type and not self._error_occurred:
            error_msg = f"{exc_type.__name__}: {str(exc_val)}"
            self.job_manager.mark_failed(self.job_id, error_msg)
        return False  # Don't suppress exceptions

    def preparing(self):
        """Mark job as preparing"""
        self.job_manager.mark_preparing(self.job_id)

    def training(self, wandb_url: Optional[str] = None):
        """Mark job as training"""
        self.job_manager.mark_training(self.job_id, wandb_url)

    def completed(
        self,
        adapter_path: str,
        base_model_id: str,
        export_format: str,
        metrics: Optional[Dict[str, Any]] = None,
        gguf_path: Optional[str] = None,
    ):
        """Mark job as completed and record evaluation metrics"""
        self.job_manager.mark_completed(
            self.job_id, adapter_path, base_model_id, export_format, metrics, gguf_path
        )

    def failed(self, error: str):
        """Mark job as failed"""
        self._error_occurred = True
        self.job_manager.mark_failed(self.job_id, error)
