import logging
from google.cloud import firestore
from schema import (
    ExportSchema,
    JobSchema,
    export_status,
    export_type,
    export_variant,
    ExportArtifact,
)
from typing import Optional
from storage import gcs_storage


class ExportUtils:
    """
    Utility class for export job operations.
    """

    def __init__(
        self,
        db: firestore.Client,
        export_id: str,
        project_id: str,
        hf_token: Optional[str] = None,
    ):
        """
        Initialize ExportUtils with database connection and export configuration.

        Args:
            db: Firestore database client instance
            export_id: Unique identifier for the export job
            project_id: Google Cloud project ID (must be set)
            hf_token: Optional Hugging Face token for model access

        Raises:
            ValueError: If project_id is not provided or empty
        """
        self.project_id = project_id
        if not self.project_id:
            raise ValueError("PROJECT_ID environment variable must be set")

        self.db = db
        self.export_id = export_id
        self.export_ref = self.db.collection("exports").document(export_id)
        self.export_doc = self.export_ref.get().to_dict()
        self.export_doc = ExportSchema(**self.export_doc)
        self.job_id = self.export_doc.job_id
        self.job_ref = self.db.collection("training_jobs").document(self.job_id)
        self.job_doc = self.job_ref.get().to_dict()
        self.job_doc = JobSchema(**self.job_doc)
        self.hf_token = hf_token

        self.logger = logging.getLogger(__name__)

    def _update_status(self, status: export_status, message: Optional[str] = None):
        """
        Update the export job status in Firestore.

        Args:
            status: The new status for the export job ("running", "completed", or "failed")
            message: Optional status message to provide additional context
        """
        self.export_ref.update(
            {
                "status": status,
                "message": message,
                "updated_at": firestore.SERVER_TIMESTAMP,
            }
        )

    def _update_export_artifacts(
        self, type: export_type, variant: export_variant, path: str
    ):
        """
        Add an artifact to the export job's artifacts list in Firestore.

        Args:
            type: Type of artifact ("adapter", "merged", or "gguf")
            variant: Variant of artifact ("raw" or "file")
            path: GCS path where the artifact is stored
        """
        artifact = ExportArtifact(type=type, variant=variant, path=path)
        self.export_ref.update({"artifacts": firestore.ArrayUnion([artifact])})

    def _update_job_artifacts(
        self, type: export_type, variant: export_variant, path: str
    ):
        """
        Update the training job's artifacts with the exported artifact path.

        Args:
            type: Type of artifact ("adapter", "merged", or "gguf")
            variant: Variant of artifact ("raw" or "file")
            path: GCS path where the artifact is stored
        """
        self.job_ref.update({f"artifacts.{variant}.{type}": path})

    def export_adapter(self):
        """
        Export the adapter model from the training job.

        Downloads the adapter from GCS, creates a zip file, uploads it to the export bucket,
        and updates both the export job and training job with the artifact information.

        Raises:
            ValueError: If adapter path is not found in the job document
        """
        self.logger.info(f"Exporting adapter for job {self.job_id}")
        self._update_status("running", "Preparing adapter export")

        if self.job_doc.artifacts.files.adapter:
            self.logger.info(f"Adapter already exported for job {self.job_id}")
            self._update_status("completed", "Adapter already present in the database.")
            return

        if not self.job_doc.adapter_path:
            raise ValueError("Adapter path not found in the database.")

        adapter_path = self.job_doc.adapter_path
        local_adapter_path = gcs_storage._download_directory(adapter_path)

        files_destination = f"gs://{gcs_storage.export_files_bucket}/{self.job_id}"
        gcs_zip_path = gcs_storage._zip_upload_file(
            local_adapter_path, files_destination, "adapter"
        )

        self._update_export_artifacts("adapter", "file", gcs_zip_path)
        self._update_job_artifacts("adapter", "file", gcs_zip_path)
        self._update_status("completed", "Adapter exported successfully.")

        return

    def export_merged(self):
        pass

    def export_gguf(self):
        pass
