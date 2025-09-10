import logging
from google.cloud import firestore
from schema import ExportSchema, JobSchema
from typing import Optional


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

    def export_adapter(self):
        pass

    def export_merged(self):
        pass

    def export_gguf(self):
        pass
