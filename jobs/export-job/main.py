import os
import logging
import sys
from datetime import datetime
from google.cloud import firestore
from dotenv import load_dotenv, find_dotenv
from utils import ExportUtils
from schema import ExportSchema

# Load environment variables
load_dotenv(find_dotenv())


def main():
    try:
        export_id = os.getenv("EXPORT_ID")
        if not export_id:
            logging.error("Error: EXPORT_ID environment variable is required")
            raise ValueError("EXPORT_ID environment variable is required")

        project_id = os.getenv("PROJECT_ID")
        if not project_id:
            logging.error("Error: PROJECT_ID environment variable is required")
            raise ValueError("PROJECT_ID environment variable is required")

        hf_token = os.getenv("HF_TOKEN", None)

        db = firestore.Client(project=project_id)

        export_ref = db.collection("exports").document(export_id)
        export_doc = export_ref.get()
        if not export_doc.exists:
            logging.error(f"Error: Export {export_id} not found")
            raise ValueError(f"Export {export_id} not found")

        export_data = export_doc.to_dict()
        export_data = ExportSchema(**export_data)

        export_type = export_data.type
        if not export_type:
            logging.error(f"Error: Export type not found for export {export_id}")
            raise ValueError(f"Export type not found for export {export_id}")

        export_utils = ExportUtils(db, export_id, project_id, hf_token)

        if export_type == "adapter":
            export_utils.export_adapter()
        elif export_type == "merged":
            export_utils.export_merged()
        elif export_type == "gguf":
            export_utils.export_gguf()
        else:
            logging.error(f"Error: Unsupported export type: {export_type}")
            raise ValueError(f"Unsupported export type: {export_type}")

        export_ref.update(
            {"status": "completed", "message": None, "finished_at": datetime.now()}
        )

    except Exception as e:
        logging.error(f"Error: {e}")
        export_ref.update(
            {"status": "failed", "message": str(e), "finished_at": datetime.now()}
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
