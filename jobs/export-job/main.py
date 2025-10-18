import os
import sys
from datetime import datetime
from google.cloud import firestore
from dotenv import load_dotenv, find_dotenv
from utils import ExportUtils, logger
from schema import ExportSchema

# Load environment variables
load_dotenv(find_dotenv())


def main():
    try:
        export_id = os.getenv("EXPORT_ID")
        if not export_id:
            logger.error("Error: EXPORT_ID environment variable is required")
            raise ValueError("EXPORT_ID environment variable is required")

        project_id = os.getenv("PROJECT_ID")
        if not project_id:
            logger.error("Error: PROJECT_ID environment variable is required")
            raise ValueError("PROJECT_ID environment variable is required")

        hf_token = os.getenv("HF_TOKEN", None)
        database_name = os.getenv("FIRESTORE_DB", None)

        db = firestore.Client(project=project_id, database=database_name)

        export_ref = db.collection("exports").document(export_id)
        export_doc = export_ref.get()
        if not export_doc.exists:
            logger.error(f"Error: Export {export_id} not found")
            raise ValueError(f"Export {export_id} not found")

        export_data = export_doc.to_dict()
        export_data = ExportSchema(**export_data)

        export_type = export_data.type
        if not export_type:
            logger.error(f"Error: Export type not found for export {export_id}")
            raise ValueError(f"Export type not found for export {export_id}")

        if "hf_hub" in export_data.destination and not hf_token:
            logger.error(f"Error: HF token not found for export {export_id}")
            raise ValueError(f"HF token not found for export {export_id}")

        if "hf_hub" in export_data.destination and not export_data.hf_repo_id:
            logger.error(f"Error: HF repo ID not found for export {export_id}")
            raise ValueError(f"HF repo ID not found for export {export_id}")

        export_utils = ExportUtils(db, export_id, project_id, hf_token)

        if export_type == "adapter":
            export_utils.export_adapter()
        elif export_type == "merged":
            export_utils.export_merged()
        elif export_type == "gguf":
            export_utils.export_gguf()
        else:
            logger.error(f"Error: Unsupported export type: {export_type}")
            raise ValueError(f"Unsupported export type: {export_type}")

        export_ref.update(
            {"status": "completed", "message": None, "finished_at": datetime.now()}
        )

    except Exception as e:
        logger.error(f"Error: {e}")
        export_ref.update(
            {"status": "failed", "message": str(e), "finished_at": datetime.now()}
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
