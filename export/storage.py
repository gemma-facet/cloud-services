import os
import logging
import shutil
import zipfile
from google.cloud import storage


class GCSStorageManager:
    """
    A utility class for managing Google Cloud Storage operations including
    downloading directories, uploading files and directories, and cleanup operations.
    """

    def __init__(self):
        """Initialize the GCS storage manager with Google Cloud Storage client."""
        self.storage_client = storage.Client()

        # Define bucket names
        self.export_bucket = os.getenv("GCS_EXPORT_BUCKET_NAME", "gemma-export-bucket")
        self.export_files_bucket = os.getenv(
            "GCS_EXPORT_FILES_BUCKET_NAME", "gemma-export-files"
        )

        self.logger = logging.getLogger(__name__)

    def _download_directory(self, gcs_full_path: str) -> str:
        """
        Download a directory from GCS to local temporary directory.

        Args:
            gcs_full_path: Full GCS path to the directory (e.g., "gs://bucket-name/path/to/directory/")

        Returns:
            str: Local path to the downloaded directory

        Raises:
            Exception: If download fails
        """
        try:
            # Parse GCS path to extract bucket and prefix
            if not gcs_full_path.startswith("gs://"):
                raise ValueError(f"Invalid GCS path format: {gcs_full_path}")

            path_parts = gcs_full_path[5:].split("/", 1)  # Remove "gs://" and split
            bucket_name = path_parts[0]
            gcs_prefix = path_parts[1] if len(path_parts) > 1 else ""

            # Extract directory name from GCS path
            directory_name = os.path.basename(gcs_prefix.rstrip("/"))
            if not directory_name:
                # If no basename, use the full path as directory name
                directory_name = gcs_prefix.replace("/", "_").strip("_")

            local_path = f"/tmp/dir/{directory_name}"

            # Create local directory
            os.makedirs(local_path, exist_ok=True)

            self.logger.info(
                f"Downloading directory from {gcs_full_path} to {local_path}"
            )

            # Get bucket and list all blobs with the prefix
            bucket = self.storage_client.bucket(bucket_name)
            blobs = bucket.list_blobs(prefix=gcs_prefix)

            downloaded_files = []

            for blob in blobs:
                # Skip directory placeholders (blobs ending with "/")
                if blob.name.endswith("/"):
                    continue

                # Calculate relative path within the directory
                relative_path = blob.name[len(gcs_prefix) :].lstrip("/")
                if not relative_path:
                    continue

                local_file_path = os.path.join(local_path, relative_path)

                # Create subdirectories if needed
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

                # Download the file
                blob.download_to_filename(local_file_path)
                downloaded_files.append(relative_path)
                self.logger.info(f"Downloaded {blob.name} to {local_file_path}")

            if not downloaded_files:
                raise Exception(f"No files found in directory {gcs_full_path}")

            self.logger.info(
                f"Successfully downloaded {len(downloaded_files)} files to {local_path}"
            )
            return local_path

        except Exception as e:
            self.logger.error(f"Failed to download directory {gcs_full_path}: {str(e)}")
            raise Exception(f"Directory download failed: {str(e)}")

    def _upload_directory(
        self, local_directory_path: str, gcs_destination_path: str
    ) -> str:
        """
        Upload a local directory to GCS, placing all contents directly in the destination.

        Args:
            local_directory_path: Local path to the directory to upload
            gcs_destination_path: GCS destination path (e.g., "gs://bucket-name/some_id/export")

        Returns:
            str: GCS path to the uploaded directory

        Raises:
            Exception: If upload fails
        """
        try:
            if not os.path.exists(local_directory_path):
                raise Exception(
                    f"Local directory does not exist: {local_directory_path}"
                )

            # Parse GCS destination path
            if not gcs_destination_path.startswith("gs://"):
                raise ValueError(f"Invalid GCS path format: {gcs_destination_path}")

            path_parts = gcs_destination_path[5:].split(
                "/", 1
            )  # Remove "gs://" and split
            bucket_name = path_parts[0]
            gcs_prefix = path_parts[1] if len(path_parts) > 1 else ""

            bucket = self.storage_client.bucket(bucket_name)

            self.logger.info(
                f"Uploading directory from {local_directory_path} to {gcs_destination_path}"
            )

            uploaded_files = []

            # Walk through all files in the directory
            for root, dirs, files in os.walk(local_directory_path):
                for file in files:
                    local_file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(
                        local_file_path, local_directory_path
                    )
                    gcs_blob_path = (
                        f"{gcs_prefix}/{relative_path}" if gcs_prefix else relative_path
                    )

                    # Upload file to GCS
                    blob = bucket.blob(gcs_blob_path)
                    blob.upload_from_filename(local_file_path)
                    uploaded_files.append(gcs_blob_path)
                    self.logger.info(
                        f"Uploaded {local_file_path} to gs://{bucket_name}/{gcs_blob_path}"
                    )

            if not uploaded_files:
                raise Exception(f"No files found in directory: {local_directory_path}")

            self.logger.info(
                f"Successfully uploaded {len(uploaded_files)} files to {gcs_destination_path}"
            )
            return gcs_destination_path

        except Exception as e:
            self.logger.error(
                f"Failed to upload directory {local_directory_path}: {str(e)}"
            )
            raise Exception(f"Directory upload failed: {str(e)}")

    def _upload_file(
        self, local_file_path: str, gcs_destination_path: str, final_filename: str
    ) -> str:
        """
        Upload a single file to GCS with custom destination path and filename.

        Args:
            local_file_path: Local path to the file to upload
            gcs_destination_path: GCS destination path (e.g., "gs://bucket-name/some_id/export")
            final_filename: Final filename for the uploaded file (without extension)

        Returns:
            str: GCS path to the uploaded file

        Raises:
            Exception: If upload fails
        """
        try:
            if not os.path.exists(local_file_path):
                raise Exception(f"Local file does not exist: {local_file_path}")

            # Parse GCS destination path
            if not gcs_destination_path.startswith("gs://"):
                raise ValueError(f"Invalid GCS path format: {gcs_destination_path}")

            path_parts = gcs_destination_path[5:].split(
                "/", 1
            )  # Remove "gs://" and split
            bucket_name = path_parts[0]
            gcs_prefix = path_parts[1] if len(path_parts) > 1 else ""

            # Get original file extension
            original_extension = os.path.splitext(local_file_path)[1]
            gcs_filename = f"{final_filename}{original_extension}"

            # Construct full GCS blob path
            gcs_blob_path = (
                f"{gcs_prefix}/{gcs_filename}" if gcs_prefix else gcs_filename
            )

            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(gcs_blob_path)

            self.logger.info(
                f"Uploading file from {local_file_path} to gs://{bucket_name}/{gcs_blob_path}"
            )

            blob.upload_from_filename(local_file_path)

            gcs_path = f"gs://{bucket_name}/{gcs_blob_path}"
            self.logger.info(f"Successfully uploaded file to {gcs_path}")
            return gcs_path

        except Exception as e:
            self.logger.error(f"Failed to upload file {local_file_path}: {str(e)}")
            raise Exception(f"File upload failed: {str(e)}")

    def _zip_upload_file(
        self, local_directory_path: str, gcs_destination_path: str, final_filename: str
    ) -> str:
        """
        Zip a local directory and upload it to GCS with custom destination path and filename.

        Args:
            local_directory_path: Local path to the directory to zip and upload
            gcs_destination_path: GCS destination path (e.g., "gs://bucket-name/some_id/export")
            final_filename: Final filename for the uploaded zip file (without .zip extension)

        Returns:
            str: GCS path to the uploaded zip file

        Raises:
            Exception: If zip creation or upload fails
        """
        zip_file_path = None
        try:
            if not os.path.exists(local_directory_path):
                raise Exception(
                    f"Local directory does not exist: {local_directory_path}"
                )

            # Create zip filename
            zip_filename = f"{final_filename}.zip"

            # Create temporary zip file
            zip_file_path = f"/tmp/{zip_filename}"

            self.logger.info(
                f"Creating zip file from {local_directory_path} to {zip_file_path}"
            )

            # Create zip file
            with zipfile.ZipFile(zip_file_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(local_directory_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, local_directory_path)
                        zipf.write(file_path, arcname)
                        self.logger.info(f"Added {file_path} to zip as {arcname}")

            # Upload zip file to GCS
            gcs_path = self._upload_file(
                zip_file_path, gcs_destination_path, final_filename
            )

            self.logger.info(
                f"Successfully created and uploaded zip file to {gcs_path}"
            )
            return gcs_path

        except Exception as e:
            self.logger.error(
                f"Failed to zip and upload directory {local_directory_path}: {str(e)}"
            )
            raise Exception(f"Zip upload failed: {str(e)}")
        finally:
            # Clean up zip file
            if zip_file_path and os.path.exists(zip_file_path):
                self._cleanup_local_file(zip_file_path)

    def _cleanup_local_directory(self, directory_path: str) -> None:
        """
        Clean up a local directory and all its contents.

        Args:
            directory_path: Path to the directory to clean up
        """
        try:
            if os.path.exists(directory_path):
                shutil.rmtree(directory_path)
                self.logger.info(f"Cleaned up local directory: {directory_path}")
            else:
                self.logger.warning(
                    f"Directory does not exist for cleanup: {directory_path}"
                )
        except Exception as e:
            self.logger.warning(
                f"Failed to clean up directory {directory_path}: {str(e)}"
            )

    def _cleanup_local_file(self, file_path: str) -> None:
        """
        Clean up a local file.

        Args:
            file_path: Path to the file to clean up
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                self.logger.info(f"Cleaned up local file: {file_path}")
            else:
                self.logger.warning(f"File does not exist for cleanup: {file_path}")
        except Exception as e:
            self.logger.warning(f"Failed to clean up file {file_path}: {str(e)}")


# Create a global instance for easy access
gcs_storage = GCSStorageManager()
