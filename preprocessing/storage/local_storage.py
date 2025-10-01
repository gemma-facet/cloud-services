import json
import logging
from typing import Union, List, Dict
from datetime import datetime
from pathlib import Path
from .base import StorageInterface

logger = logging.getLogger(__name__)


class LocalStorageManager(StorageInterface):
    """Local file system storage implementation"""

    def __init__(self, base_path: str = "./data"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def upload_data(
        self, data: Union[str, List[Dict], bytes], path: str, metadata: Dict = None
    ) -> str:
        """Save data to local file system"""
        try:
            file_path = self.base_path / path
            file_path.parent.mkdir(parents=True, exist_ok=True)

            if isinstance(data, bytes):
                with open(file_path, "wb") as f:
                    f.write(data)
            elif isinstance(data, str):
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(data)
            else:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)

            if metadata:
                metadata_path = file_path.with_suffix(file_path.suffix + ".meta")
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)

            return f"file://{file_path.absolute()}"

        except Exception as e:
            logger.error(f"Error saving to local storage: {str(e)}")
            raise

    def download_data(self, path: str) -> str:
        """Load data from local file system"""
        try:
            file_path = self.base_path / path
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading from local storage: {str(e)}")
            raise

    def download_binary_data(self, path: str) -> bytes:
        """Load binary data from local file system"""
        try:
            file_path = self.base_path / path
            with open(file_path, "rb") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading binary data from local storage: {str(e)}")
            raise

    def list_files(self, prefix: str = "") -> List[str]:
        """List files in local storage"""
        try:
            files = []
            for file_path in self.base_path.rglob("*"):
                if file_path.is_file() and not file_path.name.endswith(".meta"):
                    relative_path = file_path.relative_to(self.base_path).as_posix()
                    if prefix == "" or relative_path.startswith(prefix):
                        files.append(relative_path)
            return sorted(files)
        except Exception as e:
            logger.error(f"Error listing files: {str(e)}")
            raise

    def get_metadata(self, path: str) -> Dict:
        """Get file metadata from local storage"""
        try:
            file_path = self.base_path / path
            if not file_path.exists():
                return {}

            stat = file_path.stat()
            metadata = {
                "size": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "updated": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "metadata": {},
            }

            metadata_path = file_path.with_suffix(file_path.suffix + ".meta")
            if metadata_path.exists():
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata["metadata"] = json.load(f)

            return metadata
        except Exception as e:
            logger.error(f"Error getting metadata: {str(e)}")
            raise

    def file_exists(self, path: str) -> bool:
        """Check if file exists in local storage"""
        file_path = self.base_path / path
        return file_path.exists()

    def delete_file(self, path: str) -> bool:
        """Delete file from local storage"""
        try:
            file_path = self.base_path / path
            if file_path.exists():
                file_path.unlink()

                metadata_path = file_path.with_suffix(file_path.suffix + ".meta")
                if metadata_path.exists():
                    metadata_path.unlink()

                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting file: {str(e)}")
            raise

    def delete_directory(self, prefix: str) -> int:
        """
        Delete all files with a given prefix (directory) from local storage

        Args:
            prefix: The prefix/directory to delete

        Returns:
            Number of files deleted
        """
        try:
            import shutil

            deleted_count = 0
            prefix_path = self.base_path / prefix

            if prefix_path.exists() and prefix_path.is_dir():
                # Delete entire directory
                shutil.rmtree(prefix_path)
                # Count files that were in the directory
                for file_path in self.base_path.rglob("*"):
                    if file_path.is_file() and str(
                        file_path.relative_to(self.base_path)
                    ).startswith(prefix):
                        deleted_count += 1
                logger.info(f"Deleted directory: {prefix_path}")
            else:
                # Delete files matching prefix
                for file_path in self.base_path.rglob("*"):
                    if file_path.is_file():
                        relative_path = str(
                            file_path.relative_to(self.base_path)
                        ).replace("\\", "/")
                        if relative_path.startswith(prefix):
                            file_path.unlink()
                            # Also delete metadata file if exists
                            metadata_path = file_path.with_suffix(
                                file_path.suffix + ".meta"
                            )
                            if metadata_path.exists():
                                metadata_path.unlink()
                            deleted_count += 1
                            logger.info(f"Deleted: {relative_path}")

            logger.info(f"Deleted {deleted_count} files with prefix: {prefix}")
            return deleted_count
        except Exception as e:
            logger.error(f"Error deleting directory with prefix {prefix}: {str(e)}")
            raise
