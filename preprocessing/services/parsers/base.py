from abc import ABC, abstractmethod
from datasets import Dataset

class BaseParser(ABC):
    """Abstract base class for file parsers"""

    @abstractmethod
    def parse(self,file_path: str) -> Dataset:
        """Parse a file into a Dataset object.
        Args:
            file_path: Path to the file to be parsed
        Returns:
            Dataset : A Hugging Face Dataset containing the parsed content from the file
        """
        pass
