from datasets import Dataset
from .base import BaseParser
import pymupdf

class PDFParser(BaseParser):
    """Parser for PDF documents"""

    def parse(self, file_path: str) -> Dataset:
        """Parse a PDF file into a Dataset, with one row per page.

        Args:
            file_path: Path to the PDF file.

        Returns:
             Dataset: with 'page_num' and 'text' columns.
        """
        try:
            doc = pymupdf.open(file_path)
            data = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                if text:
                    data.append({
                        "page_num": page_num + 1,
                        "text": text
                    })
            return Dataset.from_list(data)
        except Exception as e:
            print(f"Error parsing PDF file {file_path}: {e}")
            return Dataset.from_list([])