import io
from datasets import Dataset
from .base import BaseParser
import pymupdf

class PDFParser(BaseParser):
    """Parser for PDF documents"""

    def parse(self, file_stream: io.BytesIO) -> Dataset:
        """Parse a PDF file into a Dataset from an in-memory stream, with one row per page.
    
        Args:
            file_stream: An in-memory binary stream (e.g., io.BytesIO) containing the PDF file content. 
    
        Returns:
             Dataset: with 'page_num' and 'text' columns.
        """
        try:
            doc = pymupdf.open(stream=file_stream, filetype="pdf")            
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