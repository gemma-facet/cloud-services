from datasets import Dataset
from .base import BaseParser
import docx

class DOCXParser(BaseParser):
    """Parser for Microsoft Word (DOCX) documents."""

    def parse(self, file_path: str) -> Dataset:
        """Parse a DOCX file, creating one row per paragraph or table cell.

        Args:
            file_path: Path to the DOCX file.

        Returns:
            Dataset: with a 'text' column.
        """
        try:
            doc = docx.Document(file_path)
            data = []
            
           
            for p in doc.paragraphs:
                if p.text:
                    data.append({"text": p.text.strip()})
            
           
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        if cell.text:
                            data.append({"text": cell.text.strip()})

            return Dataset.from_list(data)

        except Exception as e:
            print(f"Error parsing DOCX file {file_path}: {e}")
            return Dataset.from_list([])