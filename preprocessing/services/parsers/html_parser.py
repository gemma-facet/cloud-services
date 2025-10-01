import io
from datasets import Dataset
from .base import BaseParser
from bs4 import BeautifulSoup


class HTMLParser(BaseParser):
    """Parser for HTML files and web pages"""

    def parse(self, file_stream: io.BytesIO) -> Dataset:
        """Parse an HTML file from an in-memory stream.

        Args:
            file_stream: An in-memory binary stream (e.g., io.BytesIO)
                containing the HTML file content.

        Returns:
            Dataset: rows with tag and text
        """
        try:
            soup = BeautifulSoup(file_stream, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            data = []

            for tag in soup.find_all(
                ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "div"]
            ):
                if tag.find_parent(["script", "style", "nav", "footer", "aside"]):
                    continue
                text = tag.get_text(separator=" ", strip=True)
                if text:
                    data.append({"tag": tag.name, "text": tag.text})

            return Dataset.from_list(data)
        except Exception as e:
            print(f"Error parsing HTML file {file_path}: {e}")
            return Dataset.from_list([])
