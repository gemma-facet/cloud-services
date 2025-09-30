from datasets import Dataset
from .base import BaseParser
from bs4 import BeautifulSoup

class HTMLParser(BaseParser):
    """Parser for HTML files and web pages"""
    
    def parse(self, file_path: str) -> Dataset:
        """Parse an HTML file into a Dataset of text blocks
        
        Args:
            file_path: Path to the HTML file or URL
            
        Returns:
            Dataset: rows with tag and text
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()

            soup = BeautifulSoup(html_content, 'html.parser')

            # Remove script and style elements
            for script in soup(['script', 'style']):
                script.decompose()

            data = []

            for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li','div']):
                if tag.find_parent(['script', 'style','nav','footer','aside']):
                    continue
                text = tag.get_text(separator=' ', strip=True)
                if text:
                    data.append({ "tag": tag.name,
                                 "text": tag.text}
                                 )

            return Dataset.from_list(data)
        except Exception as e:
            print(f"Error parsing HTML file {file_path}: {e}")
            return Dataset.from_list([])
        

