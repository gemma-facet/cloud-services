# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Ingest different file formats

import os
import sys
import requests
from pathlib import Path
from typing import Optional, Dict, Any
import importlib

from synthetic_data_kit.utils.config import get_path_config


def _check_pdf_url(url: str) -> bool:
    """Check if `url` points to PDF content

    Args:
        url: URL to check


    Returns:
        bool: True if the URL points to PDF content, False otherwise
    """
    try:
        response = requests.head(url, allow_redirects=True)
        content_type = response.headers.get("Content-Type", "")
        return "application/pdf" in content_type
    except requests.RequestException:
        return False


def determine_parser(file_path: str, config: Dict[str, Any], multimodal: bool = False):
    """Determine the appropriate parser for a file or URL"""
    from synthetic_data_kit.parsers.pdf_parser import PDFParser
    from synthetic_data_kit.parsers.html_parser import HTMLParser
    from synthetic_data_kit.parsers.youtube_parser import YouTubeParser
    from synthetic_data_kit.parsers.docx_parser import DOCXParser
    from synthetic_data_kit.parsers.ppt_parser import PPTParser
    from synthetic_data_kit.parsers.txt_parser import TXTParser
    from synthetic_data_kit.parsers.multimodal_parser import MultimodalParser

    ext = os.path.splitext(file_path)[1].lower()
    if multimodal:
        if ext in [".pdf", ".docx", ".pptx"]:
            return MultimodalParser()
        else:
            raise ValueError(f"Unsupported file extension for multimodal parsing: {ext}")

    if ext == ".pdf":
        return PDFParser()

    # Check if it's a URL
    if file_path.startswith(("http://", "https://")):
        # YouTube URL
        if "youtube.com" in file_path or "youtu.be" in file_path:
            return YouTubeParser()
        # PDF URL
        elif _check_pdf_url(file_path):
            return MultimodalParser() if multimodal else PDFParser()
        # HTML URL
        else:
            return HTMLParser()

    # File path - determine by extension
    if os.path.exists(file_path):
        parsers = {
            ".html": HTMLParser(),
            ".htm": HTMLParser(),
            ".docx": DOCXParser(),
            ".pptx": PPTParser(),
            ".txt": TXTParser(),
        }

        if ext in parsers:
            return parsers[ext]
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    raise FileNotFoundError(f"File not found: {file_path}")


def process_file(
    file_path: str,
    config: Optional[Dict[str, Any]] = None,
    multimodal: bool = False,
) -> str:
    """Process a file using the appropriate parser and return in-memory content.

    Args:
        file_path: Path to the file or URL to parse.
        config: Configuration dictionary (if None, uses default).
        multimodal: Whether to use the multimodal parser.

    Returns:
        The parsed text content as a string.
    """
    parser = determine_parser(file_path, config, multimodal)

    parsed_data = parser.parse(file_path)
    
    text_content = " ".join([item['text'] for item in parsed_data if 'text' in item and item['text']])

    return text_content
