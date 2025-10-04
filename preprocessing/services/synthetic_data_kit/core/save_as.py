# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Logic for saving file format

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from synthetic_data_kit.utils.format_converter import to_jsonl, to_alpaca, to_fine_tuning, to_chatml, to_hf_dataset, to_hf_datasetdict, to_parquet
from synthetic_data_kit.utils.llm_processing import convert_to_conversation_format

def convert_format(
    input_path: Optional[str] = None,
    qa_pairs: Optional[List[Dict[str, Any]]] = None,
    output_path: Optional[str] = None,
    format_type: str = "jsonl",
    config: Optional[Dict[str, Any]] = None,
    storage_format: str = "json",
) -> str:
    """Convert data to different formats
    
    Args:
        input_path: Path to the input file (required if qa_pairs is not provided)
        qa_pairs: A list of QA pairs to convert (required if input_path is not provided)
        output_path: Path to save the output
        format_type: Output format (jsonl, alpaca, ft, chatml)
        config: Configuration dictionary
        storage_format: Storage format, either "json" or "hf" (Hugging Face dataset)
    
    Returns:
        Path to the output file or directory
    """
    if qa_pairs is None:
        if input_path and os.path.exists(input_path):
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            qa_pairs = data.get("qa_pairs", [])
        else:
            raise ValueError("Either 'qa_pairs' or a valid 'input_path' must be provided.")
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # When using HF dataset storage format
    if storage_format == "hf":
        # For HF datasets, we need to prepare the data in the right structure
        if format_type == "jsonl":
            # For JSONL, just use the QA pairs directly
            formatted_pairs = qa_pairs
        elif format_type == "alpaca":
            # Format as Alpaca structure
            formatted_pairs = []
            for pair in qa_pairs:
                formatted_pairs.append({
                    "instruction": pair["question"],
                    "input": "",
                    "output": pair["answer"]
                })
        elif format_type == "ft":
            # Format as OpenAI fine-tuning structure
            formatted_pairs = []
            for pair in qa_pairs:
                formatted_pairs.append({
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": pair["question"]},
                        {"role": "assistant", "content": pair["answer"]}
                    ]
                })
        elif format_type == "chatml":
            # Format as ChatML structure
            formatted_pairs = []
            for pair in qa_pairs:
                formatted_pairs.append({
                    "messages": [
                        {"role": "system", "content": "You are a helpful AI assistant."},
                        {"role": "user", "content": pair["question"]},
                        {"role": "assistant", "content": pair["answer"]}
                    ]
                })
            
        else:
            raise ValueError(f"Unknown format type: {format_type}")
            
        # Save as HF dataset (Arrow format)
        return to_hf_datasetdict(formatted_pairs, output_path)
        
    
    # Standard JSON file storage format
    else:
        # Convert to the requested format using existing functions
        if format_type == "jsonl":
            return to_jsonl(qa_pairs, output_path)
        elif format_type == "alpaca":
            return to_alpaca(qa_pairs, output_path)
        elif format_type == "ft":
            return to_fine_tuning(qa_pairs, output_path)
        elif format_type == "chatml":
            return to_chatml(qa_pairs, output_path)
        else:
            raise ValueError(f"Unknown format type: {format_type}")