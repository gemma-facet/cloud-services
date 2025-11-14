from pathlib import Path
from synthetic_data_kit.core.ingest import process_file as ingest
from synthetic_data_kit.core.create import process_file
from synthetic_data_kit.core.curate import curate_qa_pairs
from datasets import Dataset
import os
services_dir = Path(__file__).parent
preprocessing_dir = services_dir.parent
config_path = services_dir /"config.yaml"


class DatasetSynthesizer:
    """
    A class that handles ingestion of local file upload to generating synthesized datasets using meta-llama/synthetic-data-kit.

    Methods:
        synthesize_dataset(file_path, output_dir=None, config_path=None, include_conversations=False) -> Dataset
            
    """
    def __init__(self):
        pass

    def synthesize_dataset(
        self,  file_path: str, output_dir: str = None, config_path: str = config_path     
    )-> Dataset:
        """
        Complete synthetic data generation pipeline from file ingestion to dataset creation     
        
        Args:
            file_path: Path to the input file (e.g., PDF, DOCX, TXT)
            output_dir: Directory to save intermediate and final outputs
            config_path: Path to the configuration YAML file        
        Returns:
            Dataset: A Hugging Face Dataset object containing the curated QA pairs
        
        """
        txt_file_path =ingest(
            file_path=file_path,
            config=config_path,
            output_dir=output_dir,
        )
        qa_pairs_path = process_file(
            file_path=txt_file_path,
            config_path=config_path,
            output_dir=output_dir,
        )
        qa_pairs_path_p = Path(qa_pairs_path)
        base_output_dir = Path(output_dir) if output_dir is not None else qa_pairs_path_p.parent
        curated_output_path = base_output_dir / f"{qa_pairs_path_p.stem}.curated.json"
        curated_output_path.parent.mkdir(parents=True, exist_ok=True)

        curated_pairs = curate_qa_pairs(
            input_path=qa_pairs_path,
            config_path=config_path,
            output_path=str(curated_output_path),
        )
        print("Reached curated pair generated step...")
        dataset = Dataset.from_json(curated_pairs)
        return dataset.select_columns(["conversations"])