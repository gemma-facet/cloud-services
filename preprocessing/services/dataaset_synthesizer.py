from pathlib import Path
from synthetic_data_kit.core.create import process_file
from synthetic_data_kit.core.ingest import process_file as ingest
from synthetic_data_kit.core.curate import curate_qa_pairs

current_dir = Path(__file__).parent
config_path = current_dir /"config.yaml"

def synthetic_data_pipline(file_path,config_path):
    """ Complete synthetic data generation pipeline from file ingestion to dataset creation
    
    Args:
        file_path: Path to the input file (e.g., PDF, DOCX, TXT)
        output_dir: Directory to save intermediate and final outputs
        config_path: Path to the configuration YAML file    
        api_base: API base URL for the LLM
        model: Model name to use for the LLM

    Returns:
        curated_pairs: List of curated QA pairs generated from the input file
    """
    