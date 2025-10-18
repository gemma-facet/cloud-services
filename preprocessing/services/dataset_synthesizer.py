from pathlib import Path
from synthetic_data_kit.core.create import process_file
from synthetic_data_kit.core.ingest import process_file as ingest
from synthetic_data_kit.core.curate import curate_qa_pairs

current_dir = Path(__file__).parent
config_path = current_dir / "synthetic_data_kit" /"synthetic_data_kit" / "config.yaml"

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
    parsed_content = ingest( # parsers the file and return the text content
        file_path=file_path,
        config=config_path,
    ) 
    generated_content = process_file( # generates a Dict of summary : "" , qa_pairs : ""
        text_content=parsed_content,
        config_path=config_path,
        num_pairs=10,
        content_type='qa',
    )
    print(f"\n\nGenerated summary and qa pairs : {generated_content}\n\n")
    curated_pairs = curate_qa_pairs(
        qa_pairs=generated_content['qa_pairs'],
        config_path=config_path,
        verbose=True,
    )
    print(f"\n\nCurated Dataset : {curated_pairs}\n\n")
    return curated_pairs

