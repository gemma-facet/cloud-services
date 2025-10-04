import json
from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()
from synthetic_data_kit.core.ingest import process_file as ingest
from synthetic_data_kit.core.create import process_file
from synthetic_data_kit.core.curate import curate_qa_pairs
from synthetic_data_kit.core.save_as import convert_format

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"] 
api_key = GEMINI_API_KEY
config_path = Path(r"E:\facet-ai\my_config.yaml")
output_dir  = r"E:\facet-ai\data\processed_datasets"
api_base="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
model = "gemini-2.5-flash"

file_path = r"E:\facet-ai\sample\bitcoin.pdf"



def synthetic_data_pipline(file_path, output_dir, config_path, api_base, model):
    """ Complete synthetic data generation pipeline from file ingestion to dataset creation
    
    Args:
        file_path: Path to the input file (e.g., PDF, DOCX, TXT)
        output_dir: Directory to save intermediate and final outputs
        config_path: Path to the configuration YAML file    
        api_base: API base URL for the LLM
        model: Model name to use for the LLM

    Returns:
        Hugging Face DatasetDict format
    """

    parsed_content = ingest( # parsers the file and return the text content
        file_path=file_path,
        config=config_path,
    ) 
    generated_content = process_file( # generates a Dict of summary : "" , qa_pairs : ""
        text_content=parsed_content,
        config_path=config_path,
        api_base=api_base,
        model=model,
        provider="api-endpoint",
        num_pairs=10,
        content_type='qa'
    )
    print(f"\n\nGenerated summary and qa pairs : {generated_content}\n\n")
    curated_pairs = curate_qa_pairs(
        qa_pairs=generated_content['qa_pairs'],
        api_base=api_base,
        model=model,
        config_path=config_path,
        verbose=True,
        provider="api-endpoint",
    )
    print(f"\n\nCurated Dataset : {curated_pairs}\n\n")
    
    dataset_dict = convert_format(
        qa_pairs=curated_pairs,
        output_path=output_dir,
        format_type="chatml",
        storage_format="hf",
    )

    print(f"\n\nFinal  DatasetDict :  {dataset_dict}\n\n")
    return dataset_dict

if __name__ == "__main__":
    synthetic_data_pipline(
        file_path=file_path,
        output_dir=output_dir,
        config_path=config_path,
        api_base=api_base,
        model=model,
    )

