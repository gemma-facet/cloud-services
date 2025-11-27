from pathlib import Path
from synthetic_data_kit.core.ingest import process_file as ingest
from synthetic_data_kit.core.create import process_file
from synthetic_data_kit.core.curate import curate_qa_pairs
from datasets import Dataset
import yaml
import json
import uuid

services_dir = Path(__file__).parent
preprocessing_dir = services_dir.parent
base_config_path = services_dir / "config.yaml"


class DatasetSynthesizer:
    """
    A class that handles ingestion of local file upload to generating synthesized datasets using meta-llama/synthetic-data-kit.

    The API key is provided per-request instead of from environment variables.

    Methods:
        synthesize_dataset(file_path, gemini_api_key, output_dir=None) -> Dataset
    """

    def __init__(self):
        # Load base config (without API key - will be injected per request)
        with open(base_config_path, "r") as f:
            self.base_config = yaml.safe_load(f)

    def synthesize_dataset(
        self,
        file_path: str,
        gemini_api_key: str,
        output_dir: str = None,
        num_pairs: int = 5,
        temperature: float = 0.7,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        threshold: float = 7.0,
        batch_size: int = 5,
        multimodal: bool = False,
    ) -> Dataset:
        """
        Complete synthetic data generation pipeline from file ingestion to dataset creation

        Args:
            file_path: Path to the input file (e.g., PDF, DOCX, TXT)
            gemini_api_key: The Gemini API key for this request (compulsory)
            output_dir: Directory to save intermediate and final outputs
            num_pairs: Number of QA pairs to generate per chunk (default: 5)
            temperature: LLM temperature for generation (default: 0.7)
            chunk_size: Size of text chunks in characters (default: 4000)
            chunk_overlap: Overlap between chunks to preserve context (default: 200)
            threshold: Quality threshold for curation (1-10) (default: 7.0)
            batch_size: Number of items per batch for rating (default: 5)
            multimodal: Whether to process multimodal data (default: False)

        Returns:
            Dataset: A Hugging Face Dataset object containing the curated QA pairs
        """
        # Create a request-specific config with custom parameters and API key
        runtime_config = self.base_config.copy()

        # Inject the request's API key
        if "api-endpoint" not in runtime_config:
            runtime_config["api-endpoint"] = {}
        runtime_config["api-endpoint"]["api_key"] = gemini_api_key

        # Update generation parameters
        if "generation" not in runtime_config:
            runtime_config["generation"] = {}
        runtime_config["generation"]["num_pairs"] = num_pairs
        runtime_config["generation"]["temperature"] = temperature
        runtime_config["generation"]["chunk_size"] = chunk_size
        runtime_config["generation"]["overlap"] = chunk_overlap

        # Update curation parameters
        if "curate" not in runtime_config:
            runtime_config["curate"] = {}
        runtime_config["curate"]["threshold"] = threshold
        runtime_config["curate"]["batch_size"] = batch_size

        # Save to a request-specific runtime config file
        request_id = str(uuid.uuid4())[:8]
        request_config_path = services_dir / f"config.runtime.{request_id}.yaml"

        try:
            with open(request_config_path, "w") as f:
                yaml.dump(runtime_config, f, default_flow_style=False)

            # Step 1: Ingest the file
            txt_file_path = ingest(
                file_path=file_path,
                config=str(request_config_path),
                output_dir=output_dir,
                # Multimodal means ingest images in a separate Lance dataset column alongside text
                # It does not generate synthetic image data or multimodal QA pairs
                multimodal=multimodal,
            )

            # NOTE: If processing a directory, use process_directory_ingest
            # However, this is not exported by the library only usable by the CLI...
            # results = process_directory_ingest(
            #     directory=input,
            #     output_dir=output_dir,
            #     config=ctx.config,
            #     multimodal=multimodal,
            # )

            # Step 2: Generate QA pairs
            qa_pairs_path = process_file(
                file_path=txt_file_path,
                config_path=str(request_config_path),
                output_dir=output_dir,
            )

            # Step 3: Curate the QA pairs
            qa_pairs_path_p = Path(qa_pairs_path)
            base_output_dir = (
                Path(output_dir) if output_dir is not None else qa_pairs_path_p.parent
            )
            curated_output_path = (
                base_output_dir / f"{qa_pairs_path_p.stem}.curated.json"
            )
            curated_output_path.parent.mkdir(parents=True, exist_ok=True)

            curated_pairs = curate_qa_pairs(
                input_path=qa_pairs_path,
                config_path=str(request_config_path),
                output_path=str(curated_output_path),
            )

            # Load and flatten the JSON file to create one row per conversation
            # The curated_pairs JSON has structure: {conversations: [[conv1], [conv2], ...]}
            # We need to convert it to JSONL format with one row per conversation
            with open(curated_pairs, "r") as f:
                curated_data = json.load(f)

            # Extract conversations list
            conversations_list = curated_data.get("conversations", [])

            # Create a new JSONL-like list where each item is one conversation
            flattened_rows = []
            for conversation in conversations_list:
                flattened_rows.append({"conversations": [conversation]})

            # Load as dataset from the flattened rows
            dataset = Dataset.from_dict(
                {"conversations": [row["conversations"] for row in flattened_rows]}
            )

            # Restyle dataset to match expected format:
            # 1. Rename "conversations" to "messages"
            # 2. Transform each message's content string to content array with text object
            # Note: conversations is a list of conversation lists, where each inner list
            # contains the full conversation (system, user, assistant messages)
            def transform_row(row):
                if "conversations" not in row or not row["conversations"]:
                    return {"messages": []}

                # Get the first conversation (conversations is a list of lists)
                conversation = (
                    row["conversations"][0]
                    if isinstance(row["conversations"][0], list)
                    else row["conversations"]
                )

                messages = []
                for conv in conversation:
                    if isinstance(conv, dict):
                        message = {
                            "role": conv.get("role", "user"),
                            "content": [
                                # Synthetic data kit always generates text content only
                                {"text": conv.get("content", ""), "type": "text"}
                            ],
                        }
                        messages.append(message)

                return {"messages": messages}

            dataset = dataset.map(transform_row)
            return dataset.select_columns(["messages"])

        finally:
            # Clean up request-specific config file
            if request_config_path.exists():
                request_config_path.unlink()
