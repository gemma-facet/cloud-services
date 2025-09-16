import base64
import io
from typing import List, Tuple
from PIL import Image
import logging

logger = logging.getLogger(__name__)


def infer_storage_type_from_path(adapter_path: str) -> str:
    """
    Infer storage type from adapter path.

    Args:
        adapter_path: Path to adapter (local, GCS, or HF Hub)

    Returns:
        str: Either "local", "gcs", or "hfhub"
    """
    if adapter_path.startswith("gs://"):
        return "gcs"
    elif (
        "/" in adapter_path
        and not adapter_path.startswith("/")
        and not adapter_path.startswith("./")
    ):
        # Heuristic: if it contains "/" but doesn't start with "/" or "./" it's likely a HF Hub repo
        return "hfhub"
    else:
        # Local path (absolute or relative)
        return "local"


def infer_modality_from_messages(messages: List) -> str:
    """
    Infer the modality (text or vision) from the messages.

    Supports both batch inference and evaluation paths:
    - Batch inference: base64 strings in message content
    - Evaluation: PIL Image objects in message content

    Args:
        messages: List of message conversations in ChatML format

    Returns:
        str: Either "text" or "vision"
    """
    for msgs in messages:
        if isinstance(msgs, list):
            for msg in msgs:
                if isinstance(msg.get("content"), list):
                    for content in msg["content"]:
                        if content.get("type") == "image":
                            return "vision"
    return "text"


def prepare_vision_inputs(
    processor, messages: List
) -> Tuple[List[List[Image.Image]], List[str]]:
    """
    Prepare vision inputs for batch processing.

    Handles TWO scenarios with the SAME input format List[List[Dict]]:

    1. Batch Inference (frontend): Images are base64 strings
       [
           [
               {"role": "user", "content": [
                   {"type": "image", "image": "<base64_string>"},
                   {"type": "text", "text": "Describe the image."}
               ]},
               ...
           ]
       ]

    2. Evaluation (GCS dataset): Images are PIL objects (converted by storage service)
       [
           [
               {"role": "user", "content": [
                   {"type": "image", "image": <PIL.Image.Image>},
                   {"type": "text", "text": "Describe the image."}
               ]},
               ...
           ]
       ]

    Args:
        processor: The model processor for handling vision inputs
        messages: List of message conversations in ChatML format

    Returns:
        Tuple containing:
            - images: [[Image objects], [Image objects], ...]
            - texts: ["Formatted text prompt", ...]
            - len(images) == len(texts) == batch_size
    """
    images = []
    texts = []

    for msgs in messages:
        # Extract images from message content
        raw_images = []
        for msg in msgs:
            if isinstance(msg.get("content"), list):
                for item in msg["content"]:
                    if item.get("type") == "image" and "image" in item:
                        raw_images.append(item["image"])

        if not raw_images:
            raise ValueError("Image content not found in vision prompt")

        # Convert to PIL Images (handle both base64 strings and PIL objects)
        pil_images = []
        for img_data in raw_images:
            if isinstance(img_data, Image.Image):
                # Already PIL Image (from evaluation path)
                pil_images.append(img_data)
            elif isinstance(img_data, str):
                # Base64 string (from batch inference path)
                pil_images.append(_decode_base64_image(img_data))
            else:
                raise ValueError(f"Unsupported image type: {type(img_data)}")

        images.append(pil_images)

        # Prepare text content for this conversation
        text = processor.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
        ).strip()
        texts.append(text)

    return images, texts


def _decode_base64_image(image_content: str) -> Image.Image:
    """
    Helper function to decode base64 image string to PIL Image.

    Args:
        image_content: Base64 encoded image string (may include data URI prefix)

    Returns:
        PIL Image object
    """
    # Handle data URI header (e.g., "data:image/png;base64,...")
    if "," in image_content:
        image_content = image_content.split(",", 1)[1]

    # Pad base64 if needed
    padding = len(image_content) % 4
    if padding:
        image_content += "=" * (4 - padding)

    return Image.open(io.BytesIO(base64.b64decode(image_content)))


def get_model_device_config():
    """
    Get appropriate device configuration for model loading.

    Returns:
        dict: Model kwargs for device and dtype configuration
    """
    import torch

    return {
        "torch_dtype": torch.float16
        if torch.cuda.get_device_capability()[0] < 8
        else torch.bfloat16,
        "device_map": "auto",
    }


def get_stop_tokens(tokenizer):
    """
    Get appropriate stop tokens for generation.

    Args:
        tokenizer: Model tokenizer

    Returns:
        list: List of stop token IDs
    """
    stop_tokens = [tokenizer.eos_token_id]

    # Add Gemma-specific stop token if available
    try:
        gemma_stop = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        if gemma_stop is not None:
            stop_tokens.append(gemma_stop)
    except (KeyError, ValueError):
        pass  # Token doesn't exist, skip

    return stop_tokens
