import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from datasets import DatasetDict
import base64
import io
import json
from PIL import Image

logger = logging.getLogger(__name__)


class FormatConverter:
    """
    A class that handles conversion of datasets to various conversational formats.

    This class provides functionality to convert various dataset formats into
    standardized formats for conversational AI training data, including ChatML
    format for language modeling and specialized formats for prompt-only training.

    Supports two main processing modes:
    - LANGUAGE_MODELING: Converts to ChatML format with messages field
    - PROMPT_ONLY: Converts to prompt-only format with prompt, answer, reasoning fields
    - PREFERENCE: Converts to preference format with prompt, chosen, and rejected fields

    Attributes:
        whitespace_pattern (Pattern): Regular expression pattern for normalizing whitespace

    Example:
        >>> converter = FormatConverter()
        >>> data = converter.convert_to_conversational_chatml(dataset, "language_modeling", config)
    """

    def __init__(self):
        """
        Initialize the FormatConverter with a whitespace normalization pattern.
        """
        self.whitespace_pattern = re.compile(r"\s+")

    def convert_to_conversational_chatml(
        self, dataset: DatasetDict, processing_mode: str, config: Dict[str, Any]
    ) -> Tuple[DatasetDict, str]:
        """
        Convert a dataset to conversational format based on the processing mode.

        This is the main conversion method that handles different processing modes:
        - language_modeling: Converts to ChatML format with messages field
        - prompt_only: Converts to prompt-only format with prompt, answer, reasoning fields
        - preference: Converts to preference format with prompt, chosen, and rejected fields

        Args:
            dataset (DatasetDict): The input dataset with splits to convert
            processing_mode (str): The processing mode ("language_modeling", "prompt_only", or "preference")
            config (Dict[str, Any]): Configuration for the conversion, including field_mappings

        Returns:
            DatasetDict: Converted dataset in the appropriate format
            str: modality of the dataset

        Raises:
            ValueError: If processing_mode is not supported
            Exception: If there's an error during conversion
        """
        try:
            # Map processing modes to their conversion functions and validation fields
            # required_fields are already validated by model validator for the config structure
            mode_mapping = {
                "language_modeling": {
                    "converter": self._convert_single_example,
                    "filter_key": "messages",
                },
                "prompt_only": {
                    "converter": self._convert_single_example_prompt_only,
                    "filter_key": "prompt",
                },
                "preference": {
                    "converter": self._convert_single_example_preference,
                    "filter_key": "prompt",
                },
            }

            if processing_mode not in mode_mapping:
                raise ValueError(f"Unsupported processing mode: {processing_mode}")

            return self._convert_dataset_generic(
                dataset, config, mode_mapping[processing_mode]
            )

        except Exception as e:
            logger.error(
                f"Error converting dataset with mode {processing_mode}: {str(e)}"
            )
            raise

    def _convert_dataset_generic(
        self, dataset: DatasetDict, config: Dict[str, Any], mode_config: Dict[str, Any]
    ) -> Tuple[DatasetDict, str]:
        """
        Generic dataset conversion function that can be used for any processing mode.

        Args:
            dataset (DatasetDict): The input dataset with splits to convert
            config (Dict[str, Any]): Configuration for the conversion, including field_mappings
            mode_config (Dict[str, Any]): Mode-specific configuration containing:
                - converter: The single example conversion function
                - filter_key: The key to check in filtered results

        Returns:
            DatasetDict: Converted dataset in the appropriate format
            str: modality of the dataset

        Raises:
            Exception: If there's an error during conversion
        """
        try:
            if not dataset:
                raise ValueError("Dataset is empty")

            # Validate field mappings refer to real columns
            field_mappings = config.get("field_mappings", {})
            first_split = dataset[next(iter(dataset))]
            available_columns = set(first_split.column_names)

            self._validate_field_mappings(field_mappings, available_columns)

            # Check if we have any image fields in the configuration
            has_image_fields = self._has_image_fields(field_mappings)
            logger.info(f"Has image fields: {has_image_fields}")
            modality = "vision" if has_image_fields else "text"

            transformed_dataset = dataset.map(
                mode_config["converter"],
                fn_kwargs={
                    "field_mappings": field_mappings,
                    "is_multimodal": has_image_fields,
                },
                batched=False,
                remove_columns=dataset[next(iter(dataset))].column_names,
            )

            # Filter out failed conversions (empty dictionaries)
            filter_key = mode_config["filter_key"]
            transformed_dataset = transformed_dataset.filter(
                lambda x: isinstance(x, dict)
                and filter_key in x
                and len(x[filter_key]) > 0
            )

            logger.info(f"Converted dataset splits: {list(transformed_dataset.keys())}")
            for split_name, split_data in transformed_dataset.items():
                logger.info(
                    f"Converted split {split_name} has {len(split_data)} examples"
                )

            return transformed_dataset, modality

        except Exception as e:
            logger.error(f"Error converting dataset: {str(e)}")
            raise

    def _convert_single_example(
        self,
        example: Dict,
        field_mappings: Dict[str, Dict[str, Any]],
        is_multimodal: bool = False,
    ) -> Dict:
        """
        Convert a single example to ChatML format.

        This method converts one example from the input format to ChatML format,
        applying field mappings, templates, and ensuring proper message structure.
        Designed to work with dataset.map() functionality.

        Args:
            example (Dict): The input example to convert
            field_mappings (Dict): Maps input fields to ChatML roles with type and value:
                - type: "column" or "template"
                - value: column name or template string with {column} references

        Returns:
            Dict: The converted example in ChatML format with messages field, or empty dict if conversion fails

            The language modelling type follows this structure (this is each row, i.e. dataset[0])
            ```json
            {
                "messages": [
                    {"role": "system", "content": [{"type": "text", "text": "System message"}]},
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "User message"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "Assistant response"}]}
                ],
                "images": [<PIL.Image.Image object>, ...]
            }
            ```
        """
        try:
            messages: List[Dict[str, Any]] = []
            all_images: List[Any] = []

            # System message
            sys_msg = self._create_system_message(example, field_mappings)
            if sys_msg:
                messages.append(sys_msg)

            user_msg, user_images = self._create_user_message(
                example, field_mappings, is_multimodal=is_multimodal
            )
            if user_msg:
                messages.append(user_msg)
                all_images.extend(user_images)

            # Assistant message
            assistant_msg = self._create_assistant_message(example, field_mappings)
            if assistant_msg:
                messages.append(assistant_msg)

            # Validate -- this sometimes fail for a few samples but we still return a empty dict instead of raising an error
            if self._validate_messages(messages):
                return {"messages": messages, "images": all_images}
            else:
                logger.warning("Validation failed for example, skipping.")
                return {"messages": [], "images": []}

        except Exception as e:
            logger.error(f"Failed to convert single example: {e}")
            return {"messages": [], "images": []}

    def _convert_single_example_prompt_only(
        self,
        example: Dict,
        field_mappings: Dict[str, Dict[str, Any]],
        is_multimodal: bool = False,
    ) -> Dict:
        """
        Convert a single example to prompt-only format.

        This method converts one example from the input format to prompt-only format,
        which includes a prompt field (list of messages) and additional fields like
        answer and reasoning that are used by the reward function but ignored by
        the trainer (e.g., GRPOTrainer).

        Args:
            example (Dict): The input example to convert
            field_mappings (Dict): Maps input fields with type and value:
                - system_field: System message (optional)
                - user_field: The main prompt content
                - Any other fields: Additional data fields like answer, reasoning, etc. (optional)
                - image fields: Images to include in user message (optional)

        Returns:
            Dict: The converted example in prompt-only format with prompt, and additional fields,
                  or empty dict if conversion fails

            The prompt-only type follows this structure:
            ```json
            {
                "prompt": [
                    {"role": "system", "content": [{"type": "text", "text": "System message"}]},
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "User prompt"}]}
                ],
                "answer": "Expected answer",
                "reasoning": "Optional reasoning",
                "images": [<PIL.Image.Image object>, ...]
            }
            ```
        """
        try:
            prompt_messages: List[Dict[str, Any]] = []
            all_images: List[Any] = []
            result = {}

            # System message for the prompt
            # NOTE: since this is optional it will be ignored if not present
            sys_msg = self._create_system_message(example, field_mappings)
            if sys_msg:
                prompt_messages.append(sys_msg)

            user_msg, user_images = self._create_user_message(
                example, field_mappings, is_multimodal=is_multimodal
            )
            if user_msg:
                prompt_messages.append(user_msg)
                all_images.extend(user_images)

            if prompt_messages:
                result["prompt"] = prompt_messages
                if is_multimodal:
                    result["images"] = all_images
            else:
                logger.warning("No prompt content found in example")
                return {}

            # Extract additional fields (answer, reasoning, etc.) that are used by reward function
            # Accept any field that is not a special field and is either column or template type
            special_fields = {"system_field", "user_field"}
            for field_key in field_mappings.keys():
                if field_key not in special_fields:
                    result[field_key] = self._extract_text_content(
                        example, field_mappings, field_key
                    )

            return result

        except Exception as e:
            logger.error(f"Failed to convert single example to prompt-only: {e}")
            return {}

    def _convert_single_example_preference(
        self,
        example: Dict,
        field_mappings: Dict[str, Dict[str, Any]],
        is_multimodal: bool = False,
    ) -> Dict:
        """
        Convert a single example to preference format.

        This method converts one example from the input format to preference format,
        which includes a prompt field (list of messages), chosen response, and rejected response
        used by preference-based trainers like DPOTrainer.

        Args:
            example (Dict): The input example to convert
            field_mappings (Dict): Maps input fields with type and value:
                - system_field: System message (optional)
                - user_field: The main prompt content
                - chosen_field: The preferred response
                - rejected_field: The rejected response

        Returns:
            Dict: The converted example in preference format with prompt, chosen, and rejected fields,
                  or empty dict if conversion fails

        The preference type follows this structure:
        ```json
        {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": "What color is the sky?" },
                        { "type": "image"}
                    ]
                }
            ],
            "chosen": [
                {
                    "role": "assistant",
                    "content": [{ "type": "text", "text": "It is blue." }]
                }
            ],
            "rejected": [
                {
                    "role": "assistant",
                    "content": [{ "type": "text", "text": "It is green." }]
                }
            ],
            "images": [<PIL.Image.Image object>, ...]
        }
        ```
        """
        try:
            result = {}
            prompt_messages: List[Dict[str, Any]] = []
            all_images: List[Any] = []

            system_msg = self._create_system_message(example, field_mappings)
            if system_msg:
                prompt_messages.append(system_msg)

            user_msg, user_images = self._create_user_message(
                example, field_mappings, is_multimodal=is_multimodal
            )
            prompt_messages.append(user_msg)
            all_images.extend(user_images)

            if user_msg:
                result["prompt"] = prompt_messages
                if is_multimodal:
                    result["images"] = all_images
            else:
                logger.warning("No prompt content found in example")
                return {}

            for field in ["chosen", "rejected"]:
                field_config = field_mappings.get(f"{field}_field")
                if not field_config:
                    logging.warning(f"Field {field} not found in config!")
                    result[field] = []
                    continue

                # Check for pre-formatted message first
                if (
                    field_config["type"] == "column"
                    and field_config["value"] in example
                ):
                    content = example[field_config["value"]]
                    pre_formatted = self._extract_pre_formatted_message(content)
                    if pre_formatted:
                        result[field] = [pre_formatted]
                        continue

                # Handle regular content
                text_content = self._extract_text_content(
                    example, field_mappings, f"{field}_field"
                )

                if text_content:
                    result[field] = [
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": text_content}],
                        }
                    ]
                else:
                    result[field] = []
                    logger.warning("No rejected response found in example")

            return result

        except Exception as e:
            logger.error(f"Failed to convert single example to preference: {e}")
            logger.error(
                f"Example keys: {list(example.keys()) if isinstance(example, dict) else 'Not a dict'}"
            )
            logger.error(f"Field mappings: {field_mappings}")
            return {}

    def _extract_text_content(
        self, example: Dict, field_mappings: Dict, field_key: str
    ) -> str:
        """
        Extract and format text content for a specific role from the example.

        This method extracts content from the input example based on field mappings
        and applies template formatting if specified. It also normalizes whitespace
        in the content.

        Args:
            example (Dict): The input example
            field_mappings (Dict): Maps roles to field mapping configs
            role (str): The role to extract content for ('user' or 'assistant')

        Returns:
            str: The extracted and formatted content

        Example:
            >>> example = {"question": "What is ML?"}
            >>> field_mappings = {
            ...     "user_field": {"type": "column", "value": "question"}
            ... }
            >>> content = converter._extract_text_content(example, field_mappings, "user_field")
        """
        field_config = field_mappings.get(field_key)
        if not field_config:
            return ""

        if field_config["type"] == "column":
            if field_config["value"] not in example:
                return ""
            content = example[field_config["value"]]
        elif field_config["type"] == "template":
            try:
                template_vars = {key: str(value) for key, value in example.items()}
                content = field_config["value"].format(**template_vars)
            except (KeyError, ValueError) as e:
                logger.warning(f"Template formatting failed: {e}, using raw template")
                content = field_config["value"]
        else:
            raise ValueError(
                f"Unsupported field type '{field_config['type']}' for key '{field_key}'"
            )

        if isinstance(content, str):
            content = self.whitespace_pattern.sub(" ", content).strip()

        return str(content)

    @DeprecationWarning(
        "User message creation no longer requires extracting multimodal content but directly creates a new images column to match the new TRL docs"
    )
    def _extract_multimodal_content(
        self,
        example: Dict,
        field_mappings: Dict,
        role: str,
    ) -> List[Dict[str, Any]]:
        """
        Extract and format multimodal content (text + images) for a specific role.

        This method extracts both text and image content from the input example
        and formats it for vision ChatML format. Images are only added to user messages.

        Args:
            example (Dict): The input example
            field_mappings (Dict): Maps roles to field mapping configs
            role (str): The role to extract content for ('user' or 'assistant')

        Returns:
            List[Dict[str, Any]]: List of content items with type and data

        Example:
            >>> example = {"question": "What is in this image?", "image": PIL_Image}
            >>> field_mappings = {
            ...     "user_field": {"type": "column", "value": "question"},
            ...     "image_field": {"type": "image", "value": "image"}
            ... }
            >>> content = converter._extract_multimodal_content(example, field_mappings, "user")
            >>> [
            ...     {"type": "text", "text": "What is in this image?"},
            ...     {"type": "image", "image": PIL_Image_object}
            ... ]

        NOTE: After serialisation (parquet) the `PIL_Image_object` becomes:

            "image": {
                "bytes": "base64_encoded_image_data",
                "path": null
            }

        When bytes cannot be used (e.g. API response) we convert to base64 string.
        Otherwise, we keep the PIL Image object (e.g. for training).
        """
        content_items = []

        # Extract text content using the existing method
        text_content = self._extract_text_content(
            example, field_mappings, f"{role}_field"
        )
        if text_content:
            content_items.append({"type": "text", "text": text_content})

        # Extract image content - ONLY for user messages
        if role == "user":
            # Find all image field mappings
            for _, field_config in field_mappings.items():
                if field_config.get("type") == "image":
                    image_column = field_config.get("value")
                    if image_column and image_column in example:
                        image_data = example[image_column]
                        if image_data is not None:
                            processed_image = self._process_image_field(image_data)
                            if processed_image:
                                content_items.append(
                                    {"type": "image", "image": processed_image}
                                )
                            else:
                                logger.warning(
                                    f"Failed to process image field '{image_column}' for {role}"
                                )

        logger.debug(f"Final content items for {role}: {len(content_items)} items")
        return content_items

    def _process_image_field(self, image_data: Any) -> Optional[Any]:
        """
        Process image data and keep as PIL Image for efficient storage.

        This approach is optimized for the overall system:
        1. PIL Images are efficiently stored in parquet format by datasets library
        2. Training services can directly use PIL Images from parquet
        3. Only convert to base64 when needed for API responses or inference

        Args:
            image_data: Image data in various formats

        Returns:
            Optional[PIL.Image]: PIL Image object or None if processing fails
        """
        try:
            # Already a PIL Image - return as-is
            if Image and isinstance(image_data, Image.Image):
                # logger.info("Image is already a PIL Image, converting to RGB")
                return image_data.convert("RGB")

            # Dict with bytes (HuggingFace dataset format)
            elif isinstance(image_data, dict) and "bytes" in image_data:
                # logger.info("Image is in HuggingFace format with bytes field")
                image_bytes = image_data["bytes"]
                if Image:
                    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    return pil_image

            # Base64 encoded string
            elif isinstance(image_data, str):
                if image_data.startswith("data:image/"):
                    # Data URL format
                    # logger.info("Image is a data URL")
                    header, data = image_data.split(",", 1)
                    image_bytes = base64.b64decode(data)
                    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    return pil_image
                elif self._is_base64_image(image_data):
                    # Regular base64
                    # logger.info("Image is regular base64")
                    image_bytes = base64.b64decode(image_data)
                    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    return pil_image
                else:
                    # Might be a file path
                    try:
                        pil_image = Image.open(image_data).convert("RGB")
                        return pil_image
                    except Exception as path_error:
                        logger.warning(f"Failed to open image from path: {path_error}")
                        return None

            # Raw bytes
            elif isinstance(image_data, (bytes, bytearray)):
                pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
                return pil_image

            else:
                logger.error(f"Unsupported image data type: {type(image_data)}")
                return None

        except Exception as e:
            logger.error(f"Error processing image field: {str(e)}")
            logger.error(f"Image data type: {type(image_data)}")
            if hasattr(image_data, "__len__"):
                logger.error(f"Image data length: {len(image_data)}")
            return None

    def _is_base64_image(self, value: str) -> bool:
        """
        Check if a string is a base64 encoded image.

        Args:
            value: String to check

        Returns:
            bool: True if string appears to be base64 encoded image
        """
        try:
            # Check for data URL format
            if value.startswith("data:image/"):
                return True

            # Try to decode as base64
            if len(value) > 100:  # Reasonable minimum for image data
                decoded = base64.b64decode(value)
                return len(decoded) > 100

        except Exception:
            pass

        return False

    def _has_image_fields(self, field_mappings: Dict[str, Any]) -> bool:
        """
        Check if field mappings contain any image fields within user_field.

        Args:
            field_mappings: Dictionary of field mappings

        Returns:
            bool: True if user_field contains any image types
        """
        user_field = field_mappings.get("user_field")
        if not user_field:
            return False

        # Handle new format: List[Dict]
        if isinstance(user_field, list):
            return any(item.get("type") == "image" for item in user_field)

        # Handle backward compatibility: single Dict
        return user_field.get("type") == "image"

    def _create_system_message(
        self, example: Dict, field_mappings: Dict
    ) -> Optional[Dict]:
        """
        Create a system message from the example data.

        Args:
            example (Dict): The input example
            field_mappings (Dict): Field mappings configuration

        Returns:
            Optional[Dict]: System message dict or None if no system field
        """
        if "system_field" not in field_mappings:
            return None

        system_config = field_mappings["system_field"]

        # Check for pre-formatted message first
        if system_config["type"] == "column" and system_config["value"] in example:
            content = example[system_config["value"]]
            pre_formatted = self._extract_pre_formatted_message(content)
            if pre_formatted:
                return pre_formatted

        # Handle regular content
        system_message = ""
        if system_config["type"] == "column":
            if system_config["value"] in example:
                system_message = str(example[system_config["value"]])
        else:  # template
            try:
                template_vars = {key: str(value) for key, value in example.items()}
                system_message = system_config["value"].format(**template_vars)
            except (KeyError, ValueError) as e:
                logger.warning(f"System message template formatting failed: {e}")
                return None

        if system_message:
            return {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            }
        return None

    def _create_user_message(
        self, example: Dict, field_mappings: Dict, is_multimodal: bool = False
    ) -> Tuple[Optional[Dict], List[Any]]:
        """
        Create a user message, separating content from image data.

        Args:
            example (Dict): The input example
            field_mappings (Dict): Field mappings configuration
            is_multimodal (bool): Whether to include image content

        Returns:
            A tuple containing:
            - Optional[Dict]: User message dict with image placeholders.
            - List[Any]: A list of actual PIL.Image objects.
        """
        user_field_config = field_mappings.get("user_field")
        if not user_field_config:
            return None, []

        content_items = []
        image_items = []

        # The new format expects user_field to be a list of content parts
        if not isinstance(user_field_config, list):
            logger.warning(
                "user_field is not a list, attempting backward compatible conversion."
            )
            # Fallback for old single-dict format might be needed or deprecated
            # For this migration, we focus on the new list-based format.
            user_field_config = [user_field_config]

        for item in user_field_config:
            item_type = item.get("type")
            item_value = item.get("value")

            if item_type == "template":
                try:
                    template_vars = {key: str(value) for key, value in example.items()}
                    text = item_value.format(**template_vars)
                    text = self.whitespace_pattern.sub(" ", text).strip()
                    if text:
                        content_items.append({"type": "text", "text": text})
                except (KeyError, ValueError) as e:
                    logger.warning(f"Template formatting failed: {e}")

            elif item_type == "column":
                if item_value and item_value in example:
                    content = example[item_value]
                    pre_formatted = self._extract_pre_formatted_message(content)
                    if pre_formatted and isinstance(pre_formatted.get("content"), list):
                        content_items.extend(pre_formatted["content"])
                    else:
                        text = str(content)
                        text = self.whitespace_pattern.sub(" ", text).strip()
                        if text:
                            content_items.append({"type": "text", "text": text})

            elif item_type == "image" and is_multimodal:
                if item_value and item_value in example:
                    image_data = example[item_value]
                    if image_data is not None:
                        processed_image = self._process_image_field(image_data)
                        if processed_image:
                            # Add placeholder to content, and real image to image_items
                            content_items.append({"type": "image"})
                            image_items.append(processed_image)

        # Filter out empty content
        content_items = [
            item
            for item in content_items
            if item.get("text", "").strip() or item.get("type") == "image"
        ]

        if content_items:
            return {"role": "user", "content": content_items}, image_items

        return None, []

    def _create_assistant_message(
        self, example: Dict, field_mappings: Dict
    ) -> Optional[Dict]:
        """
        Create an assistant message from the example data.

        Args:
            example (Dict): The input example
            field_mappings (Dict): Field mappings configuration

        Returns:
            Optional[Dict]: Assistant message dict or None if no content
        """
        assistant_field_config = field_mappings.get("assistant_field")
        if not assistant_field_config:
            return None

        # Check for pre-formatted message first
        if (
            assistant_field_config["type"] == "column"
            and assistant_field_config["value"] in example
        ):
            content = example[assistant_field_config["value"]]
            pre_formatted = self._extract_pre_formatted_message(content)
            if pre_formatted:
                return pre_formatted

        # Handle regular content
        assistant_content = self._extract_text_content(
            example, field_mappings, "assistant_field"
        )
        if assistant_content:
            return {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_content}],
            }
        return None

    def _extract_pre_formatted_message(self, content: Any) -> Optional[Dict]:
        """
        Extract and normalize pre-formatted message if content is already in ChatML format.

        Checks if content is a pre-formatted message (dict with role/content or JSON string)
        and returns it with normalized content format.

        Args:
            content: Content to check and potentially extract

        Returns:
            Optional[Dict]: Pre-formatted message dict with normalized content or None if not pre-formatted
        """
        pre_formatted_msg = None

        # Check if it's already a dict with role/content
        if isinstance(content, dict) and "role" in content and "content" in content:
            pre_formatted_msg = content

        # Check if it's a JSON string that represents a message
        elif isinstance(content, str):
            # Simple regex to check if it looks like a JSON object with role and content
            pattern = r'^\s*\{.*"role".*"content".*\}\s*$'
            if re.search(pattern, content, re.DOTALL):
                try:
                    parsed = json.loads(content)
                    if (
                        isinstance(parsed, dict)
                        and "role" in parsed
                        and "content" in parsed
                    ):
                        pre_formatted_msg = parsed
                except (json.JSONDecodeError, ValueError):
                    pass

        # If we found a pre-formatted message, normalize its content to ChatML list format
        if pre_formatted_msg:
            # Normalize content to always be a list (ChatML format)
            if isinstance(pre_formatted_msg.get("content"), str):
                pre_formatted_msg["content"] = [
                    {"type": "text", "text": pre_formatted_msg["content"]}
                ]
            elif not isinstance(pre_formatted_msg.get("content"), list):
                pre_formatted_msg["content"] = [
                    {"type": "text", "text": str(pre_formatted_msg.get("content", ""))}
                ]

            return pre_formatted_msg

        return None

    def _validate_field_mappings(
        self, field_mappings: Dict, available_columns: set
    ) -> None:
        """
        Validate field mappings against available columns and new structure rules.
        NOTE: This is different from the model_validate on the schema for config because this checks the content, the former checks the structure

        Args:
            field_mappings: Field mappings configuration
            available_columns: Set of available column names in the dataset

        Raises:
            ValueError: If validation fails
        """
        for field_key, field_config in field_mappings.items():
            # Handle the new user_field structure (can be List[Dict] or Dict)
            if field_key == "user_field":
                if isinstance(field_config, list):
                    # New format: List[Dict] with mixed content
                    for item in field_config:
                        if not isinstance(item, dict):
                            raise ValueError(
                                f"user_field list items must be dictionaries, got {type(item)}"
                            )
                        self._validate_single_field_config(
                            item, available_columns, field_key
                        )
                else:
                    # Backward compatibility: single Dict
                    self._validate_single_field_config(
                        field_config, available_columns, field_key
                    )
            else:
                # All other fields must be single Dict and cannot have type="image"
                if isinstance(field_config, list):
                    raise ValueError(
                        f"Field '{field_key}' cannot be a list. Only user_field supports list format."
                    )

                if field_config.get("type") == "image":
                    raise ValueError(
                        f"Image fields are only allowed within user_field. Found image field: '{field_key}'"
                    )

                self._validate_single_field_config(
                    field_config, available_columns, field_key
                )

    def _validate_single_field_config(
        self, field_config: Dict, available_columns: set, field_key: str
    ) -> None:
        """
        Validate a single field configuration.

        Args:
            field_config: Single field configuration dict
            available_columns: Set of available column names
            field_key: Name of the field being validated
        """
        if field_config.get("type") in ("column", "image"):
            column_name = field_config.get("value")
            if column_name not in available_columns:
                raise ValueError(
                    f"Field mapping '{field_key}' refers to column '{column_name}' "
                    f"which does not exist in dataset. Available columns: {sorted(available_columns)}"
                )

    def _validate_messages(self, messages: List[Dict]) -> bool:
        """
        Validate that we have the required messages for a valid conversation for language modeling datastes.
        It requires the message to have user and assistant fields.
        NOTE: This is not applicable to any other dataset types.

        Args:
            messages (List[Dict]): List of message dictionaries

        Returns:
            bool: True if messages are valid for training
        """
        has_user = any(msg["role"] == "user" for msg in messages)
        has_assistant = any(msg["role"] == "assistant" for msg in messages)
        return has_user and has_assistant
