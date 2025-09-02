import logging
from typing import List
from abc import ABC, abstractmethod

from utils import prepare_vision_inputs, get_model_device_config, get_stop_tokens

logger = logging.getLogger(__name__)


class BaseInferenceProvider(ABC):
    """
    Abstract base class for inference providers.

    Defines the common interface that all inference providers must implement.
    Each provider handles the specifics of loading and running inference for
    their respective frameworks (HuggingFace Transformers, Unsloth, etc.).
    """

    @abstractmethod
    def run_batch_inference(
        self, base_model_id: str, adapter_path: str, messages: List, modality: str
    ) -> List[str]:
        """
        Run batch inference for the provider's framework.

        Args:
            base_model_id: Base model identifier
            adapter_path: Path to adapter (local or remote)
                NOTE: This can either be an adapter or merged model, the logic is handled by provider itself
            messages: List of message conversations
            modality: Either "text" or "vision"

        Returns:
            List of generated text responses
        """
        pass


class HuggingFaceInferenceProvider(BaseInferenceProvider):
    """
    HuggingFace Transformers-based inference provider.

    Handles inference for models using the standard HuggingFace Transformers library.
    Supports both text-only and vision models with appropriate model classes and processors.
    """

    def run_batch_inference(
        self, base_model_id: str, adapter_path: str, messages: List, modality: str
    ) -> List[str]:
        """Run batch inference using HuggingFace Transformers"""
        if modality == "vision":
            return self._run_batch_inference_vision(
                base_model_id, adapter_path, messages
            )
        else:
            return self._run_batch_inference_text(base_model_id, adapter_path, messages)

    def _run_batch_inference_text(
        self,
        base_model_id: str,
        adapter_path: str,
        messages: List,
    ) -> List[str]:
        """Run text-based batch inference using HuggingFace Transformers"""
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoTokenizer,
        )

        # Model configuration
        # NOTE: Does not need quantization_config because saved models already have one
        model_kwargs = get_model_device_config()

        # Load appropriate model class based on base model
        if base_model_id in ["google/gemma-3-1b-it", "google/gemma-3-1b-pt"]:
            # This can direclty load adapter AND merged model, no need PEFT to load adapters explicitly
            model = AutoModelForCausalLM.from_pretrained(adapter_path, **model_kwargs)
        else:
            model = AutoModelForImageTextToText.from_pretrained(
                adapter_path, **model_kwargs
            )

        tokenizer = AutoTokenizer.from_pretrained(base_model_id)

        # Prepare inputs
        chat_messages = [
            tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
            )
            for message in messages
        ]
        tokenizer.pad_token = tokenizer.eos_token
        batch_inputs = tokenizer(chat_messages, return_tensors="pt", padding=True).to(
            model.device
        )

        # Generate
        stop_tokens = get_stop_tokens(tokenizer)
        input_length = batch_inputs.input_ids.shape[1]
        generated_ids = model.generate(
            **batch_inputs,
            max_new_tokens=512,
            do_sample=False,
            top_k=50,
            eos_token_id=stop_tokens,
        )

        # Decode output
        decoded = tokenizer.batch_decode(
            generated_ids[:, input_length:], skip_special_tokens=True
        )
        return decoded

    def _run_batch_inference_vision(
        self,
        base_model_id: str,
        adapter_path: str,
        messages: List,
    ) -> List[str]:
        """Run vision-based batch inference using HuggingFace Transformers"""
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
        )
        from transformers.utils.quantization_config import BitsAndBytesConfig

        model_kwargs = get_model_device_config()
        # TODO: Is it necessary to set BnB quant config here?? or is it saved somehow already?
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        model = AutoModelForImageTextToText.from_pretrained(
            adapter_path, **model_kwargs
        )
        processor = AutoProcessor.from_pretrained(base_model_id)

        images, texts = prepare_vision_inputs(processor, messages)

        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=128)
        decoded = processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        return decoded


class UnslothInferenceProvider(BaseInferenceProvider):
    """
    Unsloth-based inference provider.

    Handles inference for models using the Unsloth framework, which provides
    optimized training and inference for LLMs. Supports both text and vision models.
    """

    def run_batch_inference(
        self, base_model_id: str, adapter_path: str, messages: List, modality: str
    ) -> List[str]:
        """Run batch inference using Unsloth"""
        if modality == "vision":
            return self._run_batch_inference_vision(
                base_model_id, adapter_path, messages
            )
        else:
            return self._run_batch_inference_text(base_model_id, adapter_path, messages)

    def _run_batch_inference_text(
        self,
        base_model_id: str,
        adapter_path: str,
        messages: List,
    ) -> List[str]:
        """Run text-based batch inference using Unsloth"""
        import unsloth  # noqa: F401
        from unsloth import FastModel
        from unsloth.chat_templates import get_chat_template

        model, tokenizer = FastModel.from_pretrained(
            model_name=adapter_path,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        FastModel.for_inference(model)
        tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
        tokenizer.pad_token = tokenizer.eos_token

        chat_messages = [
            tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in messages
        ]
        batch_inputs = tokenizer(chat_messages, return_tensors="pt", padding=True).to(
            "cuda"
        )
        generated_ids = model.generate(
            **batch_inputs,
            max_new_tokens=256,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
        )
        input_length = batch_inputs.input_ids.shape[1]
        decoded = tokenizer.batch_decode(
            generated_ids[:, input_length:], skip_special_tokens=True
        )
        return decoded

    def _run_batch_inference_vision(
        self,
        base_model_id: str,
        adapter_path: str,
        messages: List,
    ) -> List[str]:
        """Run vision-based batch inference using Unsloth"""
        import unsloth  # noqa: F401
        from unsloth import FastVisionModel
        from unsloth.chat_templates import get_chat_template

        model, processor = FastVisionModel.from_pretrained(
            model_name=adapter_path,
            load_in_4bit=True,
        )
        FastVisionModel.for_inference(model)
        processor = get_chat_template(processor, chat_template="gemma-3")

        images, texts = prepare_vision_inputs(processor, messages)

        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to("cuda")

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            use_cache=True,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
        )
        decoded = processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        return decoded
