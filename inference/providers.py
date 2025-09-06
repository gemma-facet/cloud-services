import logging
import os
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
        self,
        base_model_id: str,
        resolved_model_path: str,
        model_type: str,
        messages: List,
        modality: str,
    ) -> List[str]:
        """
        Run batch inference for the provider's framework.

        Args:
            base_model_id: Base model identifier
            resolved_model_path: Resolved path that can be used directly by the ML framework
                For GCS: local downloaded path ("/tmp/...")
                For HF Hub: remote repo ID ("user/repo")
            model_type: Type of model ("adapter", "merged", or "base")
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
        self,
        base_model_id: str,
        resolved_model_path: str,
        model_type: str,
        messages: List,
        modality: str,
    ) -> List[str]:
        """Run batch inference using HuggingFace Transformers"""
        if modality == "vision":
            return self._run_batch_inference_vision(
                base_model_id, resolved_model_path, model_type, messages
            )
        else:
            return self._run_batch_inference_text(
                base_model_id, resolved_model_path, model_type, messages
            )

    def _run_batch_inference_text(
        self,
        base_model_id: str,
        resolved_model_path: str,
        model_type: str,
        messages: List,
    ) -> List[str]:
        """Run text-based batch inference using HuggingFace Transformers"""
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        # Model configuration
        model_kwargs = get_model_device_config()

        if model_type == "base":
            # non base models already have quant config saved, but base model we always load bnb 4bit quant
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        # Load appropriate model class based on base model -- 1B and 270M, others including 3n are vision
        if "1b" in base_model_id or "270m" in base_model_id:
            # This can directly load adapter AND merged model, no need PEFT to load adapters explicitly
            model = AutoModelForCausalLM.from_pretrained(
                resolved_model_path, **model_kwargs
            )
        else:
            model = AutoModelForImageTextToText.from_pretrained(
                resolved_model_path, **model_kwargs
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
        resolved_model_path: str,
        model_type: str,
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
            resolved_model_path, **model_kwargs
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
        self,
        base_model_id: str,
        resolved_model_path: str,
        model_type: str,
        messages: List,
        modality: str,
    ) -> List[str]:
        """Run batch inference using Unsloth"""
        if modality == "vision":
            return self._run_batch_inference_vision(
                base_model_id, resolved_model_path, model_type, messages
            )
        else:
            return self._run_batch_inference_text(
                base_model_id, resolved_model_path, model_type, messages
            )

    def _run_batch_inference_text(
        self,
        base_model_id: str,
        resolved_model_path: str,
        model_type: str,
        messages: List,
    ) -> List[str]:
        """Run text-based batch inference using Unsloth"""
        import unsloth  # noqa: F401
        from unsloth import FastModel
        from unsloth.chat_templates import get_chat_template

        if model_type == "base":
            model, tokenizer = FastModel.from_pretrained(
                model_name=resolved_model_path,
                load_in_4bit=True,
                max_seq_length=2048,
            )
        else:
            # no need to set load_in_4bit here, we follow the saved model config
            model, tokenizer = FastModel.from_pretrained(
                model_name=resolved_model_path,
                max_seq_length=2048,
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
            use_cache=True,
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
        resolved_model_path: str,
        model_type: str,
        messages: List,
    ) -> List[str]:
        """Run vision-based batch inference using Unsloth"""
        import unsloth  # noqa: F401
        from unsloth import FastVisionModel
        from unsloth.chat_templates import get_chat_template

        if model_type == "base":
            model, processor = FastVisionModel.from_pretrained(
                model_name=resolved_model_path,
                load_in_4bit=True,
            )
        else:
            model, processor = FastVisionModel.from_pretrained(
                model_name=resolved_model_path
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
            max_new_tokens=256,
            use_cache=True,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
        )
        decoded = processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        return decoded


class VLLMInferenceProvider(BaseInferenceProvider):
    """
    vLLM-based inference provider.

    Supports merged and adapter export, both huggingface and unsloth text and vision models.
    Alternative can use the built-in unsloth vLLM integration with FastLanguageModel(fast_inference=True)
    but we chose to use vLLM directly here for more control and flexibility.
    """

    def run_batch_inference(
        self,
        base_model_id: str,
        resolved_model_path: str,
        model_type: str,
        messages: List,
        modality: str,
    ) -> List[str]:
        # This ensure that we use V1 engine even when running as background process (experimental)
        os.environ["VLLM_USE_V1"] = "1"
        if modality == "vision":
            return self._run_batch_inference_vision(
                base_model_id, resolved_model_path, model_type, messages
            )
        return self._run_batch_inference_text(
            base_model_id, resolved_model_path, model_type, messages
        )

    def _run_batch_inference_text(
        self,
        base_model_id: str,
        resolved_model_path: str,
        model_type: str,
        messages: List,
    ) -> List[str]:
        from vllm import LLM, SamplingParams

        if model_type == "adapter":
            from vllm.lora.request import LoRARequest

            # Load base model with LoRA support
            llm = LLM(model=base_model_id, enable_lora=True)
            tokenizer = llm.get_tokenizer()

            # Apply ChatML template to each conversation
            prompts = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            sampling_params = SamplingParams(
                max_tokens=256,
                temperature=1.0,
                top_p=0.95,
                top_k=64,
            )

            # Create LoRA request
            lora_request = LoRARequest("lora_adapter", 1, resolved_model_path)

            # Generate with LoRA
            outputs = llm.generate(
                prompts, sampling_params=sampling_params, lora_request=lora_request
            )
        else:  # merged or base model
            # Load merged vLLM model or base model
            model_path = (
                resolved_model_path if model_type == "merged" else base_model_id
            )

            # TODO: This is yet to be tested with vllm on-spot quantization with bnb, but supported on docs
            if model_type == "base":
                llm = LLM(model=model_path, quantization="bitsandbytes")
            else:
                llm = LLM(model=model_path)
            tokenizer = llm.get_tokenizer()

            # Apply ChatML template to each conversation
            prompts = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            sampling_params = SamplingParams(temperature=0.0, max_tokens=256)

            # Generate
            outputs = llm.generate(prompts, sampling_params=sampling_params)

        # Collect responses
        return [out.outputs[0].text for out in outputs]

    def _run_batch_inference_vision(
        self,
        base_model_id: str,
        resolved_model_path: str,
        model_type: str,
        messages: List,
    ) -> List[str]:
        from vllm import LLM, SamplingParams
        from transformers import AutoProcessor

        if model_type == "adapter":
            from vllm.lora.request import LoRARequest

            # Load base model with LoRA support for vision
            llm = LLM(model=base_model_id, enable_lora=True)
            processor = AutoProcessor.from_pretrained(base_model_id)

            # Create LoRA request
            lora_request = LoRARequest("lora_adapter", 1, resolved_model_path)

            sampling_params = SamplingParams(
                temperature=1.0, top_p=0.95, top_k=64, max_tokens=128
            )
        else:  # merged or base model
            # Load merged vLLM model or base model & corresponding processor
            model_path = (
                resolved_model_path if model_type == "merged" else base_model_id
            )
            llm = LLM(model=model_path)
            processor = AutoProcessor.from_pretrained(base_model_id)

            sampling_params = SamplingParams(
                temperature=0.7, top_p=0.95, max_tokens=100
            )

        prompts = []
        images = []
        # Build batch of ChatML prompts + image list
        for conv in messages:
            prompts.append(
                processor.apply_chat_template(
                    conv, tokenize=False, add_generation_prompt=True
                )
            )
            # extract images from the structured content
            for msg in conv:
                content = msg.get("content")
                if isinstance(content, list):
                    for segment in content:
                        if segment.get("type") == "image":
                            images.append(segment["image"])

        if model_type == "adapter":
            outputs = llm.generate(
                {"prompt": prompts, "multi_modal_data": {"image": images}},
                sampling_params=sampling_params,
                lora_request=lora_request,
            )
        else:
            outputs = llm.generate(
                {"prompt": prompts, "multi_modal_data": {"image": images}},
                sampling_params=sampling_params,
            )
        return [out.outputs[0].text for out in outputs]
