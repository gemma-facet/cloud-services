import logging
import torch
from typing import Any, Tuple
from datasets import Dataset

from storage import storage_service
from base import BaseTrainingService
from rewards import load_reward_functions_from_config

# from utils import create_compute_metrics, preprocess_logits_for_metrics
from utils import create_compute_metrics
from schema import TrainingConfig


def _build_shared_training_args(
    trainer_type: str,
    cfg: TrainingConfig,
    job_id: str,
    report_to: str,
    provider_specific_args: dict,
    config_classes: dict,
) -> Any:
    """
    Shared logic for building complete training configuration across providers.

    This function handles all common logic including:
    - Base training arguments
    - Trainer-specific configuration
    - Evaluation setup
    - Vision-specific configuration

    Args:
        trainer_type: Type of trainer (sft, grpo, dpo)
        cfg: Training configuration
        job_id: Job identifier
        report_to: Reporting destination
        provider_specific_args: Provider-specific base arguments
        config_classes: Dict with trainer config classes (SFTConfig, GRPOConfig, DPOConfig)

    Returns:
        Configured training arguments object ready for trainer
    """
    hyperparam = cfg.hyperparameters
    evaluation = cfg.eval_config

    # Common base arguments
    base_args = {
        "output_dir": f"/tmp/{job_id}",
        "per_device_train_batch_size": hyperparam.batch_size,
        "gradient_accumulation_steps": hyperparam.gradient_accumulation_steps,
        "warmup_steps": 5,
        "learning_rate": hyperparam.learning_rate,
        "lr_scheduler_type": hyperparam.lr_scheduler_type or "linear",
        "weight_decay": 0.01,
        "save_strategy": hyperparam.save_strategy or "epoch",
        "push_to_hub": False,
        "logging_steps": hyperparam.logging_steps or 10,
        "report_to": report_to,
    }

    # Merge with provider-specific args
    base_args.update(provider_specific_args)

    # Trainer-specific configuration
    if trainer_type == "sft":
        trainer_args = {
            **base_args,
            "num_train_epochs": hyperparam.epochs,
            "max_steps": hyperparam.max_steps or -1,
            "packing": hyperparam.packing,
        }
        args = config_classes["sft"](**trainer_args)

    elif trainer_type == "grpo":
        # GRPO-specific parameters
        max_prompt_length = hyperparam.max_prompt_length
        max_length = hyperparam.max_length
        max_completion_length = max_length - max_prompt_length

        # This makes sure the entire image token is sent for vision
        if cfg.modality == "vision":
            max_prompt_length = None
            max_completion_length = max_length

        trainer_args = {
            **base_args,
            "max_steps": hyperparam.max_steps or 50,
            "num_generations": hyperparam.num_generations or 4,
            "max_prompt_length": max_prompt_length,
            "max_completion_length": max_completion_length,
            "max_grad_norm": hyperparam.max_grad_norm or 0.1,
            "adam_beta1": hyperparam.adam_beta1 or 0.9,
            "adam_beta2": hyperparam.adam_beta2 or 0.99,
            "warmup_ratio": hyperparam.warmup_ratio or 0.1,
            "remove_unused_columns": False,  # NOTE: MUST HAVE THIS to access the additional columns
            # Being worked on by unsloth right now add: "vllm>=0.10.0" later
            # GRPO is online method and vLLM is much faster at inference
            # "use_vllm": True,
            # NOTE: We cannot user "server" because there's only one GPU on cloud run for now...
            # Otherwise we will start another vLLM inference server
            # "vllm_mode": "colocate",
        }
        args = config_classes["grpo"](**trainer_args)

    elif trainer_type == "dpo":
        trainer_args = {
            **base_args,
            "num_train_epochs": hyperparam.epochs,
            "max_steps": hyperparam.max_steps or -1,
            "beta": hyperparam.beta or 0.1,
            "max_prompt_length": hyperparam.max_prompt_length
            if cfg.modality != "vision"
            else None,
            # max_length is max_completion_length + max_prompt_length
            "max_length": hyperparam.max_length,
        }
        args = config_classes["dpo"](**trainer_args)

    else:
        raise ValueError(f"Unsupported trainer type: {trainer_type}")

    # Set eval related fields if present (common to all trainers)
    if evaluation:
        args.eval_strategy = evaluation.eval_strategy or "no"
        if evaluation.eval_strategy == "steps":
            args.eval_steps = evaluation.eval_steps
        args.per_device_eval_batch_size = cfg.hyperparameters.batch_size
        args.eval_accumulation_steps = cfg.hyperparameters.gradient_accumulation_steps

        # Set eval precision based on provider-specific args
        if "fp16_full_eval" in provider_specific_args:
            args.fp16_full_eval = provider_specific_args["fp16"]
        if "bf16_full_eval" in provider_specific_args:
            args.bf16_full_eval = provider_specific_args["bf16"]

        # Set batch eval metrics if supported by config type
        if hasattr(args, "batch_eval_metrics"):
            args.batch_eval_metrics = evaluation.batch_eval_metrics

    # Vision-specific arguments (applies to all trainers when modality is vision)
    if cfg.modality == "vision":
        args.remove_unused_columns = False
        args.gradient_checkpointing = True
        args.gradient_checkpointing_kwargs = {"use_reentrant": False}

        # SFT-specific vision settings
        if hasattr(args, "dataset_kwargs"):
            if trainer_type == "sft":
                # HuggingFace needs skip_prepare_dataset for vision
                args.dataset_kwargs = {"skip_prepare_dataset": True}
            else:
                # Unsloth uses empty dict
                args.dataset_kwargs = {}

        if hasattr(args, "dataset_text_field"):
            args.dataset_text_field = ""  # dummy field for collator

        # Unsloth-specific vision settings
        if hasattr(args, "dataset_num_proc"):
            args.dataset_num_proc = 1

        # Set max length to None to avoid truncation of image tokens
        # This applies to all trainers when modality is vision
        if hasattr(args, "max_length"):
            args.max_length = None
        if hasattr(args, "max_prompt_length"):
            args.max_prompt_length = None
        if hasattr(args, "max_completion_length"):
            args.max_completion_length = None

    return args


class HuggingFaceTrainingService(BaseTrainingService):
    def __init__(self) -> None:
        # Import HF libraries only when service is instantiated
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            BitsAndBytesConfig,
            AutoModelForImageTextToText,
            AutoProcessor,
        )
        from peft import LoraConfig, get_peft_model
        from trl import (
            SFTTrainer,
            SFTConfig,
            GRPOTrainer,
            GRPOConfig,
            DPOTrainer,
            DPOConfig,
            DataCollatorForVisionLanguageModeling,
        )

        self.AutoTokenizer = AutoTokenizer
        self.AutoModelForCausalLM = AutoModelForCausalLM
        self.BitsAndBytesConfig = BitsAndBytesConfig
        self.AutoModelForImageTextToText = AutoModelForImageTextToText
        self.AutoProcessor = AutoProcessor
        self.LoraConfig = LoraConfig
        self.get_peft_model = get_peft_model
        self.SFTTrainer = SFTTrainer
        self.SFTConfig = SFTConfig
        self.GRPOTrainer = GRPOTrainer
        self.GRPOConfig = GRPOConfig
        self.DPOTrainer = DPOTrainer
        self.DPOConfig = DPOConfig
        self.DataCollatorForVisionLanguageModelling = (
            DataCollatorForVisionLanguageModeling
        )

        # Support both IT and PT models
        # no official quantised so we apply them later with bnb
        self.supported_models = [
            "google/gemma-3-1b-pt",
            "google/gemma-3-1b-it",
            "google/gemma-3-4b-pt",
            "google/gemma-3-4b-it",
            "google/gemma-3-12b-pt",
            "google/gemma-3-12b-it",
            "google/gemma-3n-E2B",
            "google/gemma-3n-E2B-it",
            "google/gemma-3n-E4B",
            "google/gemma-3n-E4B-it",
            "google/gemma-3-270m",
            "google/gemma-3-270m-it",
        ]

        # dtype based on GPU
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            self.torch_dtype = torch.bfloat16
        else:
            self.torch_dtype = torch.float16

    def _download_dataset(self, dataset_id: str) -> Tuple[Any, Any]:
        return storage_service.download_processed_dataset(dataset_id)

    def _setup_model(self, cfg: TrainingConfig) -> Tuple[Any, Any]:
        """
        Two criterias for model setup:

        1. If base_model_id is not 1b then use AutoModelForImageTextToText, else use AutoModelForCausalLM
        2. If modality is vision, use AutoProcessor otherwise use AutoTokenizer
        """
        base_model_id = cfg.base_model_id or "google/gemma-3-1b-it"
        if base_model_id not in self.supported_models:
            raise ValueError(
                f"Unsupported base model {cfg.base_model_id}. "
                f"Supported models: {self.supported_models}"
            )

        # Model kwargs for both text and vision
        model_kwargs = {
            "torch_dtype": self.torch_dtype,
            "device_map": "auto",
            "attn_implementation": "flash_attention_2"
            if cfg.hyperparameters.use_fa2
            else "eager",
        }

        # Load the model with proper quantisation if required
        # NOTE: This can be easily extended to support other quantization methods
        if cfg.method == "QLoRA":
            model_kwargs["quantization_config"] = self.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.torch_dtype,
            )

        # Use AutoModelForImageTextToText or AutoModelForCausalLM based on model id, NOT modality!!
        if base_model_id == "google/gemma-3-1b-it":
            model = self.AutoModelForCausalLM.from_pretrained(
                base_model_id, **model_kwargs
            )
        else:
            model = self.AutoModelForImageTextToText.from_pretrained(
                base_model_id, **model_kwargs
            )

        # Setup tokenizer or preprocessor based on modality
        if cfg.modality == "vision":
            if base_model_id == "google/gemma-3-1b-it":
                raise ValueError(
                    "Gemma 3.1B does not support vision fine-tuning. Use Gemma 3.4B or larger."
                )

            # Vision models use AutoProcessor
            processor = self.AutoProcessor.from_pretrained(
                base_model_id, trust_remote_code=True
            )
            processor.tokenizer.padding_side = "right"

            return model, processor
        else:
            tokenizer = self.AutoTokenizer.from_pretrained(base_model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            return model, tokenizer

    def _prepare_dataset_if_needed(
        self,
        train_ds: Dataset,
        eval_ds: Dataset,
        tokenizer_or_processor: Any,
        modality: str,
        trainer_type: str,
    ) -> Tuple[Dataset, Dataset]:
        """
        Optionally format dataset depending on modality and trainer type.
        The dataset is already in the new format, so we only need to handle special cases.
        """
        if modality == "vision" and trainer_type == "grpo":
            # GRPOTrainer expects a single "image" column, not "images"
            def add_image_column(example):
                example["image"] = example["images"][0] if example["images"] else None
                return example

            train_ds = train_ds.map(add_image_column)
            if eval_ds is not None:
                eval_ds = eval_ds.map(add_image_column)

        return train_ds, eval_ds

    def _apply_peft_if_needed(self, model: Any, cfg: TrainingConfig) -> Any:
        if cfg.method != "Full":
            lora_config = self.LoraConfig(
                lora_alpha=cfg.hyperparameters.lora_alpha,
                lora_dropout=cfg.hyperparameters.lora_dropout,
                r=cfg.hyperparameters.lora_rank,
                bias="none",
                target_modules="all-linear",
                task_type="CAUSAL_LM",
                # modules_to_save=["lm_head", "embed_tokens"],
            )

            # NOTE: This can be easily extended to support PEFT other than LoRA

            return self.get_peft_model(model, lora_config)

        logging.warning("No PEFT applied, using full model")
        return model

    def _build_training_args(
        self, trainer_type: str, cfg: TrainingConfig, job_id: str, report_to: str
    ) -> Any:
        # Provider-specific arguments for HuggingFace
        # Determine fp16/bf16
        if self.torch_dtype == torch.bfloat16:
            bf16 = True
            fp16 = False
        else:
            bf16 = False
            fp16 = True

        provider_specific_args = {
            "fp16": fp16,  # shared with fp16_full_eval
            "bf16": bf16,  # shared with bf16_full_eval
            "optim": "adamw_torch_fused",
        }

        # Config classes for HuggingFace
        config_classes = {
            "sft": self.SFTConfig,
            "grpo": self.GRPOConfig,
            "dpo": self.DPOConfig,
        }

        # Get complete configured training arguments
        return _build_shared_training_args(
            trainer_type, cfg, job_id, report_to, provider_specific_args, config_classes
        )

    def _create_trainer(
        self,
        model: Any,
        tokenizer_or_processor: Any,  # either AutoTokenizer or AutoProcessor
        train_ds: Any,
        eval_ds: Any,
        args: Any,
        trainer_type: str,
        cfg: TrainingConfig,
    ) -> Any:
        # Common trainer arguments
        base_trainer_args = {
            "model": model,
            "args": args,
            "train_dataset": train_ds,
            "eval_dataset": eval_ds,
            "processing_class": tokenizer_or_processor,
        }

        if trainer_type == "sft":
            # For SFT Trainer, the new format is compatible with the standard vision collator
            return self.SFTTrainer(
                **base_trainer_args,
                data_collator=self.DataCollatorForVisionLanguageModelling(
                    processor=tokenizer_or_processor  # this should be AutoProcessor
                )
                if cfg.modality == "vision"  # else it's DataCollatorForLanguageModeling
                else None,
                compute_metrics=create_compute_metrics(
                    cfg.eval_config.compute_eval_metrics,
                    cfg.eval_config.batch_eval_metrics,
                )
                if cfg.eval_config
                else None,
            )
        elif trainer_type == "grpo":
            # GRPO requires reward functions
            reward_funcs = load_reward_functions_from_config(cfg.reward_config)
            # NOTE: GRPO Trainer doesn't allow for data collator it process it automatically internally in _generate_and_score_completion
            return self.GRPOTrainer(
                **base_trainer_args,
                reward_funcs=reward_funcs,
            )
        elif trainer_type == "dpo":
            # NOTE: DPO Trainer doesn't need collator for vision it's handled internally
            # NOTE: This does not work yet due to error with DPOTrainer in TRL!!!
            return self.DPOTrainer(**base_trainer_args)
        else:
            raise ValueError(f"Unsupported trainer type: {trainer_type}")


class UnslothTrainingService(BaseTrainingService):
    def __init__(self) -> None:
        # importing unsloth is unused but necessary for optimization
        import unsloth  # noqa: F401
        from unsloth import FastModel, FastVisionModel, is_bfloat16_supported
        from unsloth.trainer import UnslothVisionDataCollator
        from unsloth.chat_templates import (
            get_chat_template,
            standardize_data_formats,
            train_on_responses_only,
        )
        from unsloth import PatchDPOTrainer
        from trl import (
            SFTConfig,
            SFTTrainer,
            GRPOConfig,
            GRPOTrainer,
            DPOConfig,
            DPOTrainer,
        )

        self.FastModel = FastModel
        self.FastVisionModel = FastVisionModel
        self.is_bfloat16_supported = is_bfloat16_supported
        self.get_chat_template = get_chat_template
        self.standardize_data_formats = standardize_data_formats
        self.train_on_responses_only = train_on_responses_only
        self.UnslothVisionDataCollator = UnslothVisionDataCollator
        self.PatchDPOTrainer = PatchDPOTrainer
        self.SFTConfig = SFTConfig
        self.SFTTrainer = SFTTrainer
        self.GRPOConfig = GRPOConfig
        self.GRPOTrainer = GRPOTrainer
        self.DPOConfig = DPOConfig
        self.DPOTrainer = DPOTrainer

        # Haven't tested 270M will add that later
        self.supported_models = [
            "unsloth/gemma-3-1b-pt",
            "unsloth/gemma-3-1b-it",
            "unsloth/gemma-3-4b-pt",
            "unsloth/gemma-3-4b-it",
            "unsloth/gemma-3-12b-pt",
            "unsloth/gemma-3-12b-it",
            "unsloth/gemma-3n-E4B",
            "unsloth/gemma-3n-E4B-it",
            "unsloth/gemma-3n-E2B",
            "unsloth/gemma-3n-E2B-it",
            # Can't find 240m pt with unsloth
            "unsloth/gemma-3-270m-it",
            "unsloth/gemma-3-1b-pt-unsloth-bnb-4bit",
            "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
            "unsloth/gemma-3-4b-pt-unsloth-bnb-4bit",
            "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
            "unsloth/gemma-3-12b-pt-unsloth-bnb-4bit",
            "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
            # "unsloth/gemma-3-27b-pt-unsloth-bnb-4bit",
            # "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",
            "unsloth/gemma-3n-E4B-unsloth-bnb-4bit",
            "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit",
            "unsloth/gemma-3n-E2B-unsloth-bnb-4bit",
            "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit",
            # "unsloth/gemma-3-240m-pt-unsloth-bnb-4bit",  # No PT version
            "unsloth/gemma-3-270m-it-unsloth-bnb-4bit",
        ]

    # Hooks for Template Method:
    def _download_dataset(self, dataset_id: str) -> Tuple[Any, Any]:
        return storage_service.download_processed_dataset(dataset_id)

    def _setup_model(self, cfg: TrainingConfig) -> Tuple[Any, Any]:
        base_model_id = cfg.base_model_id or "unsloth/gemma-3-1b-it-unsloth-bnb-4bit"
        if base_model_id not in self.supported_models:
            raise ValueError(
                f"Unsupported base model {base_model_id}. "
                f"Supported models: {self.supported_models}"
            )

        # Choose Unsloth model based on modality
        if cfg.modality == "vision":
            if base_model_id == "unsloth/gemma-3-1b-it-unsloth-bnb-4bit":
                raise ValueError(
                    "Gemma 3.1B does not support vision fine-tuning. Use Gemma 3.4B or larger."
                )

            model, processor = self.FastVisionModel.from_pretrained(
                # Load in 4-bit for consistency with HuggingFace
                # NOTE: if you use unsloth you default to QLoRA lol
                base_model_id,
                load_in_4bit=True,
                max_seq_length=2048,  # From docs
                full_finetuning=True if cfg.method == "Full" else False,
            )
            # Setup chat template for vision models
            processor = self.get_chat_template(processor, "gemma-3")
            return model, processor
        else:
            model, tokenizer = self.FastModel.from_pretrained(
                base_model_id,
                load_in_4bit=True,
                max_seq_length=cfg.hyperparameters.max_seq_length,
                full_finetuning=True if cfg.method == "Full" else False,
            )
            # Setup chat template for text models
            tokenizer = self.get_chat_template(tokenizer, "gemma-3")
            return model, tokenizer

    def _prepare_dataset_if_needed(
        self,
        train_ds: Dataset,
        eval_ds: Dataset,
        tokenizer_or_processor: Any,
        modality: str,
        trainer_type: str,
    ) -> Tuple[Dataset, Dataset]:
        """
        Optionally format dataset depending on modality and trainer type.
        The dataset is already in the new format, so we only need to handle special cases.
        """
        # Unsloth SFTTrainer needs a special text format, but not for vision.
        if modality == "text" and trainer_type == "sft":
            train_ds = self._prepare_unsloth_text_dataset(
                train_ds, tokenizer_or_processor
            )
            eval_ds = (
                self._prepare_unsloth_text_dataset(eval_ds, tokenizer_or_processor)
                if eval_ds is not None
                else None
            )

        elif modality == "vision" and trainer_type == "grpo":
            # GRPOTrainer expects a single "image" column, not "images"
            def add_image_column(example):
                example["image"] = example["images"][0] if example["images"] else None
                return example

            train_ds = train_ds.map(add_image_column)
            if eval_ds is not None:
                eval_ds = eval_ds.map(add_image_column)

        return train_ds, eval_ds

    def _apply_peft_if_needed(self, model: Any, cfg: TrainingConfig) -> Any:
        # Method is either full or PEFT (LoRA or QLoRA)
        if cfg.method == "Full":
            logging.warning("No PEFT applied, using full model")
            return model

        if cfg.modality == "vision":
            model = self.FastVisionModel.get_peft_model(
                model,
                finetune_vision_layers=True,
                finetune_language_layers=True,
                finetune_attention_modules=True,
                finetune_mlp_modules=True,
                r=cfg.hyperparameters.lora_rank,
                lora_alpha=cfg.hyperparameters.lora_alpha,
                lora_dropout=cfg.hyperparameters.lora_dropout,
                bias="none",
                random_state=3407,
                use_rslora=False,
                loftq_config=None,
                target_modules="all-linear",
                modules_to_save=["lm_head", "embed_tokens"],
            )
        else:
            model = self.FastModel.get_peft_model(
                model,
                # target_modules=[
                #     "q_proj",
                #     "k_proj",
                #     "v_proj",
                #     "o_proj",
                #     "gate_proj",
                #     "up_proj",
                #     "down_proj",
                # ],
                r=cfg.hyperparameters.lora_rank,
                lora_alpha=cfg.hyperparameters.lora_alpha,
                lora_dropout=cfg.hyperparameters.lora_dropout,
                bias="none",
                random_state=3407,
                use_rslora=False,
                loftq_config=None,
                target_modules="all-linear",
            )

        return model

    def _build_training_args(
        self, trainer_type: str, cfg: TrainingConfig, job_id: str, report_to: str
    ) -> Any:
        # Provider-specific arguments for Unsloth
        provider_specific_args = {
            "fp16": not self.is_bfloat16_supported(),
            "bf16": self.is_bfloat16_supported(),
            "optim": "adamw_8bit",  # Unsloth default
        }

        # Config classes for Unsloth
        config_classes = {
            "sft": self.SFTConfig,
            "grpo": self.GRPOConfig,
            "dpo": self.DPOConfig,
        }

        # Get complete configured training arguments
        return _build_shared_training_args(
            trainer_type, cfg, job_id, report_to, provider_specific_args, config_classes
        )

    def _create_trainer(
        self,
        model: Any,
        tokenizer_or_processor: Any,
        train_ds: Any,
        eval_ds: Any,
        args: Any,
        trainer_type: str,
        cfg: TrainingConfig,
    ) -> Any:
        # Common trainer arguments
        base_trainer_args = {
            "model": model,
            "args": args,
            "train_dataset": train_ds,
            "eval_dataset": eval_ds,
            "processing_class": tokenizer_or_processor,
        }

        if trainer_type == "sft":
            # Same as hugging face except we use their built-in collator
            trainer = self.SFTTrainer(
                **base_trainer_args,
                data_collator=(
                    # if modality is vision this is processor
                    # NOTE: this collator works with both formats -- old {"type": "image", "image": ...} and new "images"
                    self.UnslothVisionDataCollator(model, tokenizer_or_processor)
                    if cfg.modality == "vision"
                    else None
                ),
                compute_metrics=create_compute_metrics(
                    cfg.eval_config.compute_eval_metrics,
                    cfg.eval_config.batch_eval_metrics,
                )
                if cfg.eval_config
                else None,
            )

            # Apply response-only for text
            if cfg.modality == "text":
                trainer = self.train_on_responses_only(
                    trainer,
                    instruction_part="<start_of_turn>user\n",
                    response_part="<start_of_turn>model\n",
                )

            return trainer

        elif trainer_type == "grpo":
            # GRPO requires reward functions and processing_class
            reward_funcs = load_reward_functions_from_config(cfg.reward_config)
            return self.GRPOTrainer(
                **base_trainer_args,
                reward_funcs=reward_funcs,
            )
        elif trainer_type == "dpo":
            self.PatchDPOTrainer()
            return self.DPOTrainer(**base_trainer_args)
        else:
            raise ValueError(f"Unsupported trainer type: {trainer_type}")

    def _prepare_unsloth_text_dataset(self, dataset: Any, tokenizer: Any) -> Any:
        """
        Prepare dataset for Unsloth text training format.
        Standardizes and formats the dataset with text field for Unsloth SFTTrainer.

        NOTE: Adds the "text" field only to text datasets because it is specified,
        otherwise we need to pass in a formatting func
        This is not required for vision because we have data collator for vision.

        Somehow unsloth requires this otherwise it breaks for random reasons,
        the original SFTTrainer doesn't care if there's a formatting func etc but the UnslothSFTTrainer does...
        """
        # Standardize format first
        dataset = self.standardize_data_formats(dataset)

        def formatting_prompts_func(examples):
            # Handle different dataset types
            if "messages" in examples:
                # Language modeling format
                convos = examples["messages"]
                texts = [
                    tokenizer.apply_chat_template(
                        convo, tokenize=False, add_generation_prompt=False
                    ).removeprefix("<bos>")
                    for convo in convos
                ]
                return {"text": texts}
            else:
                raise ValueError(
                    "Dataset does not contain messages field, do not use it with SFTTrainer"
                )

        dataset = dataset.map(formatting_prompts_func, batched=True)
        return dataset


class TrainingService:
    _providers = {
        "huggingface": HuggingFaceTrainingService,
        "unsloth": UnslothTrainingService,
    }

    @classmethod
    def from_provider(cls, provider: str) -> BaseTrainingService:
        provider = provider.lower()
        if provider not in cls._providers:
            raise ValueError(f"Unsupported provider {provider}")
        return cls._providers[provider]()

    @classmethod
    def list_providers(cls):
        return list(cls._providers.keys())
