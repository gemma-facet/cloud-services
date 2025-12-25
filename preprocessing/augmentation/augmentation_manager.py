import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from .text_augmentor import (
    EDAugmentor,
    BackTranslationAugmentor,
    ParaphraseAugmentor,
    EDASettings,
    BackTranslationSettings,
    ParaphrasingSettings,
    SynthesisSettings,
)
from .gemini_synthesizer import GeminiSynthesizer
from .text_augmentor import TextAugmentationPipeline

logger = logging.getLogger(__name__)


class AugmentationMethod(Enum):
    """Available augmentation methods"""

    EDA = "eda"
    BACK_TRANSLATION = "back_translation"
    PARAPHRASING = "paraphrasing"
    SYNTHESIS = "synthesis"


@dataclass
class AugmentationConfig:
    """
    Configuration for augmentation pipeline (used internally in the backend)
    This is different from the user-facing API request configuration (see AugmentationSetupConfig).
    """

    # Control which methods are enabled -- this is built in the pipeline based on user preferences
    enabled_methods: List[AugmentationMethod]

    # General settings
    augmentation_factor: float = 1.5

    # Method-specific settings
    eda_settings: Optional[EDASettings] = None
    back_translation_settings: Optional[BackTranslationSettings] = None
    paraphrasing_settings: Optional[ParaphrasingSettings] = None
    synthesis_settings: Optional[SynthesisSettings] = None

    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.augmentation_factor <= 1.0:
            raise ValueError("Augmentation factor must be greater than 1.0")

        if not self.enabled_methods:
            raise ValueError("At least one augmentation method must be enabled")

        # Ensure default settings exist
        if self.eda_settings is None:
            self.eda_settings = {}
        if self.back_translation_settings is None:
            self.back_translation_settings = {}
        if self.paraphrasing_settings is None:
            self.paraphrasing_settings = {}
        if self.synthesis_settings is None:
            self.synthesis_settings = {}


@dataclass
class AugmentationResult:
    """Result of augmentation process"""

    original_count: int
    final_count: int
    methods_used: List[str]
    synthesis_used: bool
    errors: List[str]

    @property
    def success_rate(self) -> float:
        """Calculate success rate based on target vs actual"""
        return min(1.0, self.final_count / max(1, self.original_count))


class AugmentationManager:
    """
    Simplified augmentation manager that provides a clean interface
    for the data preprocessor without complex configuration conflicts.
    """

    def __init__(self):
        self._pipeline = None
        self._config = None

    def configure(self, config: AugmentationConfig) -> None:
        """Configure the augmentation manager"""
        self._config = config
        self._pipeline = None  # Reset pipeline to force recreation

    def is_configured(self) -> bool:
        """Check if manager is properly configured"""
        return self._config is not None

    def _require_config(self):
        if self._config is None:
            raise ValueError("Manager not configured. Call configure() first.")

    def get_available_methods(self) -> List[str]:
        """Get list of actually available methods (not just configured)"""
        self._require_config()
        config: AugmentationConfig = self._config  # type: ignore
        available = []

        # EDA is always available (no dependencies)
        if AugmentationMethod.EDA in config.enabled_methods:
            available.append("eda")

        # Check if transformers is available for other methods
        if AugmentationMethod.BACK_TRANSLATION in config.enabled_methods:
            available.append("back_translation")

        if AugmentationMethod.PARAPHRASING in config.enabled_methods:
            available.append("paraphrasing")

        # Check if synthesis is properly configured
        if AugmentationMethod.SYNTHESIS in config.enabled_methods:
            synthesis_config = config.synthesis_settings or {}
            api_key = synthesis_config.get("gemini_api_key")
            if api_key:
                available.append("synthesis")
            else:
                logger.warning("Synthesis enabled but no API key provided, skipping")

        return available

    def augment_dataset(
        self, dataset: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], AugmentationResult]:
        """
        Augment dataset using configured methods

        Args:
            dataset: List of conversation samples

        Returns:
            Tuple of (augmented_dataset, augmentation_result)
        """
        self._require_config()
        config: AugmentationConfig = self._config  # type: ignore
        if not dataset:
            return dataset, AugmentationResult(
                original_count=0,
                final_count=0,
                methods_used=[],
                synthesis_used=False,
                errors=["Empty dataset provided"],
            )

        original_count = len(dataset)
        errors = []

        # Get actually available methods
        available_methods = self.get_available_methods()

        if not available_methods:
            return dataset, AugmentationResult(
                original_count=original_count,
                final_count=original_count,
                methods_used=[],
                synthesis_used=False,
                errors=["No augmentation methods available"],
            )

        # Create pipeline if needed
        if self._pipeline is None:
            self._pipeline = self._create_pipeline(available_methods)

        # Perform augmentation
        try:
            logger.info(
                f"Augmenting dataset from {original_count} samples with methods: {available_methods}"
            )

            augmented_dataset = self._pipeline.augment_dataset(
                dataset, augmentation_factor=config.augmentation_factor
            )

            synthesis_used = "synthesis" in available_methods

            return augmented_dataset, AugmentationResult(
                original_count=original_count,
                final_count=len(augmented_dataset),
                methods_used=available_methods,
                synthesis_used=synthesis_used,
                errors=errors,
            )

        except Exception as e:
            error_msg = f"Augmentation failed: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)

            return dataset, AugmentationResult(
                original_count=original_count,
                final_count=original_count,
                methods_used=[],
                synthesis_used=False,
                errors=errors,
            )

    def _create_pipeline(self, available_methods: List[str]):
        """Create the actual augmentation pipeline based on available methods"""
        self._require_config()
        config: AugmentationConfig = self._config  # type: ignore
        # Create augmentors
        augmentors = {}
        synthesizer = None

        if "eda" in available_methods:
            eda_settings = config.eda_settings or {}
            augmentors["eda"] = EDAugmentor(
                alpha_sr=eda_settings.get("eda_alpha_sr", 0.1),
                alpha_ri=eda_settings.get("eda_alpha_ri", 0.1),
                alpha_rs=eda_settings.get("eda_alpha_rs", 0.1),
                p_rd=eda_settings.get("eda_p_rd", 0.1),
                num_aug=eda_settings.get("num_aug", 1),
            )

        if "back_translation" in available_methods:
            bt_settings = config.back_translation_settings or {}
            augmentors["back_translation"] = BackTranslationAugmentor(
                intermediate_lang=bt_settings.get("intermediate_lang", "fr")
            )

        if "paraphrasing" in available_methods:
            para_settings = config.paraphrasing_settings or {}
            augmentors["paraphrasing"] = ParaphraseAugmentor(
                model_name=para_settings.get(
                    "paraphrase_model", "humarin/chatgpt_paraphraser_on_T5_base"
                )
            )

        if "synthesis" in available_methods:
            synth_settings = config.synthesis_settings or None
            api_key = synth_settings.get("gemini_api_key") if synth_settings else None
            synthesizer = GeminiSynthesizer(
                api_key=api_key,
                model_name=synth_settings.get(
                    "gemini_model", "gemini-2.0-flash-001"
                )
                if synth_settings
                else "gemini-2.0-flash-001",
            )

        return TextAugmentationPipeline(
            augmentors=augmentors,
            synthesizer=synthesizer,
            synthesis_settings=config.synthesis_settings,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get current configuration and availability stats"""
        self._require_config()
        config: AugmentationConfig = self._config  # type: ignore
        available_methods = self.get_available_methods()

        return {
            "configured": True,
            "enabled_methods": [m.value for m in config.enabled_methods],
            "available_methods": available_methods,
            "augmentation_factor": config.augmentation_factor,
            "synthesis_available": "synthesis" in available_methods,
        }


def run_augment_pipeline(dataset, user_preferences):
    """
    The definitive augmentation pipeline that uses the
    AugmentationManager interface and TextAugmentationPipeline backend.
    This function takes user preferences and dataset,
    builds the configuration, and runs the augmentation.

    NOTE: This function supports customising parameters for each method,
    however it is not recommended so it has been removed from the public API.

    **You should always use this function in the backend service.**
    You should not try to build your own AugmentationConfig or
    AugmentationManager instances.

    Args:
        dataset: List of conversation samples
        user_preferences: Dict with user's augmentation preferences

    Returns:
        Tuple of (augmented_dataset, result_info)

    Example:
    ```python
    from augmentation import run_augment_pipeline
    sample_user_preferences = {
        "use_eda": True,
        "use_back_translation": True,
        "use_paraphrasing": False,
        "use_synthesis": True,
        "gemini_api_key": "your_gemini_api_key",
        "augmentation_factor": 2.0,
        "other_specific_settings": "value", # do not recommend using these
    }
    sample_dataset = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm fine, thank you!"},
    ]
    augmented_dataset, result_info = run_augment_pipeline(
        dataset=sample_dataset, user_preferences=sample_user_preferences
    )
    ```
    """

    # Validate inputs
    if not dataset:
        logger.warning("Empty dataset provided to run_augment_pipeline")
        return dataset, AugmentationResult(
            methods_used=[],
            synthesis_used=False,
            original_count=0,
            final_count=0,
            errors=["Empty dataset provided"],
        )

    augmentation_factor = user_preferences.get("augmentation_factor", 1.5)

    # Build enabled methods based on preferences
    enabled_methods = []

    if user_preferences.get("use_eda", False):
        enabled_methods.append(AugmentationMethod.EDA)

    if user_preferences.get("use_back_translation", False):
        enabled_methods.append(AugmentationMethod.BACK_TRANSLATION)

    if user_preferences.get("use_paraphrasing", False):
        enabled_methods.append(AugmentationMethod.PARAPHRASING)

    if user_preferences.get("use_synthesis", False) and user_preferences.get(
        "gemini_api_key"
    ):
        enabled_methods.append(AugmentationMethod.SYNTHESIS)

    # Default to EDA if no methods selected
    if not enabled_methods:
        enabled_methods = [AugmentationMethod.EDA]
        logger.info("No augmentation methods specified, defaulting to EDA")

    try:
        # Create configuration
        config = AugmentationConfig(
            enabled_methods=enabled_methods,
            augmentation_factor=augmentation_factor,
            # Method-specific settings
            eda_settings={
                "eda_alpha_sr": user_preferences.get("eda_synonym_rate", 0.1),
                "eda_alpha_ri": user_preferences.get("eda_insert_rate", 0.1),
                "eda_alpha_rs": user_preferences.get("eda_swap_rate", 0.1),
                "eda_p_rd": user_preferences.get("eda_delete_rate", 0.1),
            },
            back_translation_settings={
                "intermediate_lang": user_preferences.get("translation_language", "fr"),
            },
            paraphrasing_settings={
                "paraphrase_model": user_preferences.get(
                    "paraphrase_model", "humarin/chatgpt_paraphraser_on_T5_base"
                ),
            },
            synthesis_settings={
                "gemini_api_key": user_preferences.get("gemini_api_key"),
                "gemini_model": user_preferences.get(
                    "gemini_model", "gemini-2.0-flash-lite"
                ),
                "synthesis_ratio": user_preferences.get("synthesis_ratio", 0.5),
                "system_message": user_preferences.get("system_message", ""),
                "max_batch_size": user_preferences.get("max_batch_size", 10),
                "custom_prompt": user_preferences.get("custom_prompt"),
            },
        )

        # Configure and run augmentation
        manager = AugmentationManager()
        manager.configure(config)

        return manager.augment_dataset(dataset)

    except Exception as e:
        logger.error(f"Error in run_augment_pipeline: {e}")
        return dataset, AugmentationResult(
            methods_used=[],
            synthesis_used=False,
            original_count=len(dataset),
            final_count=len(dataset),
            errors=[f"Configuration error: {str(e)}"],
        )
