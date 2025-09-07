from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from schema import WandbConfig, TrainingConfig
from utils import run_evaluation, save_and_track
from job_manager import JobTracker


class BaseTrainingService(ABC):
    """
    Template Method for training services.
    Defines the standard workflow; subclasses implement hooks below.
    """

    def run_training(
        self, processed_dataset_id: str, config: TrainingConfig, tracker: JobTracker
    ) -> Dict[str, Any]:
        # 1. Preparation
        tracker.preparing()

        # 2. Download datasets
        train_ds, eval_ds = self._download_dataset(processed_dataset_id)

        # Skip eval entirely if strategy is no
        if not config.eval_config or config.eval_config.eval_strategy == "no":
            eval_ds = None

        # 3. Model + tokenizer setup
        model, tokenizer = self._setup_model(config)

        # 4. Apply PEFT if needed on the model
        model = self._apply_peft_if_needed(model, config)

        # 5. Dataset preparation
        train_dataset, eval_dataset = self._prepare_dataset_if_needed(
            train_ds, eval_ds, tokenizer, config.modality, config.trainer_type
        )

        # 6. WandB initialization
        report_to, wandb_url = self._setup_wandb(config.wandb_config, tracker.job_id)

        # 7. Build training arguments
        training_args = self._build_training_args(
            config.trainer_type, config, tracker.job_id, report_to
        )

        # 8. Instantiate trainer
        trainer = self._create_trainer(
            model,
            tokenizer,
            train_dataset,
            eval_dataset,
            training_args,
            config.trainer_type,
            config,
        )

        # 9. Train
        tracker.training(wandb_url)
        trainer.train()

        # 10. Evaluate
        metrics = self._evaluate_if_needed(trainer, eval_dataset)

        # 11. Save + record in tracker
        artifact = self._save_and_track(model, tokenizer, tracker, metrics, config)

        result = {
            "adapter_path": artifact.remote_path,
            "base_model_id": artifact.base_model_id,
        }

        return result

    # --- Hooks to implement in subclasses ---
    @abstractmethod
    def _download_dataset(self, dataset_id: str) -> Tuple[Any, Any]: ...

    @abstractmethod
    def _setup_model(self, cfg: TrainingConfig) -> Tuple[Any, Any]: ...

    @abstractmethod
    def _prepare_dataset_if_needed(
        self,
        train_ds: Any,
        eval_ds: Any,
        tokenizer: Any,
        modality: str,
        trainer_type: str,
    ) -> Tuple[Any, Any]: ...

    @abstractmethod
    def _apply_peft_if_needed(self, model: Any, cfg: TrainingConfig) -> Any: ...

    @abstractmethod
    def _build_training_args(
        self, trainer: str, cfg: TrainingConfig, job_id: str, report_to: str
    ) -> Any: ...

    @abstractmethod
    def _create_trainer(
        self,
        model: Any,
        tokenizer_or_processor: Any,
        train_ds: Any,
        eval_ds: Any,
        args: Any,
        trainer: str,
        cfg: TrainingConfig,
    ) -> Any: ...

    # --- Default behaviors ---
    def _evaluate_if_needed(
        self, trainer: Any, eval_ds: Any
    ) -> Optional[Dict[str, Any]]:
        return run_evaluation(trainer) if eval_ds is not None else None

    def _save_and_track(
        self,
        model: Any,
        tokenizer: Any,
        tracker: JobTracker,
        metrics: Optional[Dict[str, Any]],
        train_config: TrainingConfig,
    ) -> Any:
        return save_and_track(
            train_config.export_config,
            model,
            tokenizer,
            tracker.job_id,
            train_config.base_model_id,
            train_config.provider,
            tracker,
            metrics,
        )

    def _setup_wandb(
        self, config: Optional[WandbConfig], job_id: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Setup WandB and return (report_to, wandb_url).
        """
        if not config or not config.api_key:
            return "none", ""

        # Delay import to avoid dependency at module load
        import wandb

        # Login
        wandb.login(key=config.api_key)

        wandb.init(
            project=config.project or "gemma-fine-tuning",
            name=job_id,
            tags=["fine-tuning"],
        )

        wandb_url = wandb.run.url if wandb.run else ""
        return "wandb", wandb_url
