from transformers.integrations import WandbCallback
import pandas as pd
import torch.distributed as dist

from ._default_samples import DEFAULT_SAMPLES


class WandbPredictionProgressCallback(WandbCallback):
    """Custom WandbCallback to log model predictions during training.

    This callback logs sample sparse representation to a wandb.Table at each
    logging step during training.

    Attributes:
        trainer (Trainer): The Hugging Face Trainer instance.
        samples (list[dict]): Override the default sample inputs by passing a list of
            ``{"id": int, "language": str, "input": str}`` dicts.
    """

    def __init__(self, trainer, samples=None):
        """Initializes the WandbPredictionProgressCallback instance.

        Args:
            trainer (Trainer): The Hugging Face Trainer instance.
            samples: Optional override for the default multilingual prompt set.
        """
        super().__init__()
        self.trainer = trainer
        self.samples = samples if samples is not None else DEFAULT_SAMPLES

    def _log_predictions(self, state):
        output = self.trainer.predict(self.samples)
        if not dist.is_initialized() or dist.get_rank() == 0:
            output = pd.DataFrame(output)
            output["step"] = state.global_step
            records_table = self._wandb.Table(dataframe=output)
            self._wandb.log({"sample_output": records_table})

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        self._log_predictions(state)

    def on_epoch_end(self, args, state, control, **kwargs):
        super().on_epoch_end(args, state, control, **kwargs)
        self._log_predictions(state)
