"""
Custom callbacks for enhanced Weights & Biases logging.
"""

from typing import Dict
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl


class WandbMetricsCallback(TrainerCallback):
    """
    Callback to log detailed metrics to Weights & Biases.
    Logs MLM and NSP metrics separately for better tracking.
    """

    def __init__(self):
        self.training_loss_history = []

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: Dict = None, **kwargs):
        """
        Log metrics to W&B with enhanced formatting.
        """
        if logs is None:
            return

        try:
            import wandb

            # Only log if wandb is initialized
            if wandb.run is None:
                return

            # Format logs for better organization in W&B
            formatted_logs = {}

            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    # Organize metrics by category
                    if 'mlm' in key.lower():
                        new_key = f"metrics/MLM/{key}"
                    elif 'nsp' in key.lower():
                        new_key = f"metrics/NSP/{key}"
                    elif 'loss' in key.lower() and 'mlm' not in key.lower() and 'nsp' not in key.lower():
                        new_key = f"metrics/total/{key}"
                    elif 'learning_rate' in key:
                        new_key = f"training/{key}"
                    elif 'epoch' in key:
                        new_key = f"training/{key}"
                    else:
                        new_key = f"metrics/{key}"

                    formatted_logs[new_key] = value

            # Log to wandb
            if formatted_logs:
                wandb.log(formatted_logs, step=state.global_step)

        except ImportError:
            pass  # W&B not installed
        except Exception as e:
            # Don't fail training due to logging errors
            pass

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics: Dict = None, **kwargs):
        """
        Log evaluation metrics with special formatting.
        """
        if metrics is None:
            return

        try:
            import wandb

            if wandb.run is None:
                return

            # Create a summary table for evaluation
            eval_summary = {}

            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    # Remove 'eval_' prefix for cleaner display
                    clean_key = key.replace('eval_', '')

                    if 'mlm' in clean_key:
                        eval_summary[f"evaluation/MLM/{clean_key}"] = value
                    elif 'nsp' in clean_key:
                        eval_summary[f"evaluation/NSP/{clean_key}"] = value
                    else:
                        eval_summary[f"evaluation/{clean_key}"] = value

            if eval_summary:
                wandb.log(eval_summary, step=state.global_step)

        except ImportError:
            pass
        except Exception as e:
            pass

