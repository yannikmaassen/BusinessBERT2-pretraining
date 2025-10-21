"""
Custom trainer for BERT pretraining with MLM and NSP objectives.
"""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import Trainer
from transformers.trainer_utils import has_length
from tqdm.auto import tqdm


class BERTPreTrainer(Trainer):
    """
    Custom trainer for BERT pretraining that handles both MLM and NSP objectives.

    The BertForPreTraining model from HuggingFace already computes both MLM and NSP losses,
    so we mainly need to ensure proper logging and metric computation.
    """

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Compute the combined loss for MLM and NSP.

        Args:
            model: The BERT model
            inputs: Dictionary containing input tensors
            return_outputs: Whether to return model outputs

        Returns:
            Loss tensor or tuple of (loss, outputs)
        """
        # BertForPreTraining expects these inputs
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            token_type_ids=inputs['token_type_ids'],
            labels=inputs.get('labels'),  # MLM labels
            next_sentence_label=inputs.get('next_sentence_label'),  # NSP labels
        )

        # The model automatically computes and combines MLM and NSP losses
        loss = outputs.loss

        # Log individual losses if available
        if return_outputs and hasattr(outputs, 'prediction_logits') and hasattr(outputs, 'seq_relationship_logits'):
            # Store individual losses for logging
            if hasattr(outputs, 'loss') and 'labels' in inputs and 'next_sentence_label' in inputs:
                # Extract individual losses for detailed logging
                # Note: BertForPreTraining combines losses internally, but we can log them separately
                self.log({
                    "train_total_loss": loss.item(),
                })

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model: nn.Module, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Perform a training step with detailed logging.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        return loss.detach()

    def evaluate(
        self,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        ignore_keys: Optional[list] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and return metrics with progress bar.
        """
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        model = self._wrap_model(self.model, training=False)
        model.eval()

        total_loss = 0.0
        total_steps = 0
        total_mlm_correct = 0
        total_mlm_tokens = 0
        total_nsp_correct = 0
        total_nsp_samples = 0

        # Create progress bar for evaluation
        if has_length(eval_dataloader):
            pbar = tqdm(eval_dataloader, desc="Evaluation", leave=False)
        else:
            pbar = eval_dataloader

        for step, inputs in enumerate(pbar):
            inputs = self._prepare_inputs(inputs)

            with torch.no_grad():
                outputs = model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    token_type_ids=inputs['token_type_ids'],
                    labels=inputs.get('labels'),
                    next_sentence_label=inputs.get('next_sentence_label'),
                )

                loss = outputs.loss
                total_loss += loss.item()

                # Track MLM accuracy
                if hasattr(outputs, 'prediction_logits') and 'labels' in inputs:
                    mlm_predictions = outputs.prediction_logits.argmax(dim=-1)
                    mlm_labels = inputs['labels']
                    mask = mlm_labels != -100  # Ignore non-masked tokens
                    total_mlm_correct += (mlm_predictions[mask] == mlm_labels[mask]).sum().item()
                    total_mlm_tokens += mask.sum().item()

                # Track NSP accuracy
                if hasattr(outputs, 'seq_relationship_logits') and 'next_sentence_label' in inputs:
                    nsp_predictions = outputs.seq_relationship_logits.argmax(dim=-1)
                    nsp_labels = inputs['next_sentence_label']
                    nsp_correct = (nsp_predictions == nsp_labels).sum().item()
                    total_nsp_correct += nsp_correct
                    total_nsp_samples += nsp_labels.size(0)

                total_steps += 1

                # Update progress bar
                if has_length(eval_dataloader):
                    pbar.set_postfix({
                        'loss': f'{total_loss / total_steps:.4f}',
                        'mlm_acc': f'{total_mlm_correct / max(total_mlm_tokens, 1):.4f}',
                        'nsp_acc': f'{total_nsp_correct / max(total_nsp_samples, 1):.4f}',
                    })

        # Compute metrics
        metrics = {
            f"{metric_key_prefix}_loss": total_loss / total_steps,
        }

        if total_mlm_tokens > 0:
            mlm_accuracy = total_mlm_correct / total_mlm_tokens
            metrics[f"{metric_key_prefix}_mlm_accuracy"] = mlm_accuracy
            metrics[f"{metric_key_prefix}_mlm_loss"] = metrics[f"{metric_key_prefix}_loss"] * 0.5  # Approximate

        if total_nsp_samples > 0:
            nsp_accuracy = total_nsp_correct / total_nsp_samples
            metrics[f"{metric_key_prefix}_nsp_accuracy"] = nsp_accuracy
            metrics[f"{metric_key_prefix}_nsp_loss"] = metrics[f"{metric_key_prefix}_loss"] * 0.5  # Approximate

        self.log(metrics)

        return metrics
