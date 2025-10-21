"""
Data collator for BERT pretraining with MLM objective.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
from transformers import DataCollatorForLanguageModeling


@dataclass
class DataCollatorForBERTPretraining:
    """
    Data collator for BERT pretraining with both MLM and NSP objectives.

    This wraps HuggingFace's DataCollatorForLanguageModeling to handle MLM
    and also passes through the NSP labels.
    """

    tokenizer: Any
    mlm_probability: float = 0.15

    def __post_init__(self):
        self.mlm_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=self.mlm_probability,
        )

    def __call__(
        self, examples: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Collate examples and apply MLM masking.

        Args:
            examples: List of examples from the dataset

        Returns:
            Batch dictionary with masked inputs and labels
        """
        # Extract NSP labels
        nsp_labels = torch.stack([ex['next_sentence_label'] for ex in examples])

        # Prepare examples for MLM collator (remove NSP label)
        mlm_examples = [
            {
                'input_ids': ex['input_ids'],
                'attention_mask': ex['attention_mask'],
                'token_type_ids': ex['token_type_ids'],
            }
            for ex in examples
        ]

        # Apply MLM masking
        batch = self.mlm_collator(mlm_examples)

        # Add NSP labels back
        batch['next_sentence_label'] = nsp_labels

        return batch

