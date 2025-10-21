"""
Dataset class for BERT pretraining with MLM and NSP objectives.
Handles loading and preprocessing of JSONL data.
"""

import json
import random
from typing import Dict, List, Tuple
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, random_split
from transformers import PreTrainedTokenizer


@dataclass
class NSPExample:
    """Data class for a single NSP training example."""
    tokens: List[str]
    segment_ids: List[int]
    is_random_next: bool


class BERTPretrainingDataset(Dataset):
    """
    Dataset for BERT pretraining with MLM and NSP objectives.

    Args:
        data_path: Path to JSONL file
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        nsp_probability: Probability of creating a random next sentence
        short_seq_probability: Probability of creating sequences shorter than max_length
        max_sentences_per_doc: Maximum sentences to consider per document
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        nsp_probability: float = 0.5,
        short_seq_probability: float = 0.1,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.nsp_probability = nsp_probability
        self.short_seq_probability = short_seq_probability

        # Load documents from JSONL
        self.documents = self._load_documents(data_path)

        # Pre-generate examples for efficiency
        self.examples = self._create_examples()

    def _load_documents(self, data_path: str) -> List[List[str]]:
        """Load documents from JSONL file."""
        documents = []

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if 'sentences' in entry:
                        # Parse sentences - they are individual strings separated by comma
                        if isinstance(entry['sentences'], list):
                            sentences = entry['sentences']
                        else:
                            sentences = [s.strip() for s in entry['sentences'].split(',') if s.strip()]

                        if len(sentences) > 0:
                            documents.append(sentences)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line")
                    continue

        print(f"Loaded {len(documents)} documents")
        return documents

    def _create_examples(self) -> List[Dict]:
        """Pre-create training examples with NSP pairs."""
        examples = []

        for doc_idx, document in enumerate(self.documents):
            # Create multiple examples from each document
            examples.extend(self._create_examples_from_document(doc_idx, document))

        print(f"Created {len(examples)} training examples")
        return examples

    def _create_examples_from_document(
        self, doc_idx: int, document: List[str]
    ) -> List[Dict]:
        """Create training examples from a single document."""
        examples = []

        # Target sequence length (accounting for special tokens)
        target_seq_length = self.max_length - 3  # [CLS], [SEP], [SEP]

        # Randomly decide if we want a short sequence
        if random.random() < self.short_seq_probability:
            target_seq_length = random.randint(2, target_seq_length)

        i = 0
        while i < len(document):
            # Get segment A
            segment_a, i = self._get_segment(document, i, target_seq_length)

            if len(segment_a) == 0:
                break

            # Get segment B (either next sentences or random)
            is_random_next = random.random() < self.nsp_probability

            if is_random_next:
                # Random document
                random_doc_idx = random.randint(0, len(self.documents) - 1)
                random_document = self.documents[random_doc_idx]
                segment_b, _ = self._get_segment(
                    random_document,
                    random.randint(0, len(random_document) - 1),
                    target_seq_length - len(segment_a)
                )
            else:
                # Actual next sentences
                segment_b, i = self._get_segment(
                    document, i, target_seq_length - len(segment_a)
                )

            # Create example
            if len(segment_b) > 0:
                examples.append({
                    'segment_a': ' '.join(segment_a),
                    'segment_b': ' '.join(segment_b),
                    'is_random_next': is_random_next
                })

        return examples

    def _get_segment(
        self, document: List[str], start_idx: int, max_length: int
    ) -> Tuple[List[str], int]:
        """Get a segment of sentences from document."""
        segment = []
        current_length = 0

        for i in range(start_idx, len(document)):
            sentence = document[i]
            # Approximate token count (actual tokenization happens later)
            sentence_length = len(sentence.split())

            if current_length + sentence_length > max_length and len(segment) > 0:
                break

            segment.append(sentence)
            current_length += sentence_length

        return segment, start_idx + len(segment)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example."""
        example = self.examples[idx]

        # Tokenize the segments
        encoding = self.tokenizer(
            example['segment_a'],
            example['segment_b'],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        # Prepare labels for NSP
        nsp_label = 1 if example['is_random_next'] else 0

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'token_type_ids': encoding['token_type_ids'].squeeze(0),
            'next_sentence_label': torch.tensor(nsp_label, dtype=torch.long),
        }


def create_train_val_datasets(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    dataset_config: Dict,
    model_config: Dict,
    seed: int = 42,
):
    """
    Create train and validation datasets.

    Args:
        data_path: Path to JSONL data file
        tokenizer: HuggingFace tokenizer
        dataset_config: Dataset configuration dictionary
        model_config: Model configuration dictionary
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    # Create full dataset
    full_dataset = BERTPretrainingDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=model_config['max_length'],
        nsp_probability=dataset_config['nsp_probability'],
        short_seq_probability=dataset_config['short_seq_probability'],
    )

    # Split into train and validation
    train_size = int(len(full_dataset) * dataset_config['train_split'])
    eval_size = len(full_dataset) - train_size

    train_dataset, eval_dataset = random_split(
        full_dataset,
        [train_size, eval_size],
        generator=torch.Generator().manual_seed(seed)
    )

    print(f"Train examples: {len(train_dataset)}")
    print(f"Eval examples: {len(eval_dataset)}")

    return train_dataset, eval_dataset
