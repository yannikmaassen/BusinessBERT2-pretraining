"""
Main training script for BERT pretraining with MLM and NSP objectives.
"""

import argparse
import os
import yaml

from transformers import (
    BertForPreTraining,
    BertTokenizer,
    TrainingArguments,
    set_seed,
)

from src.data.dataset import create_train_val_datasets
from src.data.collator import DataCollatorForBERTPretraining
from src.training.trainer import BERTPreTrainer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_model_and_tokenizer(model_name: str):
    """Initialize BERT model and tokenizer."""
    print(f"Loading model and tokenizer: {model_name}")

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForPreTraining.from_pretrained(model_name)

    print(f"Model loaded with {model.num_parameters():,} parameters")
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Pretrain BERT with MLM and NSP objectives"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to configuration YAML file")
    parser.add_argument("--data", type=str, required=True, help="Path to training data (JSONL format)")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume training from")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config['training']['seed'])
    os.makedirs(config['training']['output_dir'], exist_ok=True)
    os.makedirs(config['logging']['logging_dir'], exist_ok=True)

    model, tokenizer = setup_model_and_tokenizer(config['model']['name'])

    print(f"Loading dataset from: {args.data}")
    train_dataset, eval_dataset = create_train_val_datasets(
        data_path=args.data,
        tokenizer=tokenizer,
        dataset_config=config['dataset'],
        model_config=config['model'],
        seed=config['training']['seed'],
    )

    data_collator = DataCollatorForBERTPretraining(
        tokenizer=tokenizer,
        mlm_probability=config['model']['mlm_probability'],
    )

    training_args = TrainingArguments(
        output_dir=config['training']['output_dir'],
        overwrite_output_dir=True,
        num_train_epochs=config['training']['num_train_epochs'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        warmup_steps=config['training']['warmup_steps'],
        logging_dir=config['logging']['logging_dir'],
        logging_steps=config['training']['logging_steps'],
        save_steps=config['training']['save_steps'],
        save_total_limit=config['training']['save_total_limit'],
        eval_strategy="steps",
        eval_steps=config['training']['eval_steps'],
        fp16=config['training']['fp16'],
        dataloader_num_workers=config['training']['dataloader_num_workers'],
        report_to=config['logging']['report_to'],
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        seed=config['training']['seed'],
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )

    trainer = BERTPreTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    print(f"Saving final model to {config['training']['output_dir']}")
    trainer.save_model(config['training']['output_dir'])
    tokenizer.save_pretrained(config['training']['output_dir'])

    print("Training complete!")


if __name__ == "__main__":
    main()
