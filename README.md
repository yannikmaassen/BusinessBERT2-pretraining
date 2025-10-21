# BusinessBERT2-pretraining

A Python project for further pretraining BERT models using custom business text data with MLM (Masked Language Modeling) and NSP (Next Sentence Prediction) objectives.

## 🚀 Features

- **BERT Pretraining**: Continue training from `bert-base-uncased` checkpoint
- **MLM + NSP Objectives**: Standard BERT pretraining objectives
- **Custom Dataset Support**: Load data from JSONL format
- **Optimized for A100**: Configured for Google Colab A100 GPUs
- **Modular Design**: Easy to extend with additional pretraining heads
- **HuggingFace Integration**: Built on transformers library

## 📁 Project Structure

```
BusinessBERT2-pretraining/
├── config.yaml                 # Central configuration file
├── Makefile                    # Commands for training
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── src/
    ├── data/
    │   ├── dataset.py         # Dataset class for JSONL data
    │   └── collator.py        # Data collator for MLM
    └── training/
        ├── pretrain.py        # Main training script
        └── trainer.py         # Custom trainer with MLM+NSP
```

## 📊 Data Format

Your JSONL file should have entries like this:

```json
{"sentences": ["First sentence.", "Second sentence.", "Third sentence."], "sic2": "01", "sic3": "011", "sic4": "0111"}
{"sentences": ["Another document.", "With multiple sentences."], "sic2": "02", "sic3": "022", "sic4": "0222"}
```

Or if sentences are comma-separated strings:

```json
{"sentences": "First sentence., Second sentence., Third sentence.", "sic2": "01", "sic3": "011", "sic4": "0111"}
```

## 🛠️ Installation

### Local Installation

```bash
# Clone the repository
cd /Users/yannik/dev/masterthesis/BusinessBERT2-pretraining

# Create virtual environment (optional but recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
make install
```

### Google Colab Installation

```python
# In a Colab notebook
!git clone <your-repo-url>
%cd BusinessBERT2-pretraining
!pip install -r requirements.txt
```

## 🎯 Usage

### Training on Google Colab (A100)

```bash
# Using the Makefile
make train-data-colab DATA=/content/drive/MyDrive/businessbert_data/sample.jsonl

# Or directly with Python
python -m src.training.pretrain \
    --config config.yaml \
    --data /content/drive/MyDrive/businessbert_data/sample.jsonl
```

### Training Locally

```bash
# Using the Makefile
make train-local DATA=./data/sample.jsonl

# Or directly with Python
python -m src.training.pretrain \
    --config config.yaml \
    --data ./data/sample.jsonl
```

### Resume from Checkpoint

```bash
python -m src.training.pretrain \
    --config config.yaml \
    --data /path/to/data.jsonl \
    --resume_from_checkpoint ./outputs/checkpoint-5000
```

## ⚙️ Configuration

Edit `config.yaml` to customize training parameters:

### Model Configuration
- `model.name`: Base model to start from (default: `bert-base-uncased`)
- `model.max_length`: Maximum sequence length (default: 512)
- `model.mlm_probability`: Probability of masking tokens (default: 0.15)

### Training Configuration
- `training.num_train_epochs`: Number of epochs (default: 3)
- `training.per_device_train_batch_size`: Batch size per device (default: 16)
- `training.gradient_accumulation_steps`: Accumulation steps (default: 2)
- `training.learning_rate`: Learning rate (default: 5e-5)
- `training.fp16`: Use mixed precision training (default: true)

### Dataset Configuration
- `dataset.train_split`: Train/validation split (default: 0.95)
- `dataset.nsp_probability`: Probability of random next sentence (default: 0.5)
- `dataset.short_seq_probability`: Probability of short sequences (default: 0.1)

## 📈 Monitoring Training

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir ./logs

# View at http://localhost:6006
```

### Weights & Biases (Optional)

Edit `config.yaml`:
```yaml
logging:
  report_to: ["tensorboard", "wandb"]
```

Then login to W&B:
```bash
wandb login
```

## 🧪 What's Implemented

### Current Features (MLM + NSP)

✅ **Dataset Loading**: Load custom JSONL data with sentence parsing  
✅ **NSP Pairs Creation**: Automatic creation of next sentence prediction pairs  
✅ **MLM Masking**: 15% token masking for masked language modeling  
✅ **Custom Trainer**: Tracks MLM and NSP accuracy separately  
✅ **Evaluation**: Validation during training with metrics  
✅ **Checkpointing**: Save and resume training from checkpoints  
✅ **Mixed Precision**: FP16 training for faster computation on A100  

### Next Steps (For Future Extension)

🔜 **Custom Classification Heads**: Add SIC2, SIC3, SIC4 prediction heads  
🔜 **Multi-task Learning**: Train MLM+NSP+Classification jointly  
🔜 **Advanced Logging**: Per-task loss tracking  

## 📝 Training Output

Training will create:

- `outputs/`: Model checkpoints and final model
- `logs/`: TensorBoard logs
- Console output with metrics:
  - `loss`: Combined MLM + NSP loss
  - `eval_loss`: Validation loss
  - `eval_mlm_accuracy`: Masked token prediction accuracy
  - `eval_nsp_accuracy`: Next sentence prediction accuracy

## 🔧 Customization

### Adjusting for Different Hardware

For different GPU memory:

```yaml
# config.yaml
training:
  per_device_train_batch_size: 8  # Reduce if OOM
  gradient_accumulation_steps: 4  # Increase to maintain effective batch size
  fp16: true  # Keep enabled for A100
```

Effective batch size = `batch_size × gradient_accumulation_steps × num_gpus`

### Using Different Base Models

```yaml
# config.yaml
model:
  name: "bert-large-uncased"  # Or any HuggingFace BERT model
```

## 🐛 Troubleshooting

### Out of Memory (OOM)
- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps`
- Reduce `max_length`

### Slow Training
- Ensure `fp16: true` on A100
- Check `dataloader_num_workers` (try 2-4)
- Enable `dataloader_pin_memory: true`

### Data Loading Issues
- Check JSONL format is valid
- Ensure 'sentences' field exists in all entries
- Verify file path is correct

## 📚 Dependencies

Core libraries:
- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- Datasets >= 2.14.0
- Accelerate >= 0.20.0

See `requirements.txt` for full list.

## 📄 License

[Add your license here]

## 👥 Authors

Yannik - Master's Thesis Project

## 🙏 Acknowledgments

- HuggingFace Transformers for the excellent BERT implementation
- Google Colab for providing A100 GPU access
