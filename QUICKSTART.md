# Quick Start Guide

## Setup

1. **Install dependencies:**
   ```bash
   make install
   ```

2. **Verify your data format:**
   Your JSONL file should look like:
   ```json
   {"sentences": ["First sentence.", "Second sentence."], "sic2": "60", "sic3": "602", "sic4": "6025"}
   ```

## Training

### On Google Colab (A100)

```bash
# Mount your Google Drive first in Colab
from google.colab import drive
drive.mount('/content/drive')

# Then run training
!make train-data-colab DATA=/content/drive/MyDrive/businessbert_data/sample.jsonl
```

### Locally (for testing)

```bash
make train-local DATA=./data/sample.jsonl
```

## Configuration

Edit `config.yaml` to adjust:

- **Batch size:** Change `per_device_train_batch_size` (default: 16)
- **Epochs:** Change `num_train_epochs` (default: 3)
- **Learning rate:** Change `learning_rate` (default: 5e-5)
- **Max sequence length:** Change `max_length` (default: 512)

## Monitoring

### TensorBoard
```bash
tensorboard --logdir ./logs
```

### Watch training progress
Training will output:
- Loss (combined MLM + NSP)
- MLM accuracy (masked token prediction)
- NSP accuracy (next sentence prediction)

## Output

After training, you'll find:
- **Model checkpoint:** `./outputs/`
- **Logs:** `./logs/`
- **Best model:** Automatically saved based on validation loss

## Next Steps

The project is designed to easily add custom classification heads:
1. Add custom model class in `src/models/`
2. Extend the trainer for multi-task learning
3. Update data collator to include SIC labels

