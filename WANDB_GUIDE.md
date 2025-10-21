# Weights & Biases Integration Guide

## Overview

Your BERT pretraining project now includes comprehensive Weights & Biases logging with:
- **Live training metrics** with progress bars in Colab
- **Separate tracking** for MLM and NSP objectives
- **Total loss** and individual objective losses
- **Accuracy metrics** for both MLM and NSP
- **Organized dashboard** with categorized metrics

## Setup

### 1. Install W&B (included in requirements.txt)
```bash
make install
```

### 2. Login to W&B

**On Colab:**
```python
import wandb
wandb.login()
```

Or use the Makefile:
```bash
make wandb-login
```

**Get your API key from:** https://wandb.ai/authorize

### 3. Verify Login
```bash
make wandb-status
```

## Usage

### Basic Training with W&B

**On Colab:**
```bash
make train-data-colab DATA=/content/drive/MyDrive/businessbert_data/sample.jsonl
```

**Locally:**
```bash
make train-local DATA=./data/sample.jsonl
```

### Custom Run Name

```bash
make train-data-colab DATA=/path/to/data.jsonl WANDB_NAME=bert-experiment-1
```

Or directly:
```bash
python -m src.training.pretrain \
    --config config.yaml \
    --data /path/to/data.jsonl \
    --wandb_run_name bert-experiment-1
```

## Metrics Logged to W&B

### Training Metrics (logged every 100 steps by default)

**Total Metrics:**
- `metrics/total/loss` - Combined MLM + NSP loss
- `training/learning_rate` - Current learning rate
- `training/epoch` - Current epoch

**MLM Metrics:**
- `metrics/MLM/train_mlm_loss` - Masked Language Model loss
- `metrics/MLM/train_mlm_accuracy` - MLM prediction accuracy

**NSP Metrics:**
- `metrics/NSP/train_nsp_loss` - Next Sentence Prediction loss
- `metrics/NSP/train_nsp_accuracy` - NSP classification accuracy

### Evaluation Metrics (logged every 5000 steps by default)

**Total:**
- `evaluation/loss` - Combined validation loss

**MLM:**
- `evaluation/MLM/mlm_accuracy` - MLM accuracy on validation set
- `evaluation/MLM/mlm_loss` - MLM loss on validation set

**NSP:**
- `evaluation/NSP/nsp_accuracy` - NSP accuracy on validation set
- `evaluation/NSP/nsp_loss` - NSP loss on validation set

## Configuration

Edit `config.yaml` to customize W&B settings:

```yaml
# Logging
logging:
  report_to: ["wandb", "tensorboard"]  # Enable both W&B and TensorBoard
  logging_dir: "./logs"

# Weights & Biases configuration
wandb:
  project: "businessbert-pretraining"  # Your W&B project name
  entity: null  # Your W&B username/team (or null for default)
  name: null  # Custom run name (or null for auto-generation)
  tags: ["bert", "pretraining", "mlm", "nsp"]
  notes: "BERT pretraining with MLM and NSP objectives on business documents"
  log_model: "checkpoint"  # Options: false, "checkpoint", "end"
```

### Logging Frequency

```yaml
training:
  logging_steps: 100  # Log metrics every 100 steps
  eval_steps: 5000    # Evaluate every 5000 steps
  save_steps: 5000    # Save checkpoint every 5000 steps
```

## Progress Bars in Colab

The training now shows **live progress bars** with:

**Training Progress:**
```
Epoch 1/3: 100%|██████████| 1000/1000 [15:23<00:00, 1.08it/s, loss=2.345, lr=4.8e-5]
```

**Evaluation Progress:**
```
Evaluation: 100%|██████████| 50/50 [00:45<00:00, 1.11it/s, loss=2.134, mlm_acc=0.567, nsp_acc=0.892]
```

## Viewing Results

### During Training

1. **Console Output:** See progress bars and metrics in real-time
2. **W&B Dashboard:** Click the link printed at training start
3. **TensorBoard:** Run `tensorboard --logdir ./logs`

### After Training

Visit your W&B project dashboard:
```
https://wandb.ai/<username>/<project-name>
```

## Example W&B Dashboard Layout

Your metrics will be organized as:

```
📊 Metrics
├── 📈 Total
│   └── loss
├── 📘 MLM (Masked Language Modeling)
│   ├── train_mlm_loss
│   ├── train_mlm_accuracy
│   ├── eval_mlm_loss
│   └── eval_mlm_accuracy
├── 📗 NSP (Next Sentence Prediction)
│   ├── train_nsp_loss
│   ├── train_nsp_accuracy
│   ├── eval_nsp_loss
│   └── eval_nsp_accuracy
└── ⚙️ Training
    ├── learning_rate
    └── epoch
```

## Disable W&B (if needed)

To train without W&B, edit `config.yaml`:

```yaml
logging:
  report_to: ["tensorboard"]  # Remove "wandb"
```

Or set environment variable:
```bash
export WANDB_MODE=disabled
make train-data-colab DATA=/path/to/data.jsonl
```

## Tips for Colab

### 1. Login Once Per Session
```python
# At the start of your Colab notebook
import wandb
wandb.login()
```

### 2. Set Project Name
```python
# Optionally set via environment
import os
os.environ['WANDB_PROJECT'] = 'my-custom-project'
```

### 3. Monitor Live
Keep the W&B dashboard open in another tab to watch training progress in real-time.

### 4. Save API Key
Store your API key in Colab secrets for automatic login:
1. Click 🔑 in left sidebar
2. Add secret: `WANDB_API_KEY`
3. Use in code:
```python
from google.colab import userdata
import wandb
wandb.login(key=userdata.get('WANDB_API_KEY'))
```

## Troubleshooting

### "wandb not installed"
```bash
pip install wandb
```

### "Not logged in"
```bash
make wandb-login
```

### W&B run not appearing
- Check internet connection
- Verify login: `make wandb-status`
- Check project name in config.yaml

### Too much logging slowing down training
Increase `logging_steps` in config.yaml:
```yaml
training:
  logging_steps: 500  # Log less frequently
```

## Advanced: Custom Metrics

The `WandbMetricsCallback` automatically organizes all metrics. To add custom metrics, log them in your training code:

```python
import wandb
wandb.log({"custom_metric": value}, step=step)
```

Metrics will be automatically categorized based on their name:
- Contains "mlm" → `metrics/MLM/`
- Contains "nsp" → `metrics/NSP/`
- Contains "loss" → `metrics/total/`
- Otherwise → `metrics/`

## Summary

✅ **Automatic W&B Integration** - Just login and train
✅ **Live Metrics** - See training progress in real-time
✅ **Organized Dashboard** - MLM and NSP metrics separated
✅ **Progress Bars** - Visual feedback in Colab
✅ **Model Checkpointing** - Optionally log models to W&B
✅ **Easy Configuration** - All settings in config.yaml

Your training metrics are now comprehensively tracked and easy to monitor! 🚀

