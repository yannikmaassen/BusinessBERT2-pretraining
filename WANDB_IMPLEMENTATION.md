# W&B and Progress Tracking - Implementation Summary

## ✅ What Was Added

### 1. **Weights & Biases Integration**

#### Config Updates (`config.yaml`)
- Added W&B configuration section with project name, tags, and settings
- Enabled both W&B and TensorBoard logging
- Configurable run names and project settings

#### Training Script Updates (`pretrain.py`)
- `setup_wandb()` function to initialize W&B with project config
- `--wandb_run_name` CLI argument for custom run names
- Automatic W&B session cleanup on training completion
- Prints W&B run URL for easy access

### 2. **Custom Metrics Callback** (`callbacks.py`)
- **WandbMetricsCallback** class that organizes metrics into categories:
  - `metrics/MLM/` - All MLM-related metrics
  - `metrics/NSP/` - All NSP-related metrics
  - `metrics/total/` - Combined loss metrics
  - `training/` - Learning rate and epoch info
  - `evaluation/` - Validation metrics

### 3. **Enhanced Trainer** (`trainer.py`)
- Progress bars with **tqdm** for both training and evaluation
- Live metrics display in Colab: loss, MLM accuracy, NSP accuracy
- Separate logging for MLM and NSP losses/accuracies
- Enhanced evaluation with real-time progress updates

### 4. **Makefile Commands**
- `make wandb-login` - Easy W&B authentication
- `make wandb-status` - Check if logged in
- `WANDB_NAME` parameter for custom run names

## 📊 Metrics Logged

### During Training (every 100 steps)
- **Total Loss**: Combined MLM + NSP loss
- **MLM Loss**: Masked language modeling loss
- **MLM Accuracy**: Prediction accuracy for masked tokens
- **NSP Loss**: Next sentence prediction loss
- **NSP Accuracy**: Classification accuracy for sentence pairs
- **Learning Rate**: Current learning rate
- **Epoch**: Current epoch progress

### During Evaluation (every 5000 steps)
- **Eval Loss**: Total validation loss
- **Eval MLM Accuracy**: MLM performance on validation set
- **Eval MLM Loss**: MLM loss on validation set
- **Eval NSP Accuracy**: NSP performance on validation set
- **Eval NSP Loss**: NSP loss on validation set

## 🎯 Progress Indicators

### In Colab/Terminal
```
Epoch 1/3: 100%|██████████| 1000/1000 [15:23<00:00, 1.08it/s, loss=2.345]
```

### During Evaluation
```
Evaluation: 100%|██████████| 50/50 [00:45<00:00, loss=2.134, mlm_acc=0.567, nsp_acc=0.892]
```

## 🚀 Usage

### Step 1: Login to W&B (one-time setup)
```bash
make wandb-login
```

### Step 2: Train with W&B logging
```bash
# On Colab
make train-data-colab DATA=/content/drive/MyDrive/businessbert_data/sample.jsonl

# With custom run name
make train-data-colab DATA=/path/to/data.jsonl WANDB_NAME=experiment-1
```

### Step 3: Monitor Training
- **Colab**: Progress bars show live in notebook
- **W&B Dashboard**: Click the URL printed at start of training
- **TensorBoard**: `tensorboard --logdir ./logs`

## 📈 W&B Dashboard Organization

Metrics are automatically organized in your W&B dashboard:

```
📊 metrics/
├── 📁 MLM/
│   ├── train_mlm_loss
│   ├── train_mlm_accuracy
│   ├── eval_mlm_loss
│   └── eval_mlm_accuracy
├── 📁 NSP/
│   ├── train_nsp_loss
│   ├── train_nsp_accuracy
│   ├── eval_nsp_loss
│   └── eval_nsp_accuracy
├── 📁 total/
│   ├── loss
│   └── eval_loss
└── 📁 training/
    ├── learning_rate
    └── epoch
```

## 🔧 Configuration

All W&B settings in `config.yaml`:

```yaml
logging:
  report_to: ["wandb", "tensorboard"]
  
wandb:
  project: "businessbert-pretraining"
  entity: null  # Your W&B username
  name: null    # Auto-generated or custom
  tags: ["bert", "pretraining", "mlm", "nsp"]
```

## 🎨 What You'll See in Colab

### At Training Start
```
✓ Weights & Biases initialized: sparkling-sun-42
  View run at: https://wandb.ai/username/businessbert-pretraining/runs/abc123

Loading model and tokenizer: bert-base-uncased
Model loaded with 109,482,240 parameters
Loaded 1000 documents
Created 5000 training examples
Train examples: 4750
Eval examples: 250

================================================================================
Starting training...
================================================================================
```

### During Training
```
Epoch 1/3:  45%|████▌     | 450/1000 [07:12<08:47, 1.04it/s, loss=2.456, lr=4.5e-5]
```

### During Evaluation
```
Evaluation: 100%|██████████| 50/50 [00:45<00:00, loss=2.134, mlm_acc=0.567, nsp_acc=0.892]

{'eval_loss': 2.134, 'eval_mlm_accuracy': 0.567, 'eval_nsp_accuracy': 0.892}
```

## ✨ Key Features

1. **Live Monitoring**: See training progress in real-time both in Colab and W&B dashboard
2. **Organized Metrics**: MLM and NSP tracked separately for clear understanding
3. **Progress Bars**: Visual feedback with loss/accuracy in Colab
4. **Easy Setup**: Just `make wandb-login` once, then train normally
5. **Flexible**: Can disable W&B by editing config.yaml
6. **Custom Names**: Set run names via CLI or config

## 📚 Files Modified/Created

- ✅ `config.yaml` - Added W&B configuration
- ✅ `src/training/pretrain.py` - W&B initialization and cleanup
- ✅ `src/training/trainer.py` - Progress bars and enhanced metrics
- ✅ `src/training/callbacks.py` - NEW: Custom W&B metrics callback
- ✅ `Makefile` - W&B login commands
- ✅ `WANDB_GUIDE.md` - NEW: Comprehensive W&B documentation

## 🎯 Result

You now have:
- **Full W&B integration** with organized metric tracking
- **Live progress bars** in Colab showing loss and accuracies
- **Separate tracking** for MLM and NSP objectives
- **Easy-to-use commands** via Makefile
- **Comprehensive documentation** for reference

Everything is ready to use! Just login to W&B and start training! 🚀

