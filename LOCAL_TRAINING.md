# Running Locally - Complete Guide

## Quick Start (3 Steps)

### 1. Install Dependencies
```bash
cd /Users/yannik/dev/masterthesis/BusinessBERT2-pretraining
make install
```

### 2. (Optional) Login to Weights & Biases
```bash
make wandb-login
```
Skip this if you want to train without W&B logging.

### 3. Run Training
```bash
make train-local DATA=./data/sample.jsonl
```

That's it! Training will start immediately.

---

## Detailed Instructions

### Step-by-Step Setup

#### 1. Navigate to Project Directory
```bash
cd /Users/yannik/dev/masterthesis/BusinessBERT2-pretraining
```

#### 2. (Optional) Create Virtual Environment
```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# You should see (.venv) in your terminal prompt
```

#### 3. Install Dependencies
```bash
make install
```

This installs:
- PyTorch
- Transformers (HuggingFace)
- Weights & Biases
- TensorBoard
- All other requirements

#### 4. Set Up Weights & Biases (Optional)

**If you want W&B logging:**
```bash
make wandb-login
```
Enter your API key when prompted (get it from https://wandb.ai/authorize)

**If you don't want W&B logging:**
Edit `config.yaml` and change:
```yaml
logging:
  report_to: ["tensorboard"]  # Remove "wandb"
```

#### 5. Run Training

**Using the sample data:**
```bash
make train-local DATA=./data/sample.jsonl
```

**Using your own data:**
```bash
make train-local DATA=/path/to/your/data.jsonl
```

**With custom W&B run name:**
```bash
make train-local DATA=./data/sample.jsonl WANDB_NAME=my-experiment-1
```

---

## Alternative: Direct Python Command

Instead of using `make`, you can run directly:

```bash
python -m src.training.pretrain \
    --config config.yaml \
    --data ./data/sample.jsonl
```

**With options:**
```bash
python -m src.training.pretrain \
    --config config.yaml \
    --data ./data/sample.jsonl \
    --wandb_run_name my-experiment \
    --resume_from_checkpoint ./outputs/checkpoint-5000
```

---

## What You'll See

### At Start:
```
Loading model and tokenizer: bert-base-uncased
Model loaded with 109,482,240 parameters
Loaded 10 documents
Created 30 training examples
Train examples: 28
Eval examples: 2

✓ Weights & Biases initialized: sparkling-sun-42
  View run at: https://wandb.ai/username/businessbert-pretraining/runs/abc123

================================================================================
Starting training...
================================================================================
```

### During Training:
```
Epoch 1/3:  15%|█▌        | 4/28 [00:12<01:20, 3.35s/it, loss=10.234, lr=5e-05]
```

### During Evaluation:
```
Evaluation: 100%|██████████| 2/2 [00:01<00:00, loss=9.123, mlm_acc=0.234, nsp_acc=0.567]

{'eval_loss': 9.123, 'eval_mlm_accuracy': 0.234, 'eval_nsp_accuracy': 0.567}
```

---

## Monitoring Training Locally

### Option 1: TensorBoard
In a new terminal:
```bash
cd /Users/yannik/dev/masterthesis/BusinessBERT2-pretraining
tensorboard --logdir ./logs
```

Then open: http://localhost:6006

### Option 2: Weights & Biases
Click the URL printed at training start, or visit:
https://wandb.ai/

### Option 3: Watch Output Files
```bash
# Watch training progress
watch -n 5 'ls -lh outputs/'

# View latest checkpoint
ls -lh outputs/checkpoint-*/
```

---

## Configuration for Local Training

The default `config.yaml` is optimized for A100 GPUs. For local training on smaller GPUs:

### For GPUs with Less Memory (8-16GB)

Edit `config.yaml`:
```yaml
training:
  per_device_train_batch_size: 4   # Reduce from 16
  gradient_accumulation_steps: 8   # Increase to maintain effective batch size
  fp16: true                        # Keep enabled if GPU supports it
  dataloader_num_workers: 2         # Reduce from 4
```

### For CPU Training (Not Recommended - Very Slow)

Edit `config.yaml`:
```yaml
training:
  per_device_train_batch_size: 2
  fp16: false  # CPU doesn't support fp16
  dataloader_num_workers: 2
```

### For Quick Testing

```yaml
training:
  num_train_epochs: 1
  logging_steps: 10
  save_steps: 100
  eval_steps: 100
```

---

## Output Files

After training, you'll have:

```
BusinessBERT2-pretraining/
├── outputs/
│   ├── checkpoint-5000/       # Saved checkpoints
│   │   ├── pytorch_model.bin
│   │   ├── config.json
│   │   └── ...
│   ├── pytorch_model.bin      # Final model
│   ├── config.json
│   └── tokenizer files
├── logs/
│   └── events.out.tfevents.*  # TensorBoard logs
└── wandb/                     # W&B logs (if enabled)
```

---

## Troubleshooting Local Training

### Issue: "CUDA out of memory"
**Solution:**
```yaml
# In config.yaml
training:
  per_device_train_batch_size: 4  # Reduce
  gradient_accumulation_steps: 8  # Increase
```

### Issue: "No module named 'src'"
**Solution:**
```bash
# Make sure you're in the project root
cd /Users/yannik/dev/masterthesis/BusinessBERT2-pretraining
python -m src.training.pretrain --config config.yaml --data ./data/sample.jsonl
```

### Issue: Slow data loading
**Solution:**
```yaml
# In config.yaml
training:
  dataloader_num_workers: 0  # Try single worker
```

### Issue: W&B not working
**Solution:**
```bash
# Disable W&B
export WANDB_MODE=disabled
make train-local DATA=./data/sample.jsonl
```

Or edit `config.yaml`:
```yaml
logging:
  report_to: ["tensorboard"]
```

---

## Resume Training from Checkpoint

```bash
python -m src.training.pretrain \
    --config config.yaml \
    --data ./data/sample.jsonl \
    --resume_from_checkpoint ./outputs/checkpoint-5000
```

---

## Test with Small Dataset First

The sample data has only 10 documents, perfect for testing:

```bash
# Quick test run
make train-local DATA=./data/sample.jsonl
```

Expected training time with sample data:
- **GPU (RTX 3090/4090)**: ~2-3 minutes
- **GPU (GTX 1080)**: ~5-10 minutes
- **CPU**: ~30-60 minutes (not recommended)

---

## Complete Example Session

```bash
# 1. Navigate to project
cd /Users/yannik/dev/masterthesis/BusinessBERT2-pretraining

# 2. Activate virtual environment (if using)
source .venv/bin/activate

# 3. Install dependencies
make install

# 4. Login to W&B (optional)
make wandb-login

# 5. Check everything is ready
make wandb-status

# 6. Run training with sample data
make train-local DATA=./data/sample.jsonl

# 7. Monitor in another terminal (optional)
tensorboard --logdir ./logs

# 8. View outputs
ls -lh outputs/
```

---

## Using Your Own Data

Your data should be in JSONL format with this structure:

```json
{"sentences": ["Sentence 1.", "Sentence 2.", "Sentence 3."], "sic2": "60", "sic3": "602", "sic4": "6025"}
{"sentences": ["Another doc.", "More sentences."], "sic2": "35", "sic3": "357", "sic4": "3571"}
```

Then run:
```bash
make train-local DATA=/path/to/your/data.jsonl
```

---

## Next Steps After Local Testing

1. **Test locally** with sample data to verify everything works
2. **Try with small subset** of your real data
3. **Adjust config** based on your local GPU
4. **Upload to Colab** for full-scale training on A100

---

## Quick Reference

```bash
# Install
make install

# Train
make train-local DATA=./data/sample.jsonl

# Train with custom name
make train-local DATA=./data/sample.jsonl WANDB_NAME=test-run-1

# Clean outputs
make clean

# Check W&B status
make wandb-status

# View help
make help
```

That's everything you need to run locally! 🚀

