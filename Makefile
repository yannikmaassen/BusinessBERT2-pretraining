.PHONY: help train-data-colab train-local install clean wandb-login wandb-status

# Python and config paths
PY := python3
CONFIG := config.yaml

help:
	@echo "Available commands:"
	@echo "  make install              - Install dependencies"
	@echo "  make wandb-login          - Login to Weights & Biases"
	@echo "  make wandb-status         - Check W&B login status"
	@echo "  make train-data-colab     - Train on Colab with custom data"
	@echo "  make train-local          - Train locally with custom data"
	@echo "  make clean                - Clean outputs and cache"
	@echo ""
	@echo "Usage examples:"
	@echo "  make wandb-login"
	@echo "  make train-data-colab DATA=/content/drive/MyDrive/businessbert_data/sample.jsonl"
	@echo "  make train-local DATA=./data/sample.jsonl"
	@echo ""
	@echo "Optional parameters:"
	@echo "  WANDB_NAME=my-run-name    - Set custom W&B run name"

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

wandb-login:
	@echo "Logging in to Weights & Biases..."
	$(PY) -c "import wandb; wandb.login()"

wandb-status:
	@echo "Checking W&B login status..."
	@$(PY) -c "import wandb; print('✓ Logged in as:', wandb.api.viewer())" || echo "✗ Not logged in. Run: make wandb-login"

train-data-colab:
	@if [ -z "$(DATA)" ]; then echo "Usage: make train-data-colab DATA=/content/drive/MyDrive/businessbert_data/sample.jsonl"; exit 1; fi
	$(PY) -m src.training.pretrain --config $(CONFIG) --data $(DATA) $(if $(WANDB_NAME),--wandb_run_name $(WANDB_NAME),)

train-local:
	@if [ -z "$(DATA)" ]; then echo "Usage: make train-local DATA=./data/sample.jsonl"; exit 1; fi
	$(PY) -m src.training.pretrain --config $(CONFIG) --data $(DATA) $(if $(WANDB_NAME),--wandb_run_name $(WANDB_NAME),)

clean:
	rm -rf outputs/
	rm -rf logs/
	rm -rf wandb/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
