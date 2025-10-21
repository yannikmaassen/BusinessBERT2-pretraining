.PHONY: help train-data-colab train-local install clean

# Python and config paths
PY := python3
CONFIG := config.yaml

help:
	@echo "Available commands:"
	@echo "  make install              - Install dependencies"
	@echo "  make train-data-colab     - Train on Colab with custom data"
	@echo "  make train-local          - Train locally with custom data"
	@echo "  make clean                - Clean outputs and cache"
	@echo ""
	@echo "Usage examples:"
	@echo "  make train-data-colab DATA=/content/drive/MyDrive/businessbert_data/sample.jsonl"
	@echo "  make train-local DATA=./data/sample.jsonl"

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

train-data-colab:
	@if [ -z "$(DATA)" ]; then echo "Usage: make train-data-colab DATA=/content/drive/MyDrive/businessbert_data/sample.jsonl"; exit 1; fi
	$(PY) -m src.training.pretrain --config $(CONFIG) --data $(DATA)

train-local:
	@if [ -z "$(DATA)" ]; then echo "Usage: make train-local DATA=./data/sample.jsonl"; exit 1; fi
	$(PY) -m src.training.pretrain --config $(CONFIG) --data $(DATA)

clean:
	rm -rf outputs/
	rm -rf logs/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

