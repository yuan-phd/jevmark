.PHONY: setup test data train-sft eval kaggle-requirements

CONFIG ?= configs/base.yaml
DATA_CONFIG ?= configs/data.yaml
CKPT ?=
# uv locally; the Kaggle notebook runs make with PY=python.
PY ?= uv run python
SPLITS ?=
# evaluate.py uses a run's own config.yaml; set EVAL_CONFIG for --ckpt base.
EVAL_CONFIG ?=

setup:
	uv sync --extra dev

test:
	uv run pytest -q

data:
	$(PY) scripts/build_data.py --config $(DATA_CONFIG)

train-sft:
	$(PY) scripts/train_sft.py --config $(CONFIG)

eval:
	@test -n "$(CKPT)" || (echo "usage: make eval CKPT=runs/<run_name> [SPLITS=...] (or CKPT=base EVAL_CONFIG=configs/base_06b.yaml)"; exit 1)
	$(PY) scripts/evaluate.py --ckpt $(CKPT) $(if $(SPLITS),--splits $(SPLITS),) $(if $(EVAL_CONFIG),--config $(EVAL_CONFIG),)

kaggle-requirements:
	uv run python scripts/export_kaggle_requirements.py
