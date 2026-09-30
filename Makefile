.PHONY: setup test data data-build data-v3 train-sft eval kaggle-requirements

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

# Build, then the CPU gates (decision 42): leak probes and duplicate check. Each fails hard.
data: data-build
	$(PY) scripts/leak_probe.py --config $(DATA_CONFIG)
	$(PY) scripts/check_duplicates.py --config $(DATA_CONFIG)

# Build and build checks only; the Kaggle notebooks use this, since the gates already passed locally on the same commit.
data-build:
	$(PY) scripts/build_data.py --config $(DATA_CONFIG)

# v3 data (task 3.1, decision 56): the Banking77 log domain and the full Banking77 test split, next to the
# v1.3 files, which must exist (make data-build); then the leak probes on the two new files.
data-v3:
	$(PY) scripts/build_v3_data.py
	$(PY) scripts/leak_probe.py --config $(DATA_CONFIG) --splits v3_banking77_train v3_banking77_test_full --out data/v3_leak_probe.json

train-sft:
	$(PY) scripts/train_sft.py --config $(CONFIG)

eval:
	@test -n "$(CKPT)" || (echo "usage: make eval CKPT=runs/<run_name> [SPLITS=...] (or CKPT=base EVAL_CONFIG=configs/base_06b.yaml)"; exit 1)
	$(PY) scripts/evaluate.py --ckpt $(CKPT) $(if $(SPLITS),--splits $(SPLITS),) $(if $(EVAL_CONFIG),--config $(EVAL_CONFIG),)

kaggle-requirements:
	uv run python scripts/export_kaggle_requirements.py
