.PHONY: setup test data train-sft eval

CONFIG ?= configs/base.yaml
DATA_CONFIG ?= configs/data.yaml
CKPT ?=
SPLITS ?=

setup:
	uv sync --extra dev

test:
	uv run pytest -q

data:
	uv run python scripts/build_data.py --config $(DATA_CONFIG)

train-sft:
	uv run python scripts/train_sft.py --config $(CONFIG)

eval:
	@test -n "$(CKPT)" || (echo "usage: make eval CKPT=runs/<run_name> [SPLITS=...]"; exit 1)
	uv run python scripts/evaluate.py --ckpt $(CKPT) $(if $(SPLITS),--splits $(SPLITS),) --config $(CONFIG)
