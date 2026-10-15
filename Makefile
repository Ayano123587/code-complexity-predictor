.PHONY: install data train neural test
install:
	python -m pip install -e ".[dev]"
data:
	python -m src.data.tasty
train: data
	python -m src.models.train_model
neural: data
	python -m src.models.train_model --backend cnn --output models/tasty-cnn
test:
	python -m pytest -q

smoke:
	python -m src.data.make_dataset
	python -m src.models.train_model --data data/processed/smoke.jsonl --output models/linear
