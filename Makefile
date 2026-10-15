.PHONY: install data train neural test
install:
	python -m pip install -e ".[dev]"
data:
	python -m src.data.make_dataset
train: data
	python -m src.models.train_model
neural: data
	python -m src.models.train_model --backend cnn --output models/cnn
test:
	python -m pytest -q
