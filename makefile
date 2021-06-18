PROJECT_NAME = novartis-datathon-2020

init:
	export PYTHONPATH="/content/drive/MyDrive/Projects/novartis-datathon-2020"
	export PYTHONPATH=.

create-environment:
	@echo ">>> Creating conda DEV environment."
	conda env create -f environment.yml
	conda env update -f environment.yml
	@echo "!!!RUN RIGHT NOW:\nconda activate novartis-datathon-2020"

update-environment:
	@echo ">>> Updating conda DEV environment."
	conda env update -f environment.yml
	@echo "!!!RUN RIGHT NOW:\nconda activate novartis-datathon-2020"

lint:
	tasks/lint.sh

create-features:
	python src/features/build_features.py --help
	python src/features/build_features.py --win_len 24
	python src/features/build_features.py --win_len 48

train-models:
	python src/models/train_model.py --help
	python src/models/train_model.py --dataset win_len_24 --n_iter_search 100 --target_cols y_0

forecast:
	python src/models/predict_model.py --help
	python src/models/predict_model.py --dataset win_len_24 --model_class HistGradientBoostingRegressor

	