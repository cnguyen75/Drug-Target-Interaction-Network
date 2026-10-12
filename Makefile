# One command to start the dashboard: `make run`.
# Each step runs only if its output is missing. DrugBank is read from
# DRUGBANK_PATH (default ~/Documents/drugbank/drugbank_all_full_database.xml.zip).
PY ?= .venv/bin/python

.PHONY: run setup test clean-derived

run: setup data_raw/dti_full.parquet results/results.json models/predictions.parquet
	$(PY) app.py

setup: .venv/bin/python
.venv/bin/python:
	python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt

data_raw/dti_full.parquet:
	$(PY) src/datasources/drugbank_loader.py

results/results.json models/link_prediction_model.pkl: data_raw/dti_full.parquet
	$(PY) train_ml_models.py

models/predictions.parquet: models/link_prediction_model.pkl
	$(PY) predict_links.py

test: setup
	$(PY) -m pytest -q

# Removes everything derived from DrugBank (keeps results/results.json).
clean-derived:
	rm -rf data_raw data models
