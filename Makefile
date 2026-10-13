# One command to start the dashboard: `make run`.
# Each step runs only if its output is missing. DrugBank is read from
# DRUGBANK_PATH (default ~/Documents/drugbank/drugbank_all_full_database.xml.zip).
PY ?= .venv/bin/python

.PHONY: run setup test v2 clean-derived

run: setup data_raw/dti_full.parquet data_raw/drug_attributes.parquet \
     models/attribute_model_seen.pkl results/results_cold_similarity.json \
     results/results_attribute_ablation.json models/predictions.parquet
	$(PY) app.py

setup: .venv/bin/python
.venv/bin/python:
	python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt

data_raw/dti_full.parquet:
	$(PY) src/datasources/drugbank_loader.py

data_raw/drug_attributes.parquet: data_raw/dti_full.parquet
	$(PY) src/datasources/drugbank_attributes.py

# The app's model (v3, attribute features): evaluation plus two calibrated copies.
models/attribute_model_seen.pkl results/results_attributes.json: data_raw/drug_attributes.parquet
	$(PY) train_attribute_model.py

results/results_cold_similarity.json: models/attribute_model_seen.pkl
	$(PY) scripts/analyze_cold_holdout.py

results/results_attribute_ablation.json: models/attribute_model_seen.pkl
	$(PY) scripts/ablate_attributes_validation.py

models/predictions.parquet: models/attribute_model_seen.pkl
	$(PY) predict_links.py

# The v2 graph-only model, kept reproducible (writes results/results.json).
v2: data_raw/dti_full.parquet
	$(PY) train_ml_models.py

test: setup
	$(PY) -m pytest -q

# Removes everything derived from DrugBank (keeps the committed results files).
clean-derived:
	rm -rf data_raw data models
