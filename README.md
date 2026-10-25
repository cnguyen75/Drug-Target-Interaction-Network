# Drug–target interaction network

![The explore view: Imatinib, its known targets, drugs that share them, and predicted targets](docs/explore.png)

<sub>Interaction data from DrugBank 5.1, used under CC BY-NC 4.0 for non-commercial purposes.</sub>

A Dash app and pipeline that turn a DrugBank export into a drug–target network, learn which unrecorded drug–target pairs look like real interactions, and show the evidence for each suggestion. Drugs and human protein targets form a bipartite graph; Node2Vec embeds it; a random forest scores pairs from graph features and from drug attributes (chemical structure, protein family, chemical class, categories), so it can also suggest targets for a drug with no known interactions. The app has four views: **explore** (search a drug or target, see its neighbourhood and ranked suggestions), **predictions**, **evaluation**, and **methods**.

## Run it

You need a DrugBank full-database export (`drugbank_all_full_database.xml.zip`). DrugBank gives it free to academic users under its own licence; it is not in this repository and nothing derived from it is committed.

```bash
export DRUGBANK_PATH=~/Documents/drugbank/drugbank_all_full_database.xml.zip   # the default
make run        # creates .venv, parses, trains, scores, then serves http://127.0.0.1:8050
```

`make run` runs only the steps whose outputs are missing:

| Step | Command | Output (gitignored unless noted) |
|---|---|---|
| Parse DrugBank (streams the zip) | `python src/datasources/drugbank_loader.py` | `data_raw/*_full.parquet` |
| Parse drug and target attributes | `python src/datasources/drugbank_attributes.py` | `data_raw/*_attributes.parquet` |
| Train and evaluate the app's model (v3) | `python train_attribute_model.py` | `models/attribute_model_*.pkl`, `results/results_attributes.json` (committed) |
| Near-duplicate check | `python scripts/analyze_cold_holdout.py` | `results/results_cold_similarity.json` (committed) |
| Ablation (validation only) | `python scripts/ablate_attributes_validation.py` | `results/results_attribute_ablation.json` (committed) |
| Score suggestions | `python predict_links.py` | `models/predictions.parquet` |
| Serve | `python app.py` | |

The v2 graph-only model stays reproducible with `make v2` (`python train_ml_models.py`, which writes `results/results.json`).

Tests use synthetic data only: `make test`. Python 3.11; versions are pinned in `requirements.txt`.

## Results

![ROC AUC on held-out pairs for three splits, two models, and four baselines](figures/evaluation.svg)

<!-- results:start -->
DrugBank 5.1 (exported 2025-01-02): 7,393 drugs with a human target, 2,981 human targets, 18,038 interactions. Run 2026-10-12, commit `7b158a0`, seed 42.

ROC AUC on held-out pairs (one sampled non-interaction per known interaction), with 95% bootstrap intervals over test pairs:

| Method | Whole-drug holdout (unseen drugs) | Per-drug split (controls for target popularity) | Random edge split |
|---|---|---|---|
| Attribute model (v3, the app) | 0.852 (0.843–0.860) | 0.910 (0.898–0.921) | 0.871 (0.863–0.879) |
| Graph-only model (v2) | 0.545 (0.533–0.559) | 0.890 (0.878–0.902) | 0.746 (0.734–0.757) |
| Structural nearest neighbour | 0.692 (0.678–0.704) | 0.657 (0.635–0.678) | 0.779 (0.767–0.790) |
| Length-3 paths | 0.500 (0.500–0.500) | 0.815 (0.799–0.830) | 0.716 (0.708–0.724) |
| Preferential attachment | 0.500 (0.500–0.500) | 0.507 (0.485–0.530) | 0.645 (0.632–0.658) |
| Chance | 0.500 (0.500–0.500) | 0.500 (0.500–0.500) | 0.500 (0.500–0.500) |

On unseen drugs: 0.85 on the test set, 0.79 on the smaller validation set, and about 0.76 (validation) without curated categories and ATC codes, the fairer expectation for a genuinely new compound. 40 of 1,478 test drugs have a structurally identical twin among training drugs; without them the score is 0.86.

Default suggestion thresholds: drugs with known targets 0.95 (median 4 candidates); drugs with no known targets 0.90 (median 8 candidates). Rule: for each copy, the lowest of 0.50, 0.60, 0.70, 0.80, 0.90, 0.95 that leaves the median featured drug with at most 10 candidates.

Produced by `python src/datasources/drugbank_loader.py`, `python src/datasources/drugbank_attributes.py`, `python train_attribute_model.py`, `python scripts/analyze_cold_holdout.py`, `python scripts/ablate_attributes_validation.py` and `python predict_links.py`; the v2 graph-only model by `python train_ml_models.py`; table and figure by `python scripts/make_figures.py`.
<!-- results:end -->

## How it works

**Three splits.** The *whole-drug holdout* removes every interaction of 20% of drugs and asks the model to rank targets for them, which is the realistic use: suggesting targets for a drug the network has never seen. The *per-drug split (controls for target popularity)* hides half the interactions of 20% of drugs with two or more targets. Both pair each hidden interaction with a non-interaction for the same drug and a target of similar degree. The *random edge split* hides 20% of interactions at random against uniform random non-interactions. A quarter of each split's training drugs (or edges) is held back for validation; choices are made there and each test set is scored once.

**Two models.** The *attribute model (v3)* adds 8 features to the 12 graph features; each compares the drug with drugs already known to bind the target: structural similarity (Morgan fingerprints), similarity to binders of targets in the same Pfam family, overlap of DrugBank categories and ATC codes, and shared chemical class. The *graph-only model (v2)* needs a drug's existing interactions, so on unseen drugs it is near chance. The app uses two calibrated copies of v3: one trained like the per-drug split for drugs with known targets, one trained like the whole-drug holdout for drugs with none.

**Baselines.** Structural nearest neighbour scores a pair by the drug's highest similarity to the target's known binders, with no learning. Length-3 paths counts drug → target′ → drug′ → target routes; preferential attachment multiplies degrees; chance ties every pair. On the whole-drug holdout the two graph baselines score every pair 0. Common neighbours is not shown: a drug and a target never share a neighbour in a bipartite graph.

An earlier version of this README reported ROC AUC 0.93 on a "temporal" split. That figure came from code that leaked held-out edges into training features and could not be reproduced; the split was random, not temporal.

## Limitations

- **Unseen by the graph, not by DrugBank.** A held-out drug keeps its DrugBank structure, class, categories, and ATC codes. Categories and ATC codes are partly assigned because a drug's targets are known; the results section gives the unseen-drug score without them, the fairer expectation for a new compound.
- **Not temporal.** DrugBank has no per-interaction dates, so the evaluation cannot say how well the model anticipates future discoveries.
- **Negatives are unmeasured.** Sampled non-interactions include real interactions nobody has tested; balanced 1:1 test sets make precision look far better than it would in screening.
- **Scores are rankings.** Suggestion scores are calibrated with isotonic regression on validation pairs, but on balanced data; they are not the chance an interaction is real. Isotonic calibration is a step function, so many top scores tie; the app marks ties and orders them by target rarity.
- **Intervals** cover resampling of test pairs only, not other split seeds or DrugBank releases.
- **DrugBank entries** include ions, salts, supplements, and cell therapies. The app features approved small molecules only and hides other entry types from the graph by default.

## Licence

Interaction data from DrugBank 5.1, used under CC BY-NC 4.0 for non-commercial purposes.

Code: MIT (see `LICENSE`). DrugBank data is licensed separately by DrugBank (CC BY-NC 4.0 for the full database) and is not included. Fonts: Archivo and Courier Prime, SIL Open Font License (`assets/fonts/`).
