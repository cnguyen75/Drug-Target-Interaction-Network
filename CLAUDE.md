# Working in this repository

Follow `CODING-RULES.md` in this folder for all code, comments, and documentation. Run
`python3 rules_check.py .` and the formatters named in `rules_check.py` before every commit;
a commit with findings is not ready.

Commit as Colin Nguyen <26684223+cnguyen75@users.noreply.github.com>. Never use any other
identity. Do not push unless asked.

Nothing here may name an employer's internal systems, customers, colleagues, vendors, or
metrics. No credentials, tokens, or private paths in any file, ever.

<!-- Repo-specific notes below: what is private, how to build, what the tests cover. -->

## Repo notes

- Run: `make run` (creates `.venv`, parses the DrugBank export at `DRUGBANK_PATH`, trains, scores,
  then serves the Dash app on http://127.0.0.1:8050). Each step runs only if its output is missing.
- Tests: `make test` (synthetic data only). Run it, plus `ruff check . && ruff format --check .`,
  `npx prettier --check assets`, and `python3 rules_check.py .`, before every commit. ruff is
  configured in `pyproject.toml` under `[tool.ruff]`.
- Data: DrugBank is licensed (CC BY-NC 4.0 for the full database). The export and everything
  derived row-for-row from it (`data_raw/`, `data/`, `models/`, parsed tables, graphs, embeddings,
  predictions) are gitignored and must never be committed or redistributed. Only the four aggregate
  result JSONs under `results/` are tracked.
- The README's results block between `<!-- results:start -->` and `<!-- results:end -->` is
  written by `scripts/site_results.py`; do not hand-edit its numbers.
