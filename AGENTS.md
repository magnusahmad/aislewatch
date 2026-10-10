# Instructions for coding agents

- Read [docs/SPEC.md](docs/SPEC.md) before changing anything. It is the source of truth; update it when a decision changes.
- Python 3.12 with `uv`. Package code lives in `src/aislewatch/`; the CLI entry point is `aw`.
- The caption schema and structured prompt are a contract shared by serving, teachers, training and evaluation. Change them only together, bump the schema version, and re-run the baseline.
- Never train on or edit labels in the `test` split. The test set is hand-labelled and fixed.
- Never commit `data/`, `models/`, `.env` or any API key.
- Model prices and endpoints go in `config/models.yaml` with the date checked; don't hard-code them.
- Plain code decides when to train and whether to promote an adapter. Agents prepare data; they don't trigger training or change the live adapter.
