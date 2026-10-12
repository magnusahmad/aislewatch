# Instructions for coding agents

- Read [docs/SPEC.md](docs/SPEC.md) before changing anything. It is the source of truth; update it when a decision changes.
- Python 3.12 with `uv`. Package code lives in `src/aislewatch/`; the CLI entry point is `aw`.
- The caption schema and structured prompt are a contract shared by serving, teachers, training and evaluation. Change them only together, bump the schema version, and re-run the baseline.
- Never train on or edit labels in the `test` or `test-real` splits. Both are hand-labelled and fixed.
- All frame sampling, resizing, cropping, prompt building and schema handling goes through `aislewatch.preprocess`. Training, serving, teachers and evaluation must not re-implement it.
- Qwen3.5 thinking mode is disabled in training and in every serving request.
- Never commit `data/`, `models/`, `.env` or any API key.
- Model prices and endpoints go in `config/models.yaml` with the date checked; don't hard-code them.
- Plain code decides when to train and whether to promote an adapter. Agents prepare data; they don't trigger training or change the live adapter.
