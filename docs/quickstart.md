# Quickstart

## Install

```bash
git clone --recurse-submodules https://github.com/SSLab-CSE-IITB/tecod.git
cd tecod
uv venv
source .venv/bin/activate
uv pip install -e .
```

If the repository was cloned without `--recurse-submodules`, run:

```bash
git submodule update --init --recursive
```

## Prepare Data

The repository includes a BIRD dev financial sample:

```text
examples/financial/
├── financial.sqlite
└── raw_data.json
```

Raw data for `process-data` must include `text`, the configured SQL field
(`SQL` by default), and `nlq_masked`.

Generated TeCoD runtime files go under `data/`:

```text
data/
├── database.sqlite
├── examples.jsonl
├── templates.jsonl
├── schema.prompt
├── index.db
└── c_templs/
```

To create the deterministic TeCoD state files without loading any models:

```bash
python main.py \
  -c +data=financial \
  -c tecod.model_id=XGenerationLab/XiYanSQL-QwenCoder-14B-2504 \
  process-data examples/financial/raw_data.json --prepare-only
```

Check the prepared files:

```bash
python main.py -c +data=financial status
```

Download the local models before indexing or local generation. OpenAI-compatible
mode still uses local embeddings and local NLI.

```bash
huggingface-cli download Qwen/Qwen3-Embedding-4B
huggingface-cli download smitxxiv/Qwen3-Re4B-SQL-TeCoD-TMM
huggingface-cli download XGenerationLab/XiYanSQL-QwenCoder-14B-2504
```

Create the vector index from existing `examples.jsonl`. This loads the embedding
model:

```bash
python main.py -c +data=financial create-index
```

For local constrained decoding, compile templates. This loads the local SQL
generation model:

```bash
python main.py -c +data=financial compile-templates
```

## Run

```bash
python main.py --env local -c +data=financial tecod
```

Python API:

```python
from src.api import TeCoD

with TeCoD(
    config_overrides=["+data=financial"],
) as tecod:
    result = tecod.generate(
        "What is the count of accounts opting for post-transaction issuance that are located in north Moravia?"
    )
    print(result.pred_sql)
```

OpenAI-compatible provider:

```bash
export OPENAI_API_KEY=...
python main.py --env openai -c +data=financial -c tecod.model_id=gpt-4o-mini tecod
```

Check setup:

```bash
python main.py -c +data=financial status
python main.py version
```

The `+data=financial` profile is defined in `conf/data/financial.yaml`. You can
also set `TECOD_DATA_DIR` and `TECOD_DB_PATH` to point the same profile at a
different prepared data directory and SQLite database.

## Command Dependencies

| Command | Writes | Model requirements |
| --- | --- | --- |
| `process-data --prepare-only` | `examples.jsonl`, `templates.jsonl`, `schema.prompt` | None |
| `create-index` | `index.db` | Local embedding model |
| `compile-templates` | `c_templs/*.pkl` | Local embedding model and local SQL model |
| `tecod --env openai` | None | Local embedding and NLI models, plus API credentials |
| `tecod --env local` | None | Local embedding, NLI, and SQL models |
