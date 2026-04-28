# Reliable Answers for Recurring Questions: Boosting Text-to-SQL Accuracy with Template Constrained Decoding (TeCoD)

[![Project Page](https://img.shields.io/badge/project-page-blue.svg)](https://sslab-cse-iitb.github.io/tecod/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the public code and project page repository for the TeCoD paper. TeCoD is a template-guided constrained decoding system for text-to-SQL. It combines vector retrieval, NLI-based template selection, and multiple SQL generation modes, including partitioned constrained decoding for local models and OpenAI-compatible API generation for hosted models.

## Install

TeCoD targets Python 3.12+.

```bash
git clone --recurse-submodules https://github.com/SSLab-CSE-IITB/tecod.git
cd tecod

uv venv
source .venv/bin/activate
uv pip install -e .
```

If you cloned without submodules, initialize `src/pdec` before running TeCoD:

```bash
git submodule update --init --recursive
```

## Data Layout

The repository includes a BIRD dev financial sample:

```text
examples/financial/
├── financial.sqlite
└── raw_data.json
```

By default TeCoD reads runtime data from `data/`:

```text
data/
├── database.sqlite
├── examples.jsonl
├── templates.jsonl
├── schema.prompt
├── index.db
└── c_templs/
    ├── 0.pkl
    └── ...
```

Raw input to `process-data` must be a JSON array whose objects include:

- `text`: natural-language question.
- configured SQL field: defaults to `SQL`; override with `-c tecod.sql_key=<field>`.
- `nlq_masked`: question skeleton used by retrieval and NLI.

TeCoD does not generate `nlq_masked` automatically. Generate it during data preparation by masking literals and schema-specific spans while preserving the question structure.

Create the deterministic prepared files first. This step does not load any
models:

```bash
python main.py \
  -c +data=financial \
  -c tecod.model_id=XGenerationLab/XiYanSQL-QwenCoder-14B-2504 \
  process-data examples/financial/raw_data.json --prepare-only
```

Then build or refresh the vector index. This loads the local embedding model:

```bash
python main.py -c +data=financial create-index
```

For local constrained decoding, compile templates after the index exists. This
loads the local SQL generation model:

```bash
python main.py -c +data=financial compile-templates
```

## CLI Usage

Start an interactive local-model session:

```bash
python main.py --env local -c +data=financial tecod
```

Use an OpenAI-compatible endpoint:

```bash
export OPENAI_API_KEY=...
python main.py --env openai -c +data=financial -c tecod.model_id=gpt-4o-mini tecod
```

OpenAI-compatible mode only moves SQL generation to the API. TeCoD still uses local embeddings, local NLI, prepared examples/templates, `schema.prompt`, and the vector index.

Check the resolved configuration and expected files:

```bash
python main.py -c +data=financial status
python main.py version
```

## Python API

For the bundled financial sample, select the financial data profile:

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

To choose a generation method explicitly:

```python
from src.api import TeCoD
from src.models.data import GenerationRequest

with TeCoD(
    config_overrides=["+data=financial"],
) as tecod:
    request = GenerationRequest(
        query="How many accounts from north Bohemia are eligible to receive loans?",
        method="icl",
    )
    result = tecod.generate_with_method(request)
    print(result.pred_sql)
```

Supported methods are `auto`, `gcd`, `base-gcd`, `sgc`, `icl`, and `zs`. API-backed models support prompt-based methods such as `icl`, `sgc`, and `zs`; direct constrained decoding methods require local model access.

## Configuration

Configuration uses Hydra files in `conf/` plus Pydantic validation:

- `conf/config.yaml`: shared defaults and path settings.
- `conf/env/local.yaml`: local Hugging Face model profile.
- `conf/env/openai.yaml`: OpenAI-compatible API profile.
- `conf/data/financial.yaml`: bundled financial sample paths.

Default models:

- NLI reranker: `smitxxiv/Qwen3-Re4B-SQL-TeCoD-TMM`
- Embedding model: `Qwen/Qwen3-Embedding-4B`
- Local SQL generator profile: `XGenerationLab/XiYanSQL-QwenCoder-14B-2504`
- OpenAI-compatible profile: `gpt-4o-mini`

Local embedding, NLI, and local SQL-generation models are loaded with `local_files_only=True`. Download them into your Hugging Face cache before first use:

```bash
huggingface-cli download Qwen/Qwen3-Embedding-4B
huggingface-cli download smitxxiv/Qwen3-Re4B-SQL-TeCoD-TMM
huggingface-cli download XGenerationLab/XiYanSQL-QwenCoder-14B-2504
```

Command dependencies:

| Command | Writes | Model requirements |
| --- | --- | --- |
| `process-data --prepare-only` | `examples.jsonl`, `templates.jsonl`, `schema.prompt` | None |
| `create-index` | `index.db` | Local embedding model |
| `compile-templates` | `c_templs/*.pkl` | Local embedding model and local SQL model |
| `tecod --env openai` | None | Local embedding and NLI models, plus API credentials |
| `tecod --env local` | None | Local embedding, NLI, and SQL models |

Common overrides:

```bash
python main.py \
  -c +data=financial \
  -c tecod.model_id=XGenerationLab/XiYanSQL-QwenCoder-14B-2504 \
  tecod
```

Common environment variables include `TECOD_ROOT_DIR`, `TECOD_DATA_DIR`, `TECOD_DB_PATH`, `TECOD_DEVICE`, `TECOD_MODEL_ID`, `TECOD_PROVIDER`, `TECOD_NLI_MODEL`, `TECOD_EMB_MODEL`, `OPENAI_API_KEY`, and `OPENAI_BASE_URL`.

## Documentation

- [Quickstart](docs/quickstart.md)
- [Configuration](docs/configuration.md)
- [Data preparation](docs/data-preparation.md)
- [API reference](docs/api.md)
- [Generation methods](docs/generation-methods.md)
- [Architecture](docs/architecture.md)

## Examples

Runnable examples live in `examples/`:

- `basic_usage.py`: minimal Python API usage.
- `advanced_usage.py`: batching, timing, and method metadata.
- `integration_example.py`: wrapping TeCoD inside an application service.
- `openai_usage.py`: using an OpenAI-compatible generation backend.

The bundled example queries target the financial sample. If you use your own
database, replace them with natural-language questions that match your SQLite
schema and prepared `examples.jsonl`.

## Citation

If you use TeCoD in academic work, cite:

```bibtex
@article{10.1145/3769822,
  author = {Jivani, Smit and Maheshwari, Saravam and Sarawagi, Sunita},
  title = {Reliable Answers for Recurring Questions: Boosting Text-to-SQL Accuracy with Template Constrained Decoding},
  journal = {Proceedings of the ACM on Management of Data},
  volume = {3},
  number = {6},
  pages = {1--26},
  year = {2025},
  month = dec,
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  doi = {10.1145/3769822},
  url = {https://doi.org/10.1145/3769822}
}
```

## License

TeCoD is released under the MIT License. See [LICENSE](LICENSE).
