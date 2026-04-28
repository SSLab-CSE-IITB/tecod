# Configuration

TeCoD uses Hydra for layered YAML configuration and Pydantic for runtime validation.

## Files

| File | Purpose |
| --- | --- |
| `conf/config.yaml` | Shared defaults for paths, state files, models, retrieval, generation, and logging. |
| `conf/env/local.yaml` | Local Hugging Face model profile. |
| `conf/env/openai.yaml` | OpenAI-compatible API profile. |
| `conf/data/financial.yaml` | Bundled financial sample data and database paths. |

The default environment is `local`.

## Precedence

Highest precedence wins:

```text
CLI -c overrides
Environment profile selected by --env or TECOD_ENV
conf/config.yaml
Pydantic defaults
```

Examples:

```bash
python main.py --env local -c data_dir=data tecod

python main.py \
  --env openai \
  -c +data=financial \
  -c tecod.model_id=gpt-4o-mini \
  tecod
```

Use `+data=financial` to select the bundled sample paths. `TECOD_DATA_DIR` and
`TECOD_DB_PATH` can still override those paths without editing YAML.

## Common Keys

| Key | Default | Purpose |
| --- | --- | --- |
| `root_dir` | `${oc.env:TECOD_ROOT_DIR,.}` | Base project path. |
| `data_dir` | `${root_dir}/data` | Runtime data directory. |
| `db_path` | `${root_dir}/data/database.sqlite` | SQLite database used for schema extraction and grammar checks. |
| `device` | `auto` | Default device for local components. |
| `tecod.provider` | `local` | `local` or `openai`. |
| `tecod.model_id` | profile-dependent | SQL generation model. |
| `nli.model` | `smitxxiv/Qwen3-Re4B-SQL-TeCoD-TMM` | NLI reranker used for template selection. |
| `emb.model` | `Qwen/Qwen3-Embedding-4B` | Embedding model used for retrieval. |
| `tecod.icl_cnt` | `3` | Number of in-context examples. |
| `tecod.vectorsearch_top_k` | `1000` | Retrieval depth before NLI reranking. |
| `tecod.nli_top_k` | `30` | Number of candidates scored by NLI. |
| `tecod.sql_key` | `SQL` | Field containing SQL in examples. |
| `tecod.dialect` | `sqlite` | SQL dialect passed to `sqlglot`. |

## Environment Variables

Common variables:

| Variable | Maps to |
| --- | --- |
| `TECOD_ROOT_DIR` | `root_dir` |
| `TECOD_DATA_DIR` | `data_dir` |
| `TECOD_DB_PATH` | `db_path` |
| `TECOD_DEVICE` | `device` |
| `TECOD_MODEL_ID` | `tecod.model_id` |
| `TECOD_PROVIDER` | `tecod.provider` |
| `TECOD_NLI_MODEL` | `nli.model` |
| `TECOD_NLI_DEVICE` | `nli.device` |
| `TECOD_EMB_MODEL` | `emb.model` |
| `TECOD_EMB_DEVICE` | `emb.device` |
| `TECOD_PROMPT_CLASS` | `tecod.prompt_class` |
| `TECOD_LOG_LEVEL` | `logging.console_level` |
| `OPENAI_API_KEY` | `tecod.api_key` |
| `OPENAI_BASE_URL` | `tecod.base_url` |

`-c` overrides take precedence over environment variables.

## Model Roles

TeCoD uses three model roles:

- The NLI model, `smitxxiv/Qwen3-Re4B-SQL-TeCoD-TMM` by default, reranks retrieved candidates and decides whether a query entails a stored SQL template.
- The embedding model, `Qwen/Qwen3-Embedding-4B` by default, builds query/example vectors for first-stage retrieval.
- The SQL generation model is selected by `tecod.model_id`. The local profile defaults to `XGenerationLab/XiYanSQL-QwenCoder-14B-2504`; the OpenAI-compatible profile defaults to `gpt-4o-mini`.
