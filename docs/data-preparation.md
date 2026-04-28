# Data Preparation

TeCoD expects a SQLite database plus JSONL state files under `data/` unless `data_dir` and `db_path` are overridden.

## Runtime Files

```text
data/
├── database.sqlite
├── examples.jsonl
├── templates.jsonl
├── schema.prompt
├── index.db
├── masked_questions.jsonl
└── c_templs/
    ├── 0.pkl
    └── ...
```

Important files:

| File | Purpose |
| --- | --- |
| `database.sqlite` | Database used for schema extraction and SQL grammar utilities. |
| `examples.jsonl` | Natural-language questions and SQL examples. |
| `templates.jsonl` | SQL templates grouped from examples. |
| `schema.prompt` | Serialized schema prompt used during generation. |
| `index.db` | Milvus Lite vector index. |
| `c_templs/` | Compiled templates used by partitioned constrained decoding. |

## Raw Data

`process-data` reads a JSON array. Every object must include:

- `text`: natural-language question.
- configured SQL field: defaults to `SQL`; override with `-c tecod.sql_key=<field>`.
- `nlq_masked`: retrieval/NLI skeleton for the question.

TeCoD does not generate `nlq_masked` automatically. Build it in your data preparation pipeline by replacing literal values and schema-specific spans with stable placeholders while preserving the question structure.

Example:

```json
[
  {
    "text": "What is the count of accounts opting for post-transaction issuance that are located in north Moravia?",
    "SQL": "SELECT COUNT(T2.account_id) FROM district AS T1 INNER JOIN account AS T2 ON T1.district_id = T2.district_id WHERE T1.A3 = 'north Moravia' AND T2.frequency = 'POPLATEK PO OBRATU'",
    "nlq_masked": "What is the count of accounts opting for post-transaction issuance that are located in _?"
  }
]
```

The repository includes a BIRD dev financial sample at `examples/financial/`.

```bash
python main.py \
  -c +data=financial \
  -c tecod.sql_key=SQL \
  -c tecod.model_id=XGenerationLab/XiYanSQL-QwenCoder-14B-2504 \
  process-data examples/financial/raw_data.json --prepare-only
```

Prepare-only mode normalizes SQL with `sqlglot`, writes `examples.jsonl`,
creates `templates.jsonl`, and writes `schema.prompt`. It does not load
embedding, NLI, or SQL-generation models.

To run the full pipeline in one command, omit `--prepare-only`. The full
pipeline also builds the vector index and, for local providers, compiles
templates. When `tecod.provider=openai`, template compilation is skipped because
API providers do not expose logits.

## Existing Prepared Data

If `examples.jsonl` and `templates.jsonl` already exist:

```bash
python main.py -c +data=financial create-index
python main.py -c +data=financial compile-templates
```

Use `status` to verify paths:

```bash
python main.py -c +data=financial status
```

Do not commit prepared datasets, indexes, compiled templates, or model files to this repository. The checked-in `examples/financial/financial.sqlite` file is the intentional sample database for onboarding.

The bundled sample paths live in `conf/data/financial.yaml`. Set
`TECOD_DATA_DIR` and `TECOD_DB_PATH` if you want to reuse the same profile with
another prepared data directory or SQLite database.

## Command Dependencies

| Command | Writes | Model requirements |
| --- | --- | --- |
| `process-data --prepare-only` | `examples.jsonl`, `templates.jsonl`, `schema.prompt` | None |
| `create-index` | `index.db` | Local embedding model |
| `compile-templates` | `c_templs/*.pkl` | Local embedding model and local SQL model |
| `tecod --env openai` | None | Local embedding and NLI models, plus API credentials |
| `tecod --env local` | None | Local embedding, NLI, and SQL models |
