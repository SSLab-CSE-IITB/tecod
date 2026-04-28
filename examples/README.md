# Examples

These examples exercise the public TeCoD API and CLI-facing configuration paths.

They require a prepared TeCoD data directory and vector index. By default they
target the bundled financial sample at `data/financial` with
`examples/financial/financial.sqlite` through the `+data=financial` profile.

```bash
python main.py \
  -c +data=financial \
  process-data examples/financial/raw_data.json --prepare-only

python main.py -c +data=financial create-index
python examples/basic_usage.py
```

The profile is defined in `conf/data/financial.yaml`. Set `TECOD_DATA_DIR` and
`TECOD_DB_PATH` to point it at another prepared dataset without editing YAML:

```bash
export TECOD_DATA_DIR=/path/to/prepared/tecod-data
export TECOD_DB_PATH=/path/to/database.sqlite
python examples/basic_usage.py
```

The bundled natural-language queries match the financial sample. Replace them
with questions that match your SQLite database and prepared `examples.jsonl` if
you use another dataset.

Verified example properties:

- All scripts import the current `src.api.TeCoD` API.
- All scripts use the bundled `+data=financial` profile.
- The OpenAI example selects both `env@_global_=openai` and `+data=financial`.
- OpenAI-compatible generation still requires local embeddings, local NLI, and a prepared vector index.
- The integration example cleans up its temporary input/output files.
