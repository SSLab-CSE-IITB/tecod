# API Reference

The public Python entrypoint is `src.api.TeCoD`.

## TeCoD

```python
TeCoD(
    data_dir: str = "data",
    device: str | None = None,
    config_overrides: list[str] | None = None,
    log_level: str = "INFO",
    console_log_level: str = "ERROR",
    file_log_level: str = "DEBUG",
    log_file: str | None = "tecod_api.log",
)
```

Example:

```python
from src.api import TeCoD

with TeCoD(
    device="auto",
    config_overrides=["+data=financial"],
) as tecod:
    result = tecod.generate(
        "What is the count of accounts opting for post-transaction issuance that are located in north Moravia?"
    )
    print(result.pred_sql)
```

## generate

```python
result = tecod.generate(
    query="How many accounts from north Bohemia are eligible to receive loans?",
    max_new_tokens=4096,
    num_beams=1,
    do_sample=False,
    top_k=1000,
    regex_grammar=None,
)
```

Returns a `GenerationOutput` with the generated SQL, selected method, template metadata, retrieval/NLI scores, and timing fields.

## generate_with_method

Use `GenerationRequest` to choose a method explicitly:

```python
from src.models.data import GenerationRequest

request = GenerationRequest(
    query="How many accounts from north Bohemia are eligible to receive loans?",
    method="icl",
)
result = tecod.generate_with_method(request)
```

Supported method values are `auto`, `gcd`, `base-gcd`, `sgc`, `icl`, and `zs`.

## GenerationRequest Fields

| Field | Default | Meaning |
| --- | --- | --- |
| `query` | required | Natural-language request. |
| `top_k` | `1000` | Retrieval depth. |
| `max_new_tokens` | `4096` | Generation token cap. |
| `num_beams` | `1` | Beam search width for local models. |
| `do_sample` | `False` | Sampling flag. |
| `regex_grammar` | `None` | Optional regex constraint. |
| `method` | `auto` | Generation method. |
| `schema_sequence` | `None` | Optional schema override. |
| `content_sequence` | `None` | Optional content override. |
| `zs_prompt` | `None` | Optional zero-shot prompt. |
| `use_oracle` | `False` | Use oracle template selection when supported. |
| `gold_sql` | `None` | Gold SQL for oracle workflows. |

## GenerationOutput Fields

Common fields:

| Field | Meaning |
| --- | --- |
| `query` | Original natural-language query. |
| `pred_sql` | Generated SQL string. |
| `method` | Method used for generation. |
| `template_id` | Selected template id. |
| `nli_score` | Entailment score for selected template/example. |
| `cosine_score` | Vector similarity score. |
| `nli_label` | NLI label. |
| `icl_examples` | In-context examples used in the prompt. |
| `total_time` | End-to-end generation time in seconds. |
| `timing_data` | Detailed timing dictionary. |
| `prompt` | Prompt text when retained for debugging. |

## Convenience Constructor

```python
from src.api import create_tecod

tecod = create_tecod(
    config_overrides=["+data=financial"],
)
```
