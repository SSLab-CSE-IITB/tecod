import re
from functools import lru_cache


@lru_cache(maxsize=1)
def _get_prompt_classes():
    """Lazy, cached map of prompt-class name → class. Avoids the circular
    import between this module and the prompt class modules (which import
    DefaultDict / clean_multi_line_string from here)."""
    from . import Arctic, CodeS, Default, Granite, Llama, QwenCoder

    return {
        "llama": Llama,
        "qwen": QwenCoder,
        "arctic": Arctic,
        "granite": Granite,
        "codes": CodeS,
        "default": Default,
    }


# TODO: Prompt placeholder validation (deferred)
# Two-level check to implement:
#
# 1. Class-level error (fires at import via __init_subclass__):
#    Add a PromptBase class that each prompt class (Llama, QwenCoder, etc.) inherits from.
#    In __init_subclass__, extract placeholders from cls.PROMPT using re.findall and verify
#    that _SYSTEM_KEYS = {"database_schema", "question", "sql_template"} are all present.
#    Raise TypeError if any are missing. This catches template authoring mistakes at load
#    time rather than during a generation run.
#    Caveats to resolve before expanding _SYSTEM_KEYS:
#      - {matched_content} is absent from QwenCoder.PROMPT and Arctic.PROMPT
#      - CodeS uses {icl_inst} instead of {icl_examples} (internal rename in prepare_dict)
#    Callers passing prompt_fn bypass this check entirely (escape hatch by design).
#
# 2. Call-level warning (fires at render via DefaultDict.__missing__):
#    Change __missing__ to return "" instead of "{key}" and log a warning via
#    logging.getLogger("app"). This prevents literal {key} strings from entering the
#    prompt when a template placeholder has no value.
#    IMPORTANT: do NOT scan the rendered output for {word} patterns — rendered SQL,
#    schema text, or question content can legitimately contain {word} patterns and would
#    produce false positives.


class DefaultDict(dict):
    """Dict subclass that returns '{key}' for missing keys in format_map.

    Safe from injection: str.format_map() only processes {placeholders}
    in the format string, never in substituted values.
    """

    def __missing__(self, key):
        return f"{{{key}}}"


re_multiplelines = re.compile(r"\n{3,}")


def clean_multi_line_string(input_string):
    """
    Cleans a multi-line string by removing leading and trailing white spaces and newlines.

    Args:
        input_string (str): The input string to clean.

    Returns:
        str: The cleaned string.

    Raises:
        None

    """

    return re_multiplelines.sub("\n\n", input_string)


def generate_prompt(
    *,
    model_id,
    prompt_class=None,
    prompt_fn=None,
    schema_sequence="",
    content_sequence="",
    question_text="",
    icl_examples=None,
    template=None,
    database_engine=None,
):
    if icl_examples is None:
        icl_examples = []

    # TODO: add database engine in the prompt. Check Arctic prompt

    classes = _get_prompt_classes()

    if prompt_class:
        prompt_class = prompt_class.strip().lower()

    model_id = model_id.lower()
    if prompt_fn:
        get_prompt = prompt_fn
    elif prompt_class:
        if prompt_class not in classes:
            raise ValueError(
                f"Unknown prompt_class {prompt_class!r}. Valid values: {sorted(classes)}"
            )
        get_prompt = classes[prompt_class].get_prompt
    elif "llama" in model_id:
        get_prompt = classes["llama"].get_prompt
    elif "qwen" in model_id:
        get_prompt = classes["qwen"].get_prompt
    elif "arctic" in model_id:
        get_prompt = classes["arctic"].get_prompt
    elif "granite" in model_id:
        get_prompt = classes["granite"].get_prompt
    elif "codes" in model_id:
        get_prompt = classes["codes"].get_prompt
    else:
        get_prompt = classes["default"].get_prompt

    kwds = dict()
    if template is not None:
        kwds["sql_template"] = template
    if icl_examples is not None and len(icl_examples) > 0:
        kwds["icl_examples"] = icl_examples

    prompt = get_prompt(
        database_schema=schema_sequence,
        matched_content=content_sequence,
        question=question_text,
        database_engine=database_engine if database_engine else "",
        **kwds,
    )

    return prompt
