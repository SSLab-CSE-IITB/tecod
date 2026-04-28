from . import consts
from .utils import DefaultDict, clean_multi_line_string


class Default:
    ICL_INST = "Here are some examples of user questions and their corresponding SQL queries:\n"

    SQL_TEMPLATE_INST = "Here is the SQL template for the question. Follow this template to generate the SQL query.\n"

    ICL_EXAMPLE_FORMAT = """Question: {question}\nAnswer: {sql}\n"""

    PROMPT = """Write SQL query to answer the user question using the given database schema. Do not print any additional information, formatting, explanation, or notes.

{database_schema}
{matched_content}

{sql_template}

{icl_examples}

Question: {question}

Output Format:
In your answer, please enclose the generated SQL query in a code block:
```sql
-- Your SQL query
```

Answer: """

    @staticmethod
    def prepare_icl_examples(icl_examples: list[tuple]):
        icl_examples_str = Default.ICL_INST

        for question, sql in icl_examples:
            icl_examples_str += Default.ICL_EXAMPLE_FORMAT.format(question=question, sql=sql)
            icl_examples_str += "\n"

        return icl_examples_str

    @staticmethod
    def prepare_dict(*args, **kwds):
        if "icl_examples" in kwds and len(kwds["icl_examples"]) > 0:
            kwds["icl_examples"] = Default.prepare_icl_examples(kwds["icl_examples"])
        else:
            kwds["icl_examples"] = ""
        if "sql_template" in kwds:
            template = kwds["sql_template"]
            for key, value in consts.EBNF_RULES_TO_REPLACE.items():
                template = template.replace(key, value)
            kwds["sql_template"] = Default.SQL_TEMPLATE_INST + template
        else:
            kwds["sql_template"] = ""
        return kwds

    @staticmethod
    def get_prompt(*args, **kwds):
        kwds = Default.prepare_dict(**kwds)
        default_dict = DefaultDict(kwds)
        prompt = Default.PROMPT.format_map(default_dict)
        return clean_multi_line_string(prompt)
