from . import consts
from .utils import DefaultDict, clean_multi_line_string


class Arctic:
    QUESTION_DEL = "\n\nQuestion:\n"

    ICL_DEL = "\n\nExamples:\n"
    ICL_DEL_LEN = len(ICL_DEL)

    SGC_INST = "\n\nSQL Template:\nFollow the template given below to generate the SQL query.\n{sql_template}\n"

    PROMPT_SUFFIX_DEL = "\nInstructions:\n"

    ICL_EXAMPLE_FORMAT = "Example #{i}\nQuestion: {question}\nSQL: {sql}\n"

    PROMPT = """<|im_start|>system
You are a SQL generation assistant.<|im_end|>
<|im_start|>user
Task Overview:
You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.

Database Engine:
{database_engine}

{database_schema}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

{sql_template}

{icl_examples}

Question:
{question}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.

Output Format:
In your answer, please enclose the generated SQL query in a code block:
```sql
-- Your SQL query
```

Take a deep breath and think step by step to find the correct SQL query.
<|im_end|>
<|im_start|>assistant
"""

    @staticmethod
    def prepare_icl_examples(icl_examples: list[tuple]):
        icl_examples_str = ""

        for i, (question, sql) in enumerate(icl_examples, 1):
            icl_examples_str += Arctic.ICL_EXAMPLE_FORMAT.format(i=i, question=question, sql=sql)

        return Arctic.ICL_DEL + icl_examples_str

    @staticmethod
    def prepare_dict(*args, **kwds):
        if "icl_examples" in kwds and len(kwds["icl_examples"]) > 0:
            kwds["icl_examples"] = Arctic.prepare_icl_examples(kwds["icl_examples"])
        else:
            kwds["icl_examples"] = ""
        if "sql_template" in kwds:
            template = kwds["sql_template"]
            for key, value in consts.EBNF_RULES_TO_REPLACE.items():
                template = template.replace(key, value)
            kwds["sql_template"] = Arctic.SGC_INST.format(sql_template=template)
        else:
            kwds["sql_template"] = ""
        if "database_engine" not in kwds:
            kwds["database_engine"] = "SQLite"
        return kwds

    @staticmethod
    def get_prompt(*args, **kwds):
        kwds = Arctic.prepare_dict(**kwds)
        default_dict = DefaultDict(kwds)
        prompt = Arctic.PROMPT.format_map(default_dict)
        return clean_multi_line_string(prompt)
