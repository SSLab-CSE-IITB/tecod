from . import consts
from .utils import DefaultDict, clean_multi_line_string


class QwenCoder:
    ICL_DEL = "\n\n# 【Examples】\n"
    ICL_DEL_LEN = len(ICL_DEL)

    SGC_INST = "\n\n# 【SQL Template】\n# Follow the template given below to generate the SQL query.\n# {sql_template}\n"

    PROMPT = """<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
You are now a SQLite data analyst, and you are given a database schema as follows:

{database_schema}

{sql_template}

{icl_examples}

# 【Question】
# {question}

# Please read and understand the database schema carefully, and generate an executable SQL based on the user's question and evidence. The generated SQL is protected by ```sql and ```.<|im_end|>
<|im_start|>assistant
"""

    ICL_EXAMPLE_FORMAT = """Example #{i}\n  Question: {question}\n  SQL: {sql}\n"""

    @staticmethod
    def prepare_icl_examples(icl_examples: list[tuple]):
        icl_examples_str = ""

        for i, (question, sql) in enumerate(icl_examples, 1):
            icl_examples_str += QwenCoder.ICL_EXAMPLE_FORMAT.format(i=i, question=question, sql=sql)

        return icl_examples_str

    @staticmethod
    def prepare_dict(*args, **kwds):
        if "icl_examples" in kwds and len(kwds["icl_examples"]) > 0:
            kwds["icl_examples"] = QwenCoder.ICL_DEL + QwenCoder.prepare_icl_examples(
                kwds["icl_examples"]
            )
        else:
            kwds["icl_examples"] = ""
        if "sql_template" in kwds:
            template = kwds["sql_template"]
            for key, value in consts.EBNF_RULES_TO_REPLACE.items():
                template = template.replace(key, value)
            kwds["sql_template"] = QwenCoder.SGC_INST.format(sql_template=template)
        else:
            kwds["sql_template"] = ""
        return kwds

    @staticmethod
    def get_prompt(*args, **kwds):
        kwds = QwenCoder.prepare_dict(**kwds)
        default_dict = DefaultDict(kwds)
        prompt = QwenCoder.PROMPT.format_map(default_dict)
        return clean_multi_line_string(prompt)
