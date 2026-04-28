from . import consts
from .utils import DefaultDict, clean_multi_line_string


class CodeS:
    ICL_INST = "\nHere are some examples of user questions and their corresponding SQL queries:\n"

    SQL_TEMPLATE_INST = "\nHere is the SQL template for the question. Follow this template to generate the SQL query.\n"

    ICL_EXAMPLE_FORMAT = """Question: {question}\nAnswer: {sql}\n"""

    PROMPT = """{database_schema}
{matched_content}
{sql_template}
{icl_inst}
{question}
"""

    @staticmethod
    def prepare_icl_examples(icl_examples: list[tuple]):
        icl_examples_str = CodeS.ICL_INST

        for question, sql in icl_examples:
            icl_examples_str += CodeS.ICL_EXAMPLE_FORMAT.format(question=question, sql=sql)
            icl_examples_str += "\n"

        return icl_examples_str

    @staticmethod
    def prepare_dict(*args, **kwds):
        if "icl_examples" in kwds and len(kwds["icl_examples"]) > 0:
            kwds["icl_inst"] = CodeS.prepare_icl_examples(kwds["icl_examples"]) + "\n"
        else:
            kwds["icl_inst"] = ""
        if "sql_template" in kwds:
            template = kwds["sql_template"]
            for key, value in consts.EBNF_RULES_TO_REPLACE.items():
                template = template.replace(key, value)
            kwds["sql_template"] = CodeS.SQL_TEMPLATE_INST + template + "\n\n"
        else:
            kwds["sql_template"] = ""
        return kwds

    @staticmethod
    def get_prompt(*args, **kwds):
        kwds = CodeS.prepare_dict(**kwds)
        default_dict = DefaultDict(kwds)
        prompt = CodeS.PROMPT.format_map(default_dict).strip() + "\n"
        return clean_multi_line_string(prompt)
