# source: https://github.com/myamma/CodeS/blob/main/utils/db_utils.py

import logging
import os
import random
import sqlite3

logger = logging.getLogger(__name__)


def _quote_ident(name: str) -> str:
    """Return a backtick-quoted SQLite identifier, rejecting anything unsafe.

    Backticks inside the name are doubled (the standard escape); NUL bytes
    and semicolons are refused outright, since no legitimate identifier
    contains them and either would risk breaking out of the quoted form.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    if "\x00" in name or ";" in name:
        raise ValueError(f"Unsafe SQL identifier: {name!r}")
    return "`" + name.replace("`", "``") + "`"


# get the database cursor for a sqlite database path
def get_cursor_from_path(sqlite_path):
    try:
        if not os.path.exists(sqlite_path):
            logger.info("Opening new connection: %s", sqlite_path)
        connection = sqlite3.connect(sqlite_path, check_same_thread=False)
    except Exception as e:
        logger.error("Failed to connect: %s", sqlite_path)
        raise e
    connection.text_factory = lambda b: b.decode(errors="ignore")
    cursor = connection.cursor()
    return connection, cursor


def execute_sql(cursor, sql):
    cursor.execute(sql)

    return cursor.fetchall()


def check_sql_executability(generated_sql, db):
    if not generated_sql.strip():
        return "Error: empty string"
    connection = None
    try:
        connection, cursor = get_cursor_from_path(db)
        execute_sql(cursor, generated_sql)
        execution_error = None
    except Exception as e:
        logger.warning("SQL execution error: %s", e)
        execution_error = str(e)
    finally:
        if connection is not None:
            connection.close()

    return execution_error


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def detect_special_char(name):
    for special_char in ["(", "-", ")", " ", "/"]:
        if special_char in name:
            return True

    return False


def add_quotation_mark(s):
    return "`" + s + "`"


def get_column_contents(column_name, table_name, cursor, column_content_limit=2):
    col = _quote_ident(column_name)
    tbl = _quote_ident(table_name)
    select_column_sql = f"SELECT DISTINCT {col} FROM {tbl} WHERE {col} IS NOT NULL;"
    results = execute_sql(cursor, select_column_sql)
    column_contents = [str(result[0]).strip() for result in results]
    random.shuffle(column_contents)
    # remove empty and extremely-long contents
    column_contents = [
        content for content in column_contents if len(content) != 0 and len(content) <= 25
    ]

    return column_contents[: min(len(column_contents), column_content_limit)]


def get_db_schema_sequence(schema):
    # schema_sequence = "database schema :\n"
    schema_sequence = ""
    for table in schema["schema_items"]:
        table_name, table_comment = table["table_name"], table["table_comment"]
        if detect_special_char(table_name):
            table_name = add_quotation_mark(table_name)

        if table_comment != "":
            table_name += " ( comment : " + table_comment + " )"

        column_info_list = []
        for (
            column_name,
            column_type,
            column_comment,
            column_content,
            pk_indicator,
        ) in zip(
            table["column_names"],
            table["column_types"],
            table["column_comments"],
            table["column_contents"],
            table["pk_indicators"],
        ):
            if detect_special_char(column_name):
                column_name = add_quotation_mark(column_name)
            additional_column_info = []
            # column type
            additional_column_info.append(column_type)
            # pk indicator
            if pk_indicator != 0:
                additional_column_info.append("primary key")
            # column comment
            if column_comment != "":
                additional_column_info.append("comment : " + column_comment)
            # column content
            if len(column_content) != 0:
                additional_column_info.append("values : " + " , ".join(column_content))

            column_info_list.append(
                table_name + "." + column_name + " ( " + " | ".join(additional_column_info) + " )"
            )

        schema_sequence += (
            "table " + table_name + " , columns = [ " + " , ".join(column_info_list) + " ]\n"
        )

    if len(schema["foreign_keys"]) != 0:
        schema_sequence += "foreign keys :\n"
        for foreign_key in schema["foreign_keys"]:
            # Build a local quoted copy; mutating foreign_key in place would
            # corrupt the caller's schema dict and double-quote on re-entry.
            quoted = [
                add_quotation_mark(part) if detect_special_char(part) else part
                for part in foreign_key
            ]
            schema_sequence += f"{quoted[0]}.{quoted[1]} = {quoted[2]}.{quoted[3]}\n"
    else:
        schema_sequence += "foreign keys : None\n"

    return schema_sequence.strip()


def get_matched_content_sequence(matched_contents):
    content_sequence = ""
    if len(matched_contents) != 0:
        content_sequence += "matched contents :\n"
        for tc_name, contents in matched_contents.items():
            table_name = tc_name.split(".")[0]
            column_name = tc_name.split(".")[1]
            if detect_special_char(table_name):
                table_name = add_quotation_mark(table_name)
            if detect_special_char(column_name):
                column_name = add_quotation_mark(column_name)

            content_sequence += (
                table_name + "." + column_name + " ( " + " , ".join(contents) + " )\n"
            )
    else:
        content_sequence = "matched contents : None"

    return content_sequence.strip()


def get_db_schema(db_path, db_comments, db_id, column_content_limit=2):
    if db_id in db_comments:
        db_comment = db_comments[db_id]
    else:
        db_comment = None

    connection, cursor = get_cursor_from_path(db_path)

    try:
        # obtain table names
        results = execute_sql(cursor, "SELECT name FROM sqlite_master WHERE type='table';")
        table_names = [result[0].lower() for result in results]

        schema = dict()
        schema["schema_items"] = []
        foreign_keys = []
        # for each table
        for table_name in table_names:
            # skip SQLite system table: sqlite_sequence
            if table_name == "sqlite_sequence":
                continue
            # obtain column names in the current table. The PRAGMA
            # table-valued function accepts a bind parameter, which removes
            # any identifier-injection risk from the table name.
            cursor.execute("SELECT name, type, pk FROM pragma_table_info(?)", (table_name,))
            results = cursor.fetchall()
            column_names_in_one_table = [result[0].lower() for result in results]
            column_types_in_one_table = [result[1].lower() for result in results]
            pk_indicators_in_one_table = [result[2] for result in results]

            column_contents = []
            for column_name in column_names_in_one_table:
                column_contents.append(
                    get_column_contents(
                        column_name,
                        table_name,
                        cursor,
                        column_content_limit=column_content_limit,
                    )
                )

            # obtain foreign keys in the current table (parameterised)
            cursor.execute("SELECT * FROM pragma_foreign_key_list(?);", (table_name,))
            results = cursor.fetchall()
            for result in results:
                if None not in [result[3], result[2], result[4]]:
                    foreign_keys.append(
                        [
                            table_name.lower(),
                            result[3].lower(),
                            result[2].lower(),
                            result[4].lower(),
                        ]
                    )

            # obtain comments for each schema item
            if db_comment is not None:
                if table_name in db_comment:  # record comments for tables and columns
                    table_comment = db_comment[table_name]["table_comment"]
                    column_comments = [
                        (
                            db_comment[table_name]["column_comments"][column_name]
                            if column_name in db_comment[table_name]["column_comments"]
                            else ""
                        )
                        for column_name in column_names_in_one_table
                    ]
                else:  # current database has comment information, but the current table does not
                    table_comment = ""
                    column_comments = ["" for _ in column_names_in_one_table]
            else:  # current database has no comment information
                table_comment = ""
                column_comments = ["" for _ in column_names_in_one_table]

            schema["schema_items"].append(
                {
                    "table_name": table_name,
                    "table_comment": table_comment,
                    "column_names": column_names_in_one_table,
                    "column_types": column_types_in_one_table,
                    "column_comments": column_comments,
                    "column_contents": column_contents,
                    "pk_indicators": pk_indicators_in_one_table,
                }
            )

        schema["foreign_keys"] = foreign_keys

        return schema
    finally:
        connection.close()
