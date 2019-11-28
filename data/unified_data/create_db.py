import sqlite3
from sqlite3 import Error
 

def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)

def create_connection(db_file):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
    except Error as e:
        print(e)
    return conn
 
def main():
    db_path =  r"./data/unified_data/pythonsqlite.db"
    conn = create_connection(db_path)

    sql_create_kb_clauses = """CREATE TABLE IF NOT EXISTS kb_clauses (
        id integer PRIMARY KEY,
        raw_id integer,
        clause_ind integer,
        context_string varchar,
        raw_string varchar,
        processed_string varchar,
        created_at timestamp
    );"""

    sql_create_labeled_queries = """CREATE TABLE IF NOT EXISTS labeled_queries (
        id integer PRIMARY KEY,
        query_string varchar NOT NULL,
        clause_id integer,
        span_start integer,
        span_end integer,
        created_at timestamp
    );"""

    sql_create_kb_raw = """CREATE TABLE IF NOT EXISTS kb_raw (
        id integer PRIMARY KEY,
        filepath varchar,
        kb_name varchar NOT NULL,
        type varchar,
        directory_id int
    );"""    

    sql_create_kb_directory = """CREATE TABLE IF NOT EXISTS kb_directory (
        id integer PRIMARY KEY,
        created_at timestamp,
        dir_name varchar NOT NULL,
        user_id int
    );"""

    sql_create_query_log = """CREATE TABLE IF NOT EXISTS query_log (
        id integer PRIMARY KEY,
        created_at timestamp,
        query_string varchar,
        kb_dir_id int
    );"""

    sql_create_users = """CREATE TABLE IF NOT EXISTS users (
        id integer PRIMARY KEY,
        created_at timestamp,
        email varchar NOT NULL UNIQUE,
        full_name varchar,
        org_name varchar,
        hashkey varchar
    );"""

    if conn:
        # create tables
        create_table(conn, sql_create_kb_clauses)
        create_table(conn, sql_create_labeled_queries)
        create_table(conn, sql_create_kb_raw)
        create_table(conn, sql_create_kb_directory)
        create_table(conn, sql_create_query_log)
        create_table(conn, sql_create_users)
        conn.close()

    else:
        print('Error! cannot create db connection')

if __name__ == '__main__':
    main()