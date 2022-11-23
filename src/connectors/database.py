import os
import sys
import traceback
from datetime import timedelta

import numpy as np
import pandas as pd
import pyodbc

if os.name == 'nt':
    import PyUber

    use_db2 = False
else:
    use_db2 = True


def getSchema(curs):
    schema = dict()
    for col in curs.description:
        if col[1] == PyUber.STRING:
            schema[col[0]] = "str"
        elif col[1] == PyUber.NUMBER_FLOAT:
            schema[col[0]] = "float32"
        elif col[1] == PyUber.NUMBER_INTEGER:
            schema[col[0]] = "int64"
        elif col[1] == PyUber.DATETIME:
            schema[col[0]] = "datetime64[ns]"
        elif col[1] == PyUber.NUMBER:
            schema[col[0]] = "float32"
        else:
            schema[col[0]] = "object"
    return schema


def getSchemaODBC(curs):
    schema = dict()
    for col in curs.description:
        print(col[1])
    for col in curs.description:
        if col[1] == type(str):
            schema[col[0]] = "str"
        elif col[1] == type(float):
            schema[col[0]] = "float32"
        elif col[1] == type(int):
            schema[col[0]] = "int64"
        elif col[1] == type(dt.datetime):
            schema[col[0]] = "datetime64[ns]"
        else:
            schema[col[0]] = "object"
    return schema


def standardizeSchema(schema, df):
    for col in df.columns:
        stype = schema[col]
        if stype == 'float32':
            df[col] = pd.to_numeric(df[col], errors='coerce', downcast='float').astype(np.float32)
        elif stype == 'str':
            df[col] = df[col].astype(str)
        elif stype == 'datetime64[ns]' and df[col].dtype is not object:
            df[col] = pd.to_datetime(df[col],
                                     errors='coerce')
        elif stype == 'datetime64[ns]' and df[col].dtype is object:
            df[col] = pd.to_datetime(df[col],
                                     format='%Y-%m-%d %H:%M:%S',
                                     errors='coerce')
        elif stype == 'int64':
            df[col] = pd.to_numeric(df[col], errors='coerce', downcast="float").astype(np.float64)
        else:
            df[col] = df[col].astype(object)

    return df


def get_metadatadb_connection(conn_config):
    try:
        conn = pyodbc.connect(conn_config)
    except:
        raise Exception("Cannot connect to metadata database")
    return conn


def get_connection(conn_config, db_config=None, secrets=None):
    try:
        if use_db2:

            usr = self.user_config['USER']
            pwd = self.user_config['PASSWORD']

            connstring = self.db_config[datasource]
            connstring = connstring + f";UID={usr};PWD={pwd}"
            conn = None

            try:
                conn = pyodbc.connect(connstring)
            except pyodbc.Error as err:
                print(err)

            return pyodbc.connect(connstring)

        else:
            conn = PyUber.connect(**conn_config)
    except:
        raise Exception("Cannot connect to database with config: {conn_config}")

    return conn


def create_history_table(conn):
    conn = get_metadatadb_connection(conn)
    sql = '''CREATE TABLE IF NOT EXISTS LOAD_HISTORY (
                END_LOAD_DATE timestamp ,
                START_LOAD_DATE timestamp,
                REAL_LOAD_DATE timestamp,
                LOADER_STRING_ID text,
                SOURCE text,
                IS_LOADED integer DEFAULT 0,
                PRIMARY KEY (LOADER_STRING_ID, SOURCE, END_LOAD_DATE));'''
    try:
        curs = conn.cursor()
        curs.execute(sql)
        conn.commit()
        del conn
    except Exception:
        print("Exception in user code:")
        print("-" * 60)
        traceback.print_exc(file=sys.stdout)
        print("-" * 60)


def get_last_load(conn_config, source, loader_string_id, backload_timedelta):
    default_start_time = datetime.now() - backload_timedelta

    # last load will never load through PyUber.... will use raw connector

    conn = get_metadatadb_connection(conn_config)

    create_history_table(conn_config)

    data = pd.read_sql("select REAL_LOAD_DATE FROM LOAD_HISTORY WHERE SOURCE=? AND LOADER_STRING_ID=? AND IS_LOADED=1",
                       conn,
                       params=(source, loader_string_id))
    del conn
    if data.shape[0] == 0:
        return default_start_time

    try:
        real_load_date = data["real_load_date"].max()
        return real_load_date
    except:
        raise Exception(fr"Failed to find most recent load date for {loader_string_id}, {source}")


def insert_load_start(conn_config, param_tuple_list):
    '''Param tuple should be (END_LOAD_DATE, START_LOAD_DATE, SOURCE, LOADER_STRING_ID)'''
    # last load will never load through PyUber.... will use raw connector

    if type(conn_config) is str:
        # could be a c connection string
        conn = pyodbc.connect(conn_config)
    else:
        # else it is a connection
        conn = conn_config

    sqlite_insert_with_param = """INSERT INTO LOAD_HISTORY
                              (END_LOAD_DATE, START_LOAD_DATE, SOURCE, LOADER_STRING_ID) 
                              VALUES (?, ?, ?, ?);"""

    cursor = conn.cursor()
    try:
        for param_tuple in param_tuple_list:
            cursor.execute(sqlite_insert_with_param, param_tuple)
        conn.commit()
        del conn
    except:
        del conn
        raise Exception("Failed to save load to metadata database.")


def set_load_finish(conn_config, param_tuple):
    '''param tuple (REAL_LOAD_DATE, END_LOAD_DATE, START_LOAD_DATE, LOADER_STRING_ID, SOURCE)'''
    # last load will never load through PyUber.... will use raw connector
    conn = pyodbc.connect(conn_config)

    sql_update_with_param = """UPDATE LOAD_HISTORY SET REAL_LOAD_DATE = ?, IS_LOADED=1 where END_LOAD_DATE=? AND START_LOAD_DATE=? AND LOADER_STRING_ID=? AND SOURCE=?"""
    cursor = conn.cursor()
    try:
        cursor.execute(sql_update_with_param, *param_tuple)
        conn.commit()
        del conn
    except:
        del conn
        raise Exception("Failed to update LOAD_HISTORY table with new data")


def drop_load_history_by_string_id(conn_config, loader_string_id, data_source=None):
    conn = pyodbc.connect(conn_config)
    cursor = conn.cursor()

    sql = f"DELETE FROM LOAD_HISTORY WHERE LOADER_STRING_ID='{loader_string_id}'"

    if data_source is not None:
        sql += f" and SOURCE='{data_source}'"

    cursor.execute(sql)
    conn.commit()
    del conn


def drop_load_history_table(conn_config):
    conn = pyodbc.connect(conn_config)
    cursor = conn.cursor()

    sql = "DROP TABLE IF EXISTS LOAD_HISTORY;"
    cursor.execute(sql)
    conn.commit()
    del conn


def query_data(conn, sql_text, params):
    '''Params should have the following form: (start_datetime, end_datetime)'''
    # conn = get_connection(conn_config)

    curs = conn.cursor()

    # need to handle the different OS configs
    if os.name == 'nt':
        '''PyUber wants a set of named params.  Need to unwrap the dict'''
        curs.execute(sql_text, START=params[0], END=params[1])
    else:
        # pyodbc wants a parameter tuple with ? for parameters in the sql.  In most scripts
        # the only params are :START, :END for the extract window
        sql_text = sql_text.replace(":START", "?").replace(":END", "?")
        curs.execute(sql_text, params)

    columns = [x[0] for x in curs.description]
    df = pd.DataFrame().from_records(curs.fetchall(), columns=columns)

    if os.name == 'nt':
        schema = getSchema(curs)
    else:
        schema = getSchemaODBC(curs)

    df = standardizeSchema(schema, df)

    return df


if __name__ == "__main__":

    last_load = get_last_load(connstring, "F32_PROD_XEUS", "INLINE_ETEST", timedelta(days=65))

    from datetime import timedelta, datetime

    connstring = 'DRIVER={PostgreSQL Unicode(x64)};Port=5432;Database=test;UID=postgres;PWD=K1ll3rk1ng'

    last_load = get_last_load(connstring, "F32_PROD_XEUS", "INLINE_ETEST", timedelta(days=65))
    inc = timedelta(hours=12)
    cload = last_load + inc
    loads = []
    while cload < datetime.now():
        loads.append((last_load, cload, "F32_PROD_XEUS", "INLINE_ETEST"))
        last_load += inc
        cload += inc

    insert_load_start(connstring, loads)

    drop_load_history_table(connstring)
