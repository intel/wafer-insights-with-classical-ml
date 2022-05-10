
from connectors.database import get_connection, get_metadatadb_connection, query_data, insert_load_start, set_load_finish, create_history_table
from loaders.base_loader import utils
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

#######################################################################################################################
#################################################### SQ2L #############################################################
#######################################################################################################################

sql = """
SELECT /*+  ordered index(ett et_tp_str_uk) index(ets wesr_pk) */
          --(SELECT pl99.PROCESS_SECURED_BY FROM A_LOT_AT_OPERATION pl99 WHERE ets.lot = pl99.lot AND ets.operation = pl99.operation AND ets.facility = pl99.facility AND rownum <= 1) AS PROCESS
          SUBSTR(ets.PROCESS,2,4) as PROCESS
         ,ets.LOAD_END_DATE_TIME as LOAD_DATE
         ,ets.facility AS facility
         ,SUBSTR(ets.lot,1,7) AS LOT7
         ,ets.devrevstep AS DEVREVSTEP
         ,substr(ets.devrevstep,1,4) as SHORTDEVICE
         ,ets.wafer_id AS WAFER3
         ,ets.wafer_scribe AS WAFER
         ,ets.operation AS OPERATION
         ,ets.test_end_date_time AS TEST_END_DATE
         ,ets.program_name AS PROGRAM_NAME
         ,ett.test_name AS TEST_NAME
         ,ett.structure_name AS structure_name
         ,wer.median AS median
         ,wer.mean AS mean
         ,wer.stdev AS stdev
FROM
A_ETest_Testing_Session ets
INNER JOIN A_ETest_Test ett ON ets.program_name = ett.program_name
INNER JOIN A_Wafer_Etest_Setup_Rollup wer ON ets.lao_start_ww = wer.lao_start_ww AND ets.les_id = wer.les_id AND ets.wafer_id = wer.wafer_id AND ett.Test_Name = wer.Test_Name AND ett.Structure_Name = wer.Structure_Name
WHERE
              ets.valid_flag = 'Y'
AND  ets.LOAD_END_DATE_TIME > :START AND ets.LOAD_END_DATE_TIME <= :END AND ett.structure_name NOT LIKE '%PROBE%' AND ett.structure_name NOT LIKE '%RALPH%' AND ett.structure_name NOT LIKE '%PRBRES%' and ett.structure_name NOT LIKE '%LISA%'
AND SUBSTR(ets.PROCESS,2,4) NOT IN ('1276', '1278') 
AND (ets.devrevstep like '8PFU%' OR ets.devrevstep like '8PJS%' OR ets.devrevstep like '8PJR%')
"""

#######################################################################################################################
############################################### CONSTANTS #############################################################
#######################################################################################################################
storage_root = "C:/Users/eander2/PycharmProjects/WaferInsights/data/inline_etest"


def datetime_to_pathlike_string(dt):
    return dt.strftime("%Y%m%d-%H%M%S")

def get_chunks_from_metadatadb(connstring, datasource, table, max_delta_load=timedelta(days=65), incremental_load = timedelta(hours=12)):


    from datetime import timedelta, datetime
    # connstring = 'DRIVER={PostgreSQL Unicode(x64)};Port=5432;Database=test;UID=postgres;PWD=K1ll3rk1ng'
    last_load = get_last_load(connstring, datasource, table, max_delta_load)
    last_load = datetime.now() - timedelta(days=90)

    cload = last_load + inc
    loads = []
    while cload < datetime.now():
        loads.append((last_load, cload, datasource, table))
        last_load += inc
        cload += inc

    return loads

@utils.retry(Exception, tries=4)
def query_chunk(metadatadb_conconfig, datasource, loader_string_id, start, end):
    key = ['LOT7', 'WAFER3', 'TEST_END_DATE', 'OPERATION', 'PROGRAM_NAME']
    columns = 'TESTNAME`STRUCTURE_NAME'
    values = 'MEDIAN'

    with get_metadatadb_connection(metadatadb_conconfig) as mdb_conn:
        with get_connection({'datasource': datasource}) as db_conn:
            # (END_LOAD_DATE, START_LOAD_DATE, SOURCE, LOADER_STRING_ID)
            print([(end, start, datasource, loader_string_id)])
            insert_load_start(metadatadb_conconfig, [(end, start, datasource, loader_string_id)])
            print(f"extracting: {start}, {end}")
            data = query_data(db_conn, sql, params=(start, end))

            return data



def clean_data(data):
    data['TESTNAME`STRUCTURE_NAME'] = 'fcol`' + data['OPERATION'] + '`'  + data['TEST_NAME'] + '`' + data['STRUCTURE_NAME']
    # data['aggname_median'] = data['TESTNAME`STRUCTURE_NAME'] + '`MEDIAN'
    # data['aggname_mean'] = data['TESTNAME`STRUCTURE_NAME'] + '`MEAN'
    # index = ['PROCESS','SHORTDEVICE', 'LOT7', 'WAFER3', 'DEVREVSTEP', 'PROGRAM_NAME', 'LOAD_DATE']
    # pivot_data_median = pd.pivot_table(data, values = 'MEDIAN', columns='aggname_median', index=index)
    # pivot_data_mean = pd.pivot_table(data, values='MEAN', columns='aggname_mean', index=index)
    #
    # alldata = pivot_data_median.join(pivot_data_mean, on=index)
    return data



def store_chunk_observational_parquet(data_df, keys_columns, columns, value_column,  params):
    '''params is a dictionary.  many possible chunk storage start with 'storage_path', 'load_start', 'load_end',
     'partition_columns' for the resulting parquet'''

    pvdf = data_df.pivot_table(keys=keys_columns, columns=columns, values=value_column, aggfunc=np.median)

    load_start = datetime_to_pathlike_string(params['load_start'])
    load_end = datetime_to_pathlike_string(params['load_end'])
    fname = params['storage_path'] + f"/{load_start}`{load_end}.parquet"

    if 'partition_columns' in params:
        partition_columns = params['partition_columns']
        data_df.to_parquet(fname, partition_cols=partition_columns)
    else:
        data_df.to_parquet(fname)

def store_raw_file(data_df, params):
    load_start = datetime_to_pathlike_string(params['load_start'])
    load_end = datetime_to_pathlike_string(params['load_end'])
    #fname = params['storage_path'] + f"/{load_start}--{load_end}.parquet"
    fname = params['storage_path'] + f"/LOAD_END={load_end}"
    data_df = data_df.reset_index()

    data_df.to_parquet(fname, partition_cols=['PROCESS', 'SHORTDEVICE','OPERATION', 'LOT7'])

def update_cache(backload = timedelta(days=60)):
    from connectors.database import drop_load_history_table, get_last_load

    mdb_connstring = "DRIVER={PostgreSQL Unicode(x64)};Port=5432;Database=test;UID=postgres;PWD=K1ll3rk1ng"

    last_load = get_last_load(mdb_connstring, "F32_PROD_XEUS", "INLINE_ETEST", backload)
    print(f"last_load: {last_load}")
    #last_load = datetime.now() - timedelta(days=180)

    dt = timedelta(days=1)

    next_load = last_load + dt

    while next_load < datetime.now():
        data = query_chunk(mdb_connstring, "F32_PROD_XEUS", "INLINE_ETEST", start=last_load,
                           end=next_load)
        cleaned = clean_data(data)

        store_raw_file(cleaned, params={'load_start': last_load, 'load_end': next_load, 'storage_path': storage_root})

        real_load_date = data['LOAD_DATE'].max().to_pydatetime()
        set_load_finish(mdb_connstring, [(real_load_date, next_load, last_load, "INLINE_ETEST", "F32_PROD_XEUS")])
        last_load += dt
        next_load += dt



if __name__=="__main__":
    import PyUber
    from connectors.database import drop_load_history_table, drop_load_history_by_string_id


    mdb_connstring = "DRIVER={PostgreSQL Unicode(x64)};Port=5432;Database=test;UID=postgres;PWD=K1ll3rk1ng"
    #drop_load_history_table(mdb_connstring)
    drop_load_history_by_string_id(mdb_connstring, "INLINE_ETEST")
    update_cache()

