
from connectors.database import get_connection,
from datetime import datetime, timedelta
import numpy as np
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
"""

def datetime_to_pathlike_string(dt):
    return dt.strftime("%Y%m%d-%H%M%S")

def get_chunks_from_metadatadb(connstring, datasource, table, max_delta_load=timedelta(days=65), incremental_load = timedelta(hours=12)):


    from datetime import timedelta, datetime
    # connstring = 'DRIVER={PostgreSQL Unicode(x64)};Port=5432;Database=test;UID=postgres;PWD=K1ll3rk1ng'
    last_load = get_last_load(connstring, datasource, table, max_delta_load)

    inc = timedelta(hours=12)
    cload = last_load + inc
    loads = []
    while cload < datetime.now():
        loads.append((last_load, cload, datasource, table))
        last_load += inc
        cload += inc

    return loads


def query_chunk(connstring, datasource, table, query, start, end):
    with get_connection({'datasource': datasource}) as con:
        data = pd.read_sql(con, sql, params={"START": start, "END": end})


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


def update_cache():
    raise NotImplemented()

