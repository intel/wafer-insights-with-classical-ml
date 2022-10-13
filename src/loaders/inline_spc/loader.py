from datetime import datetime, timedelta

import numpy as np
from connectors.database import get_connection, get_metadatadb_connection, query_data, insert_load_start, \
    set_load_finish
from loaders.base_loader import utils

from configs.configuration import get_config

# read the config
configs = get_config()
#######################################################################################################################
############################################### CONSTANTS #############################################################
#######################################################################################################################

storage_root = configs['SPC_PATH']
devices = configs['1274']
device_string = ",".join([f"'{t}'" for t in devices])

#######################################################################################################################
#################################################### SQ2L #############################################################
#######################################################################################################################

sql = f"""
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
AND substr(ets.devrevstep,1,4) in ({device_string})
"""

sql = f"""
SELECT 
          a0.lot7 AS lot7
         ,a0.devrevstep AS devrevstep
         ,(SELECT lrc99.last_pass FROM F_LOT_RUN_CARD lrc99 where lrc99.lot =a0.lot AND lrc99.operation = a0.operation AND lrc99.site_prevout_date=a0.prev_moveout_time and rownum<=1) AS last_pass
         ,a0.rework_latest_flag AS rework_latest_flag
         ,a0.lot_rework_flag AS rework_flag
         ,a1.entity AS entity
         ,a1.ceid AS ceid
         ,a2.monitor_set_name AS monitor_set_name
         ,To_Char(a2.data_collection_time,'yyyy-mm-dd hh24:mi:ss') AS data_collect_date
         ,a2.area AS area
         ,a2.monitor_process AS process
         ,a2.status AS mon_set_status
         ,a2.spc_data_id AS spc_data_id
         ,a2.violation_flag AS violation_flag
         ,a2.latest_flag AS mon_set_latest_flag
         ,a5.chart_point_seq AS chart_point_seq
         ,a5.value AS chart_value
         ,a5.process_chamber AS process_chamber
         ,a5.wafer AS chart_wafer
         ,a5.latest_flag AS chart_pt_latest_flag
         ,a5.status AS chart_pt_status
         ,a5.chart_type AS chart_type
         ,a5.spc_chart_category AS spc_chart_category
         ,a5.spc_chart_subset AS spc_chart_subset
         ,a10.lo_control_lmt AS lo_control_lmt
         ,a10.target AS target
         ,a10.up_control_lmt AS up_control_lmt
         ,a2.test_name AS test_name
         ,a0.operation AS spc_operation
         ,a5.wafer3 AS chart_wafer3
         ,a2.monitor_type AS monitor_type
         ,a3.parameter_class AS parameter_class
         ,a9.chart_parameter AS chart_parameter
         ,a9.chart_on AS chart_on
         ,a3.measurement_set_name AS measurement_set_name
         ,a0.load_date AS load_date
FROM 
P_SPC_LOT a0
LEFT JOIN P_SPC_ENTITY a1 ON a1.spcs_id = a0.spcs_id AND a1.entity_sequence=1
INNER JOIN P_SPC_SESSION a2 ON a2.spcs_id = a0.spcs_id AND a2.data_collection_time = a0.data_collection_time
INNER JOIN P_SPC_MEASUREMENT_SET a3 ON a3.spcs_id = a2.spcs_id
LEFT JOIN P_SPC_CHART_POINT a5 ON a5.spcs_id = a3.spcs_id AND a5.measurement_set_name = a3.measurement_set_name
LEFT JOIN P_SPC_CHART a9 ON a9.chart_id = a5.chart_id
LEFT JOIN P_SPC_CHART_LIMIT a10 ON a10.chart_id = a5.chart_id AND a10.limit_id = a5.limit_id

WHERE
              SUBSTR(a0.devrevstep,1, 4) IN ({device_string})
AND      a0.LOAD_DATE > :START AND a0.LOAD_DATE <= :END
 AND      (SELECT lrc99.last_pass FROM F_LOT_RUN_CARD lrc99 where lrc99.lot =a0.lot AND lrc99.operation = a0.operation AND lrc99.site_prevout_date=a0.prev_moveout_time and rownum<=1) = 'Y' 
 AND      a0.rework_latest_flag = 'Y' 
 AND      a3.valid_flag <> 'I' 
 AND      a5.latest_flag = 'Y' 
 AND      a3.latest_flag = 'Y' 
 AND      a5.value is not NULL

"""


#######################################################################################################################


def datetime_to_pathlike_string(dt):
    return dt.strftime("%Y%m%d-%H%M%S")


def get_chunks_from_metadatadb(connstring, datasource, table, max_delta_load=timedelta(days=65),
                               incremental_load=timedelta(hours=12)):
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
    data['colnames'] = data['SPC_OPERATION'] + '`' + data['TEST_NAME'] + '`' + data['MONITOR_SET_NAME'] + '`' + data[
        'MEASUREMENT_SET_NAME'] + '`' + data['SPC_CHART_CATEGORY'] + '`' + data['SPC_CHART_SUBSET'] + '`' + data[
                           'CHART_TYPE'] + '`' + data['PARAMETER_CLASS'] + '`' + data['CHART_ON']

    sub = data.groupby(['LOT7', 'colnames']).size()
    r = sub[sub > 1].reset_index()

    # filter out degenerate column indexes
    data = data[~data['colnames'].isin(r['colnames'].unique())]

    # pv = pd.pivot_table(data, index=['PROCESS', 'DEVREVSTEP', 'LOT7'], columns='colnames', values='CHART_VALUE', aggfunc=np.median)

    return data


def store_chunk_observational_parquet(data_df, keys_columns, columns, value_column, params):
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
    # fname = params['storage_path'] + f"/{load_start}--{load_end}.parquet"
    fname = params['storage_path'] + f"/LOAD_END={load_end}"
    data_df = data_df.reset_index()

    data_df.to_parquet(fname, partition_cols=['PROCESS', 'DEVREVSTEP', 'LOT7'])


def update_cache(backload=timedelta(days=180)):
    from connectors.database import get_last_load

    mdb_connstring = "DRIVER={PostgreSQL Unicode(x64)};Port=5432;Database=test;UID=postgres;PWD=K1ll3rk1ng"

    last_load = get_last_load(mdb_connstring, "F32_PROD_XEUS", "INLINE_SPC", backload)
    print(f"last_load: {last_load}")
    last_load = datetime.now() - timedelta(days=180)

    dt = timedelta(hours=6)

    next_load = last_load + dt

    while next_load < datetime.now():
        data = query_chunk(mdb_connstring, "F32_PROD_XEUS", "INLINE_SPC", start=last_load,
                           end=next_load)
        cleaned = clean_data(data)

        store_raw_file(cleaned, params={'load_start': last_load, 'load_end': next_load, 'storage_path': storage_root})

        real_load_date = data['LOAD_DATE'].max().to_pydatetime()
        set_load_finish(mdb_connstring, [(real_load_date, next_load, last_load, "INLINE_SPC", "F32_PROD_XEUS")])
        last_load += dt
        next_load += dt


if __name__ == "__main__":
    from connectors.database import drop_load_history_by_string_id

    mdb_connstring = "DRIVER={PostgreSQL Unicode(x64)};Port=5432;Database=test;UID=postgres;PWD=K1ll3rk1ng"
    # drop_load_history_table(mdb_connstring)
    drop_load_history_by_string_id(mdb_connstring, "INLINE_ETEST")
    update_cache()
