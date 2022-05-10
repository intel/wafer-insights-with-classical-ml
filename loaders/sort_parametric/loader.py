
from connectors.database import get_connection, get_metadatadb_connection, query_data, insert_load_start, set_load_finish, create_history_table
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

#######################################################################################################################
#################################################### SQ2L #############################################################
#######################################################################################################################

sql = """
select /*+ ordered */
            TS.Lot LOT
            ,substr(TS.Lot,1,7) LOT7
           ,TS.Wafer_ID Wafer3
           ,DT.Sort_X x
           ,DT.sort_Y y
           ,DT.Interface_Bin sortBin
           ,TS.program_name Program_name
           ,TS.test_end_date_time test_end_date
           ,TS.devrevstep
           ,substr(TS.devrevstep,1,4) SHORTDEVICE
           ,PR.numeric_result result
           ,TS.OPERATION
           ,T.TEST_NAME
           ,TS.LOAD_END_DATE_TIME AS LOAD_DATE
           ,substr(TS.process,1,4) AS process
           ,TS.wafer_scribe AS wafer_scribe
    from    a_testing_session TS
           ,a_test T
           ,a_parametric_result PR
           ,a_device_testing DT
    where   DT.lao_start_ww            = TS.lao_start_ww
        and DT.ts_id                   = TS.ts_id
        and DT.latest_flag             = 'Y'
        and TS.latest_flag             = 'Y'
        and T.devrevstep               = TS.devrevstep
        and T.program_name             = TS.program_name
        and PR.lao_start_ww            = TS.lao_start_ww
        and PR.ts_id                   = TS.ts_id
        and PR.latest_flag             = 'Y'
        and PR.lao_start_ww            = DT.lao_start_ww
        and PR.ts_id                   = DT.ts_id
        and PR.dt_id                   = DT.dt_id
        and PR.t_id                    = T.t_id
        and nvl(T.temperature,-99.999) = nvl(TS.temperature,-99.999)
        and DT.Goodbad_Flag = 'G'
        and TS.wafer_id is not null
        and DT.sort_x is not null
        and DT.sort_y is not null
        and ts.LOAD_END_DATE_TIME > :START AND ts.LOAD_END_DATE_TIME <= :END
        and (ts.devrevstep like '8PFU%' OR ts.devrevstep like '8PJS%' OR ts.devrevstep like '8PJR%') 
        and T.test_name in ('IDV_1204_XNOM3GRIDNESTED_FULLDIE_0950_MED', 'PTH_POWER::POWER_X_SCREEN_K_BEGIN_X_X_X_X_SICC_CALC_PP_SICC_VCC0_V1_500MA_FC',
        'IDV_2204_XNOM3GNES12_FULLDIE_0950_MED', 'PTH_POWER_SDT_SICC_ALLCORES_SCALED_V2', 'IDV_2204_XNOM3GNES12_FULLDIE_0950_MED',
        'PP_PWR_SICC_GLC0_V1')
"""

#######################################################################################################################
############################################### CONSTANTS #############################################################
#######################################################################################################################
storage_root = "C:/Users/eander2/PycharmProjects/WaferInsights/data/sort_parametric"


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


def query_chunk(metadatadb_conconfig, datasource, loader_string_id, start, end):
    key = ['LOT7', 'WAFER3', 'TEST_END_DATE', 'OPERATION', 'PROGRAM_NAME', 'TEST_NAME']
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
    data['fcol_mean'] = 'fcol`' + data['TEST_NAME'] + '`MEAN'
    data['fcol_median'] = 'fcol`' + data['TEST_NAME'] + '`MEDIAN'



    #pivoted parameter data
    key = ['DEVREVSTEP', 'SHORTDEVICE', 'LOT', 'LOT7', 'WAFER3', 'PROGRAM_NAME', 'TEST_END_DATE', 'OPERATION', 'TEST_NAME']

    wl_data = data.groupby(by=key).agg(result_mean = pd.NamedAgg(column='RESULT', aggfunc='mean'),
                                       result_median = pd.NamedAgg(column='RESULT', aggfunc='median'))


    print(wl_data.head())


    # data['TESTNAME`STRUCTURE_NAME'] = data['OPERATION'] + data['TEST_NAME'] + '`' + data['STRUCTURE_NAME']
    # data['aggname_median'] = data['TESTNAME`STRUCTURE_NAME'] + '`MEDIAN'
    # data['aggname_mean'] = data['TESTNAME`STRUCTURE_NAME'] + '`MEAN'
    # index = ['PROCESS','SHORTDEVICE', 'LOT7', 'WAFER3', 'DEVREVSTEP', 'PROGRAM_NAME', 'LOAD_DATE']
    # pivot_data_median = pd.pivot_table(data, values = 'MEDIAN', columns='aggname_median', index=index)
    # pivot_data_mean = pd.pivot_table(data, values='MEAN', columns='aggname_mean', index=index)
    #
    # alldata = pivot_data_median.join(pivot_data_mean, on=index)
    return wl_data



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
    fname = params['storage_path'] + f"/LOAD_END={load_end}"
    data_df.reset_index().to_parquet(fname, partition_cols=['SHORTDEVICE', 'OPERATION'])

def update_cache(backload = timedelta(days=90)):
    from connectors.database import drop_load_history_table, get_last_load

    mdb_connstring = "DRIVER={PostgreSQL Unicode(x64)};Port=5432;Database=test;UID=postgres;PWD=K1ll3rk1ng"

    last_load = get_last_load(mdb_connstring, "F32_PROD_XEUS", "SORT_PARAMETRIC", backload)
    print(f"last_load: {last_load}")
    last_load = datetime.now() - timedelta(days=90)
    dt = timedelta(days=1)

    next_load = last_load + dt

    while next_load < datetime.now():
        data = query_chunk(mdb_connstring, "F32_PROD_XEUS", "SORT_PARAMETRIC", start=last_load,
                           end=next_load)
        cleaned = clean_data(data)

        store_raw_file(cleaned, params={'load_start': last_load, 'load_end': next_load, 'storage_path': storage_root})

        real_load_date = data['LOAD_DATE'].max().to_pydatetime()
        set_load_finish(mdb_connstring, [(real_load_date, next_load, last_load, "SORT_PARAMETRIC", "F32_PROD_XEUS")])
        last_load += dt
        next_load += dt



if __name__=="__main__":
    import PyUber
    from connectors.database import drop_load_history_table

    import PyUber
    from connectors.database import drop_load_history_table

    mdb_connstring = "DRIVER={PostgreSQL Unicode(x64)};Port=5432;Database=test;UID=postgres;PWD=K1ll3rk1ng"
    # drop_load_history_table(mdb_connstring)
    update_cache()

    exit()

    mdb_connstring = "DRIVER={PostgreSQL Unicode(x64)};Port=5432;Database=test;UID=postgres;PWD=K1ll3rk1ng"
    #drop_load_history_table(mdb_connstring)
    last_load = datetime.now() - timedelta(days=90)
    next_load = datetime.now()
    data = query_chunk(mdb_connstring, "F32_PROD_XEUS", "SORT_PARAMETRIC", start=last_load,
                       end=next_load)

    print(data.head())

    for c in data.columns:
        print(c)

    clean_data(data)

    data.to_parquet("data/sort_parametric.parquet")
    #update_cache()

