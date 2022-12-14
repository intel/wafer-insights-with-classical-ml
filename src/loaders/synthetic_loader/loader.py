import pandas as pd
import numpy as np
from os import environ
from sklearn.datasets import make_regression
from datetime import datetime, timedelta

def datetime_to_pathlike_string(dt):
    return dt.strftime("%Y%m%d-%H%M%S")

def create_synthetic_data(observation_count=1e6):
    n_features=2000
    n_samples = observation_count
    n_informative=10
    n_targets = 2
    effective_rank=100
    tail_strength = 0.5
    noise=0.0

    X, y = make_regression(n_features=n_features, n_samples=n_samples, n_informative=n_informative,
                           n_targets=n_targets, effective_rank=effective_rank, tail_strength=tail_strength, noise=noise)

    print(X)
    feature_names = [f"fcol`feature_{x}" for x in range(X.shape[1])]
    df = pd.DataFrame(data = X, columns=feature_names)

    start = datetime.now() - timedelta(days=90)
    end = datetime.now()
    dt = (end - start) / df.shape[0]

    times = [start + dt * idx for idx in range(df.shape[0])]
    df['TEST_END_DATE'] = times
    print(df.head())
    data = parse_features(df)



    response = parse_response_df(df, y)
    return data, response

def parse_features(df):
    feature_names = list(df.columns)
    feature_names.remove('TEST_END_DATE')
    lots = []
    wafers = []
    lot_counter = 0
    wafer_counter = 0
    for idx in range(df.shape[0]):
        if idx % 25 == 0:
            lot_counter += 1
            wafer_counter = 0
        lcounterstring = str(lot_counter)
        lcounterstring = lcounterstring.rjust(10, '0')
        lotstring = f"DG{lcounterstring}"
        lots.append(lotstring)
        wafers.append(wafer_counter)
        wafer_counter += 1

    df['LOT7'] = lots
    df['WAFER3'] = wafers
    df['PROCESS'] = "1234"
    df['OPERATION'] = "6543"
    df['SHORTDEVICE'] = "DPMLD"
    df['TESTNAME'] = "TEST1"

    tostack = []
    # ['PROCESS', 'DEVREVSTEP', 'LOT7', 'WAFER3', 'TEST_END_DATE']
    print("started_stacking")
    for col in feature_names:
        dff = df.loc[:, ['LOT7', 'WAFER3', 'PROCESS','OPERATION','SHORTDEVICE', 'TESTNAME', 'TEST_END_DATE', col]]
        dff['DEVREVSTEP'] = dff['SHORTDEVICE']
        dff['TESTNAME`STRUCTURE_NAME'] = col
        dff = dff.rename(columns={col: 'MEDIAN'})
        tostack.append(dff)

    data = pd.concat(tostack, axis=0)
    print(data.head())

    return data

def parse_response_df(feature_df, y):
    colnames = [f'response_{x}' for x in range(y.shape[1])]
    df = pd.DataFrame(data = y, columns=colnames)


    df = pd.concat([df, feature_df.loc[:, ['LOT7', 'WAFER3', 'SHORTDEVICE', 'PROCESS', 'TEST_END_DATE']]], axis=1)
    df['OPERATION'] = "1234"
    tostack = []

    for col in colnames:
        dfs = df.loc[:, ['LOT7', 'WAFER3', 'SHORTDEVICE', 'PROCESS', 'OPERATION', 'TEST_END_DATE', col]]
        dfs['LOT'] = dfs['LOT7']
        dfs['PROGRAM_NAME'] = 'TEST'
        dfs['TEST_NAME'] = col
        dfs  = dfs.rename(columns={col: 'result_median'})
        tostack.append(dfs)

    alldf = pd.concat(tostack, axis=0)
    maxt = alldf['TEST_END_DATE'].max()

    filterpoint = maxt - timedelta(days=10)

    alldfs = alldf[alldf['TEST_END_DATE'] < filterpoint]

    return alldfs


def store_raw_file_et(data_df, path = "/data/features"):
    load_start = datetime_to_pathlike_string(datetime.now() - timedelta(days=90))
    load_end = datetime_to_pathlike_string(datetime.now())
    #fname = params['storage_path'] + f"/{load_start}--{load_end}.parquet"
    fname = path + f"/LOAD_END={load_end}"
    data_df = data_df.reset_index()

    data_df.to_parquet(fname, partition_cols=['PROCESS','OPERATION', 'SHORTDEVICE', 'LOT7'])

def store_raw_file_sp(data_df, path = "/data/response"):
    load_start = datetime_to_pathlike_string(datetime.now() - timedelta(days=90))
    load_end = datetime_to_pathlike_string(datetime.now())
    fname = path + f"/LOAD_END={load_end}"
    data_df.reset_index().to_parquet(fname, partition_cols=['SHORTDEVICE', 'OPERATION'])

if __name__ == "__main__":
    et_path = "../../../data/synthetic_etest"
    sp_path = "../../../data/synthetic_response"

    if environ.get('OUTPUT_DIR') is not None:
        rpath = environ.get('OUTPUT_DIR')
        et_path = rpath + "/data/synthetic_etest"
        sp_path = rpath + "/data/synthetic_response"

    num_samples = 25000

    if environ.get("NUM_SAMPLES") is not None:
        num_samples = int(environ.get("NUM_SAMPLES"))

    features, response = create_synthetic_data(num_samples)

    store_raw_file_et(features, et_path)
    store_raw_file_sp(response, sp_path)
