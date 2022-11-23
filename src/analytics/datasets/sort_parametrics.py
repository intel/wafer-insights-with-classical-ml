from datetime import datetime, timedelta

import pandas as pd
import pyarrow.parquet as pq

sort_parametric_path = "C:/Users/eander2/PycharmProjects/WaferInsights/data/sort_parametric"


def filter_dataframe(path, device, sort_start, sort_end, operations):
    df = pd.read_parquet(path).reset_index()
    print(df.head())
    df = df[df['SHORTDEVICE'] == device]
    df = df[df['OPERATION'].isin(operations)]
    df = df[(df['TEST_END_DATE'] > sort_start) & (df['TEST_END_DATE'] <= sort_end)]
    return df


def get_loaded_devices(path):
    df = pq.ParquetDataset(path).read(columns=['SHORTDEVICE']).to_pandas()
    return df['SHORTDEVICE'].unique()


def get_loaded_tokens(path, device):
    df = pq.ParquetDataset(path, filters=[('SHORTDEVICE', '=', device)]).read(columns=['TEST_NAME']).to_pandas()
    return df['TEST_NAME'].unique()


def get_loaded_operations(path, device):
    df = pq.ParquetDataset(path, filters=[('SHORTDEVICE', '=', device)]).read(columns=['TEST_NAME']).to_pandas()
    return df['OPERATION'].unique()


def load_sort_parametric(path, device, sort_start, sort_end, fmax_token, fmax_operation, sicc_token, sicc_operation):
    data = pq.ParquetDataset(path, filters=[('SHORTDEVICE', '=', device),
                                            ('OPERATION', 'in', [fmax_operation, sicc_operation])]).read().to_pandas()
    df = data[(data['TEST_END_DATE'] > sort_start) & (data['TEST_END_DATE'] <= sort_end)]

    sicc_data = df[(df['TEST_NAME'] == sicc_token) & (df['OPERATION'] == sicc_operation)]
    fmax_data = df[(df['TEST_NAME'] == fmax_token) & (df['OPERATION'] == fmax_operation)]
    fdata = pd.concat([sicc_data, fmax_data])
    fdata = fdata.sort_values(by='TEST_END_DATE')
    fdata = fdata.drop_duplicates(subset=['LOT7', 'WAFER3', 'TEST_END_DATE', 'TEST_NAME'], keep='first')
    return fdata.pivot(index=['LOT7', 'WAFER3', 'TEST_END_DATE'], columns='TEST_NAME',
                       values='result_median').reset_index()


if __name__ == "__main__":
    device = '8PFU'
    import pyarrow.parquet as pq

    tokens = get_loaded_tokens(sort_parametric_path, device)
    ops = get_loaded_operations(sort_parametric_path, device)
    print(tokens, ops)
    fmax_operation = 132110
    sicc_operation = 132110
    start = datetime.now()
    end = start - timedelta(days=15)

    # load_sort_parametric(sort_parametric_path, device, start, end, fmax_operation, sicc_operation)

    tfilter = [('SHORTDEVICE', '==', device), ('OPERATION', 'in', [fmax_operation, sicc_operation]),
               ('TEST_END_DATE', '>', start), ('TEST_END_DATE', '<=', end)]

    ds = pq.ParquetDataset(sort_parametric_path,
                           filters=[('SHORTDEVICE', '=', device)])
    t = ds.read()
    df = t.to_pandas()
    print(df.head())
    print(f"___ {df.reset_index()['SHORTDEVICE'].unique()}")
    print(df.shape)
    print(df['TEST_NAME'].unique())
    print(df.iloc[0, :])
