import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pyarrow.parquet as pq
import pyarrow as pa
etest_path = "C:/Users/eander2/PycharmProjects/WaferInsights/data/inline_etest"
sort_parametric_path = "C:/Users/eander2/PycharmProjects/WaferInsights/data/sort_parametric"

def filter_dataframe(path, device, sort_start, sort_end, operations):
    df = pd.read_parquet(path).reset_index()
    print(df.head())
    df = df[df['SHORTDEVICE'] == device]
    df = df[df['OPERATION'].isin(operations)]
    df = df[(df['TEST_END_DATE'] > sort_start) & (df['TEST_END_DATE'] <= sort_end)]
    return df

def get_loaded_tokens(path, device):
    df = pq.ParquetDataset(path, filters=[('SHORTDEVICE', '=', device)]).read(columns=['TEST_NAME']).to_pandas()
    return df['TEST_NAME'].unique()

def get_loaded_operations(path, device):
    df = pq.ParquetDataset(path, filters=[('SHORTDEVICE', '=', device)]).read(columns=['OPERATION']).to_pandas()
    return df['OPERATION'].unique()

def filter_feature_column(column, operations):
    column_split = column.split('`')
    if len(column_split) == 1:
        #cannot be a feature column and need to keep
        return True

    if column_split[0] == 'fcol':
        op = int(column_split[1])
        if op in operations:
            return True
        else:
            #is a feature column but at wrong operation
            return False

def load_etest(path, device, operation, test_start, test_end, values_column = 'MEDIAN'):
    index = ['PROCESS', 'DEVREVSTEP', 'LOT7', 'WAFER3', 'TEST_END_DATE']

    data = pq.ParquetDataset(path, filters=[('device', '=', device), ('OPERATION', '=', operation)])

    data = data.read().to_pandas()
    mask = (data['TEST_END_DATE'] > test_start) & (data['TEST_END_DATE'] < test_end)
    data = data.loc[mask, :]

    # handle potential double loading
    data = data.drop_duplicates(subset=[*index, 'TESTNAME`STRUCTURE_NAME'])
    data = data.sort_values('TEST_END_DATE')
    data = data.drop_duplicates(subset=['LOT7', 'WAFER3', 'TESTNAME`STRUCTURE_NAME'], keep='last')

    from timeit import default_timer as dt

    start = dt()
    result = data.pivot(index=index, columns='TESTNAME`STRUCTURE_NAME', values=values_column).reset_index()
    end = dt()

    print(f"pivot took: {end - start} seconds for {result.shape}")

    return result


def load_etest_by_lotlist(path, lot7_list, operations, values_column = 'MEDIAN'):
    index = ['PROCESS', 'DEVREVSTEP', 'LOT7', 'WAFER3', 'TEST_END_DATE']

    if type(operations) is not list:
        operations = [operations]

    data = pq.ParquetDataset(path, filters=[('LOT7', 'in', list(lot7_list)),('OPERATION', 'in', operations)])

    data = data.read().to_pandas()

    #handle potential double loading
    data = data.drop_duplicates(subset=[*index, 'TESTNAME`STRUCTURE_NAME'])
    data = data.sort_values('TEST_END_DATE')
    data = data.drop_duplicates(subset=['LOT7', 'WAFER3', 'TESTNAME`STRUCTURE_NAME'], keep='last')

    from timeit import default_timer as dt

    start = dt()
    result = data.pivot(index=index, columns='TESTNAME`STRUCTURE_NAME', values=values_column).reset_index()
    end = dt()

    print(f"pivot took: {end-start} seconds for {result.shape}")

    return result


if __name__=="__main__":
    device = '8PFU'
    import pyarrow as pa
    import pyarrow.parquet as pq
    from sort_parametrics import get_loaded_tokens, get_loaded_operations
    tokens = get_loaded_tokens(sort_parametric_path, device)
    ops = get_loaded_operations(sort_parametric_path, device)
    print(tokens, ops)
    fmax_operation = 132110
    sicc_operation = 132110
    start = datetime.now()
    end = start - timedelta(days=15)


    # load_sort_parametric(sort_parametric_path, device, start, end, fmax_operation, sicc_operation)


    tfilter =[('SHORTDEVICE', '==', device), ('OPERATION', 'in', [fmax_operation, sicc_operation]),
                                      ('TEST_END_DATE', '>', start), ('TEST_END_DATE', '<=', end)]

    ds = pq.ParquetDataset(sort_parametric_path,
                           filters = [('SHORTDEVICE', '=', device)])

    t = ds.read()
    df = t.to_pandas()
    lot7s = df['LOT7'].unique().tolist()
    print(lot7s)

    #vals = load_etest_by_lotlist(etest_path, lot7s, ['117113'])


    start = datetime.now()
    # vs = pd.pivot_table(vals, index = ['DEVREVSTEP', 'LOT7', 'WAFER3', 'TEST_END_DATE'], columns='TESTNAME`STRUCTURE_NAME',
    #                     values='MEDIAN', aggfunc=np.median)
    vals = load_etest_by_lotlist(etest_path, lot7s, ['117113'])

    end = datetime.now()
    print(f"query tooK: {(end - start).total_seconds()} for {vals.shape[0]} wafers")
    print(vals.reset_index().columns)

    df = df.sort_values(by=['TEST_END_DATE'])
    df = df.drop_duplicates(subset=['LOT7', 'WAFER3'], keep='last')
    print(df.iloc[0,:])
    alldata = pd.merge(df, vals, on=['LOT7', 'WAFER3'], how='right')

    print(alldata.shape, df.shape, vals.shape)
    print(alldata.iloc[0,:])
