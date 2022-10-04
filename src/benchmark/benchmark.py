import time
from datetime import datetime, timedelta
from os import environ
import pandas as pd
import numpy as np
#get my imports
import src.analytics.datasets.sort_parametrics as sortparam_dataset
import src.analytics.datasets.etest as etest_dataset
import src.analytics.pipelines.et_pipeline as et_pipeline
sort_param_dir = "../../data/synthetic_response"
etest_dir = "../../data/synthetic_etest"

if environ.get('OUTPUT_DIR') is not None:
    rpath = environ.get('OUTPUT_DIR')
    sort_param_dir = rpath + "/data/synthetic_response"
    etest_dir = rpath + "/data/synthetic_etest"


def run_benchmark(devices='DPMLD', token_0='response_0', token_0_op='1234',
                  token_1='response_1', token_1_op='1234', feature_op='6543'):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=45)
    data = sortparam_dataset.load_sort_parametric(sort_param_dir, devices, start_date, end_date, token_0, token_0_op,
                                                  token_1, token_1_op)
    data = data.reset_index()

    edata = etest_dataset.load_etest_by_lotlist(etest_dir, data['LOT7'].unique(), feature_op)

    alldata = pd.merge(data, edata, on=['LOT7', 'WAFER3'], how='inner')

    sort_dt = (alldata['TEST_END_DATE_x'] - alldata['TEST_END_DATE_y']).median()

    prediction_data = etest_dataset.load_etest(etest_dir, devices, etest_op,
                                               data['TEST_END_DATE'].max() - sort_dt, datetime.now())

    data, alldata, prediction_data, fi_fmax, fi_sicc = et_pipeline.get_model(data, alldata, prediction_data, edata,
                                                                             token_0, token_1)


if __name__ == "__main__":
    run_benchmark()