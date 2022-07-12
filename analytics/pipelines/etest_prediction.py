from analytics.datasets import etest, sort_parametrics
from pandas import pandas
from numpy import nanprod

from datetime import datetime, timedelta


def get_etest_pipeline(devices, process, start_date, end_date, fmax_token, sicc_token, fmax_op, sicc_op, etest_op):
