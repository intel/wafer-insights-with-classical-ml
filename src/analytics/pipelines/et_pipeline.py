from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, MinMaxScaler, QuantileTransformer, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LassoCV

from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np
from datetime import timedelta


def standardize(column):
    return (column - column.median()) / (column.quantile(0.9) - column.quantile(0.1))

def get_model(data, alldata, prediction_data, edata, fmax_token, sicc_token):


    # QuantileTransformer(output_distribution='normal')
    # fmaxpipe = Pipeline([('scaler', StandardScaler()), ('vt', VarianceThreshold(0.8*(1-0.8))), ("imputer", SimpleImputer()), ('regressor', LassoCV(alphas=[0.001, 0.01, 0.1, 0.5], cv=TimeSeriesSplit(n_splits=3)))])GradientBoostingRegressor( cv=TimeSeriesSplit(n_splits=3)))]
    fmaxpipe = Pipeline([('scaler', QuantileTransformer(output_distribution='normal')), ("imputer", SimpleImputer()),
                         ('regressor', LassoCV(alphas=[0.001, 0.01, 0.1, 0.5], cv=TimeSeriesSplit(n_splits=3)))])
    siccpipe = Pipeline([('scaler', QuantileTransformer(output_distribution='normal')), ("imputer", SimpleImputer()),
                         ('regressor', LassoCV(alphas=[0.001, 0.01, 0.1, 0.5], cv=TimeSeriesSplit(n_splits=3)))])

    sort_dt = (alldata['TEST_END_DATE_x'] - alldata['TEST_END_DATE_y']).median()

    # prediction_data = etest_dataset.load_etest(etest_dir, devices, etest_op,
    #                                            data['TEST_END_DATE'].max() - sort_dt, datetime.now())
    prediction_data['SORT_DATE'] = prediction_data['TEST_END_DATE'] + sort_dt

    sort_timedata = data.loc[:, ['LOT7', 'WAFER3', 'TEST_END_DATE']]
    sort_timedata = sort_timedata.rename(columns={'TEST_END_DATE': "SORT_DATE"})
    mintime = sort_timedata['SORT_DATE'].min()

    edata = pd.merge(edata, sort_timedata, on=['LOT7', 'WAFER3'], how='left')
    # replace_mask = edata['SORT_DATE'].isnull()
    edata['SORT_DATE'] = edata['SORT_DATE'].fillna(edata['TEST_END_DATE'] + sort_dt)
    # edata.loc[replace_mask, ['SORT_DATE']] = edata.loc[replace_mask, ['TEST_END_DATE']] + sort_dt

    fcols = []
    for col in alldata.columns:
        if 'fcol`' in col:
            fcols.append(col)

    nfcols = []
    for col in prediction_data.columns:
        if 'fcol' in col:
            nfcols.append(col)

    fcols = list(set(fcols).intersection(set(nfcols)))

    fmask = alldata[fcols].notna().sum() / alldata.shape[0] > 0.8
    fcols = np.asarray(fcols)[fmask]

    alldata = alldata.dropna(subset=[fmax_token, sicc_token])
    alldata = alldata.set_index("TEST_END_DATE_y")
    alldata[fmax_token] = standardize(alldata[fmax_token])
    alldata[sicc_token] = standardize(alldata[sicc_token])
    data[fmax_token] = standardize(data[fmax_token])
    data[sicc_token] = standardize(data[sicc_token])

    train_mask = np.random.default_rng().choice([True, False], size=len(alldata), p=[0.7, 0.3])

    fmaxpipe.fit(alldata.loc[train_mask, fcols], alldata.loc[train_mask, [fmax_token]] / alldata[fmax_token].max())
    siccpipe.fit(alldata.loc[train_mask, fcols], alldata.loc[train_mask, [sicc_token]] / alldata[sicc_token].max())

    def get_feature_importance(pipe, fcols):
        # mask = pipe.named_steps["xfr_select"].get_support()
        r1 = pipe.named_steps["regressor"]
        fi = zip(np.asarray(fcols), np.abs(r1.coef_))
        fi = sorted(fi, key=lambda x: x[1], reverse=True)
        return fi

    def get_transform(pipe, data):
        Xt = data
        for name, transform in pipe.steps[:-1]:
            Xt = transform.transform(Xt)

        return Xt

    Xfmax_T = get_transform(fmaxpipe, alldata.loc[:, fcols])
    Xsicc_T = get_transform(siccpipe, alldata.loc[:, fcols])
    Xfmax_T = pd.DataFrame(data=Xfmax_T, columns=[x.replace(".", '`') for x in fcols])
    Xsicc_T = pd.DataFrame(data=Xsicc_T, columns=[x.replace(".", '`') for x in fcols])

    from sklearn.linear_model import Lasso
    fmax_lr = Lasso(alpha=fmaxpipe.named_steps["regressor"].alpha_)

    fmax_lr.fit(Xfmax_T.loc[train_mask, :],
                (alldata.loc[train_mask, [fmax_token]][fmax_token] / alldata[fmax_token].max()).ravel())

    sicc_lr = Lasso(alpha=siccpipe.named_steps["regressor"].alpha_)

    sicc_lr.fit(Xsicc_T.loc[train_mask, :],
                (alldata.loc[train_mask, [sicc_token]][sicc_token] / alldata[sicc_token].max()).ravel())

    fi_fmax = get_feature_importance(fmaxpipe, fcols)
    fi_sicc = get_feature_importance(siccpipe, fcols)

    prediction_data['FMAX_Predict'] = fmaxpipe.predict(prediction_data.loc[:, fcols]) * alldata[fmax_token].max()
    prediction_data['SICC_Predict'] = siccpipe.predict(prediction_data.loc[:, fcols]) * alldata[sicc_token].max()

    alldata['FMAX_PREDICT'] = fmaxpipe.predict(alldata.loc[:, fcols]) * alldata[fmax_token].max()
    alldata['SICC_PREDICT'] = siccpipe.predict(alldata.loc[:, fcols]) * alldata[sicc_token].max()

    # prediction_data = prediction_data.set_index('SORT_DATE')
    prediction_data = prediction_data.sort_values('SORT_DATE').set_index("SORT_DATE")
    prediction_data['FMAX_EWMA'] = prediction_data.rolling(window=timedelta(days=14))['FMAX_Predict'].mean()
    prediction_data['SICC_EWMA'] = prediction_data.rolling(window=timedelta(days=14))['SICC_Predict'].mean()

    prediction_data = prediction_data.reset_index()

    return data, alldata, prediction_data, fi_fmax, fi_sicc,