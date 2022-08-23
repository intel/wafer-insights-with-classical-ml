from analytics.datasets import etest, sort_parametrics
from pandas import pandas
from numpy import nanprod

from datetime import datetime, timedelta

#helper method to get only the transform steps of data preprocessing

def get_transform(pipe, data):
    Xt = data
    for name, transform in pipe.steps[:-1]:
        Xt = transform.transform(Xt)

    return Xt

def get_prediction_metadata(all_data, prediction_data, fmax_col, sicc_col, type=""):
    prediction_data['SORT_DATE'] = prediction_data['TEST_END_DATE'] + sort_dt

    sort_timedata = data.loc[:, ['LOT7', 'WAFER3', 'TEST_END_DATE']]
    sort_timedata = sort_timedata.rename(columns={'TEST_END_DATE': "SORT_DATE"})
    mintime = sort_timedata['SORT_DATE'].min()

    edata = pd.merge(edata, sort_timedata, on=['LOT7', 'WAFER3'], how='left')


    #Estimate sort time with current operation test plus median_lag to sort test
    edata['SORT_DATE'] = edata['SORT_DATE'].fillna(edata['TEST_END_DATE'] + sort_dt)


    #Find feature columns in the dataframe.  All valid feature columns have the text fcol` in the columne header
    fcols = []
    for col in alldata.columns:
        if 'fcol`' in col:
            fcols.append(col)

    #there may be missing, or new, columns in recent data due to changes in the underlying data source (test programs)
    nfcols = []
    for col in prediction_data.columns:
        if 'fcol' in col:
            nfcols.append(col)

    #create set of only valid features
    fcols = list(set(fcols).intersection(set(nfcols)))

    #filter features that are actually populated.  Many have sparsity that are very high and
    #are not userfull for prediction
    fmask = alldata[fcols].notna().sum() / alldata.shape[0] > 0.8
    fcols = np.asarray(fcols)[fmask]

    #drop the response data
    alldata = alldata.dropna(subset=[fmax_token, sicc_token])
    alldata = alldata.set_index("TEST_END_DATE_y")

    #############################################################################################
    ######                       missing scalable feature engineering                   #########
    #############################################################################################
    alldata = alldata.groupby('LOT7').transform(lambda x: x.median())




    train_mask = np.random.default_rng().choice([True, False], size=len(alldata), p=[0.7, 0.3])

    #instantiate the pipelines
    fmaxpipe = Pipeline([('scaler', QuantileTransformer(output_distribution='normal')), ("imputer", SimpleImputer()),
                         ('regressor', GradientBoostingRegressor())])
    siccpipe = Pipeline([('scaler', QuantileTransformer(output_distribution='normal')), ("imputer", SimpleImputer()),
                         ('regressor', GradientBoostingRegressor())])


    fmaxpipe.fit(alldata.loc[train_mask, fcols], alldata.loc[train_mask, [fmax_token]] / alldata[fmax_token].max())
    siccpipe.fit(alldata.loc[train_mask, fcols], alldata.loc[train_mask, [sicc_token]] / alldata[sicc_token].max())

    Xfmax_T = get_transform(fmaxpipe, alldata.loc[:, fcols])
    Xsicc_T = get_transform(siccpipe, alldata.loc[:, fcols])
    Xfmax_T = pd.DataFrame(data=Xfmax_T, columns=[x.replace(".", '`') for x in fcols])
    Xsicc_T = pd.DataFrame(data=Xsicc_T, columns=[x.replace(".", '`') for x in fcols])

    fmax_lr = GradientBoostingRegressor()
    fmax_lr.fit(Xfmax_T.loc[train_mask, :], (alldata.loc[train_mask, [fmax_token]] / alldata[fmax_token].max()).ravel())

    # sicc_lr = Lasso(alpha=siccpipe.named_steps["regressor"].alpha_)
    sicc_lr = GradientBoostingRegressor()
    sicc_lr.fit(Xsicc_T.loc[train_mask, :], (alldata.loc[train_mask, [fmax_token]] / alldata[fmax_token].max()).ravel())

    #actually perform predictions.  Need to predict using pipeline to get all data and preprocessing.
    # Also predictions need to be scaled since they must be normalized for training.
    prediction_data['FMAX_Predict'] = fmaxpipe.predict(prediction_data.loc[:, fcols]) * alldata[fmax_token].max()
    prediction_data['SICC_Predict'] = siccpipe.predict(prediction_data.loc[:, fcols]) * alldata[sicc_token].max()

    alldata['FMAX_PREDICT'] = fmaxpipe.predict(alldata.loc[:, fcols]) * alldata[fmax_token].max()
    alldata['SICC_PREDICT'] = siccpipe.predict(alldata.loc[:, fcols]) * alldata[sicc_token].max()

    # Get rolling averages of predictions
    prediction_data = prediction_data.sort_values('SORT_DATE').set_index("SORT_DATE")
    prediction_data['FMAX_EWMA'] = prediction_data.rolling(window=timedelta(days=14))['FMAX_Predict'].mean()
    prediction_data['SICC_EWMA'] = prediction_data.rolling(window=timedelta(days=14))['SICC_Predict'].mean()

    return fmax_lr, sicc_lr, fmaxpipe, siccpipe, alldata, prediction_data