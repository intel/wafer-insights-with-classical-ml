import time
from datetime import datetime, timedelta
from os import environ
import pandas as pd
import numpy as np
#get my imports
import src.analytics.datasets.sort_parametrics as sortparam_dataset
import src.analytics.datasets.etest as etest_dataset

# sort_param_dir = "C:/Users/eander2/PycharmProjects/WaferInsights/data/sort_parametric"
# etest_dir = "C:/Users/eander2/PycharmProjects/WaferInsights/data/inline_etest"

sort_param_dir = "C:/Users/eander2/PycharmProjects/WaferInsights/data/synthetic_response"
etest_dir = "C:/Users/eander2/PycharmProjects/WaferInsights/data/synthetic_etest"

if environ.get('OUTPUT_DIR') is not None:
    rpath = environ.get('OUTPUT_DIR')
    sort_param_dir = rpath + "/data/synthetic_response"
    etest_dir = rpath + "/data/synthetic_etest"

# Dash Imports
import dash
from dash import html
from dash.dependencies import Input, Output
import utils.dash_reusable_components as drc
from dash import dcc

# Dash App Definition
app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)
app.title = "Wafer Insights"
server = app.server


# def generate_data(n_samples, dataset, noise):
#     if isinstance(dataset, list):
#         dataset = dataset[0]
#     if dataset == "moons":
#         return datasets.make_moons(n_samples=n_samples, noise=noise, random_state=0)
#
#     elif dataset == "circles":
#         return datasets.make_circles(
#             n_samples=n_samples, noise=noise, factor=0.5, random_state=1
#         )
#
#     elif dataset == "linear":
#         X, y = datasets.make_classification(
#             n_samples=n_samples,
#             n_features=2,
#             n_redundant=0,
#             n_informative=2,
#             random_state=2,
#             n_clusters_per_class=1,
#         )
#
#         rng = np.random.RandomState(2)
#         X += noise * rng.uniform(size=X.shape)
#         linearly_separable = (X, y)
#
#         return linearly_separable
#
#     else:
#         raise ValueError(
#             "Data type incorrectly specified. Please choose an existing dataset."
#         )


app.layout = html.Div(
    children=[
        # .container class is fixed, .container.scalable is scalable
        html.Div(
            className="banner",
            children=[
                # Change App Name here
                html.Div(
                    className="container scalable",
                    children=[
                        # Change App Name here
                        html.H2(
                            id="banner-title",
                            children=[
                                html.A(
                                    "WaferInsights",
                                )
                            ],
                        ),
                    ],
                )
            ],
        ),
        html.Div(
            id="body",
            className="container scalable",
            children=[
                html.Div(
                    id="app-container",
                    # className="row",
                    children=[
                        html.Div(
                            # className="three columns",
                            id="left-column",
                            children=[
                                drc.Card(
                                    id='product-card',
                                    children=[
                                        drc.NamedDropdown(
                                            name="Select Technology",
                                            id="dropdown-select-process",
                                            options=[
                                                {
                                                    "label": "1274",
                                                    "value": "1274"
                                                },
{
                                                    "label": "1272",
                                                    "value": "1272"
                                                }
                                            ],
                                        ),
                                        drc.NamedDropdown(
                                            name="Select Device",
                                            id="dropdown-select-device",
                                            options=[
                                                {
                                                    'label': "8PBXFCS",
                                                    'value': '8PBXFCS'
                                                },
                                                {
                                                    'label': '8PBMESS',
                                                    'value': '8PBMESS'
                                                }
                                            ]
                                        ),
                                        html.Div([
                                        drc.NamedDropdown(
                                            name="Select FMAX Token",
                                            id='fmax-token-select',
                                            options=[
                                                {
                                                    'label': "FMAX",
                                                    'value': 'FMAX'
                                                }
                                            ]
                                        ),
                                        drc.NamedDropdown(
                                            name="Token Operation",
                                            id='fmax-operation-select',
                                            options=[]
                                        )
                                        ]),
                                        html.Div([
                                        drc.NamedDropdown(
                                            name="Select SICC Token",
                                            id='sicc-token-select',
                                            options=[
                                                {
                                                    'label': "SICC",
                                                    'value': 'SICC'
                                                }
                                            ]
                                        ),
                                        drc.NamedDropdown(
                                            name="Token Operation",
                                            id = 'sicc-operation-select',
                                            options = []
                                        ),
                                        drc.NamedDropdown(
                                            name="ETEST Operation",
                                            id = "etest-operation-select",
                                            options = []
                                        )
                                        ]),
                                        html.Div([
                                            "Sort Date Range",
                                            dcc.DatePickerRange(
                                                id='sort-date-picker-range',
                                                min_date_allowed=datetime.now()-timedelta(days=365),
                                                max_date_allowed=datetime.now(),
                                                initial_visible_month=datetime.now(),
                                                start_date=datetime.now() - timedelta(days=20),
                                                end_date=datetime.now()
                                            )
                                        ]),
                                        html.Button(
                                            "Query Data",
                                            id='query-data-button',
                                            style={"margin-top": "20px"}
                                        ),
                                        html.Div(
                                            "",
                                            id="dummy"
                                        )

                                    ]
                                ),
                            ],
                        ),
                        html.Div(
                            id="div-graphs",
                            children=dcc.Graph(
                                id="graph-sklearn-scatter",
                            ),
                        ),
                    ],
                )
            ],
        ),
    ]
)


@app.callback(
    Output("slider-svm-parameter-gamma-power", "disabled"),
    [Input("dropdown-svm-parameter-kernel", "value")],
)
def disable_slider_param_gamma_power(kernel):
    return kernel not in ["rbf", "poly", "sigmoid"]
    #Output("dummy", "id"),
@app.callback(
    Output("div-graphs", "children"),
    [Input('query-data-button', "n_clicks"),
     Input("dropdown-select-device", 'value'),
     Input("dropdown-select-process", 'value'),
     Input('sort-date-picker-range', 'start_date'),
     Input('sort-date-picker-range', 'end_date'),
     Input('fmax-token-select', 'value'),
     Input('sicc-token-select', 'value'),
     Input('fmax-operation-select', 'value'),
     Input('sicc-operation-select', 'value'),
     Input("etest-operation-select", 'value')]
)
def query_data(n_clicks, devices, process, start_date, end_date, fmax_token, sicc_token, fmax_op, sicc_op, etest_op):
    import colorlover as cl
    import plotly.graph_objs as go
    from plotly.subplots import  make_subplots
    import plotly.figure_factory as ff


    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    print(changed_id)
    if 'query-data-button' not in changed_id:
        print("not querying")
        return "dummy2"

    data = sortparam_dataset.load_sort_parametric(sort_param_dir, devices, start_date, end_date, fmax_token, fmax_op, sicc_token, sicc_op)
    data = data.reset_index()



    print(data['TEST_END_DATE'].dtype)

    edata = etest_dataset.load_etest_by_lotlist(etest_dir, data['LOT7'].unique(), etest_op)

    def standardize(column):
        return (column - column.median())/(column.quantile(0.9) - column.quantile(0.1))


    alldata = pd.merge(data, edata, on=['LOT7', 'WAFER3'], how='inner')

    from sklearn.pipeline import  Pipeline
    from sklearn.preprocessing import RobustScaler, MinMaxScaler,  QuantileTransformer, PowerTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.linear_model import LassoCV

    from sklearn.ensemble import GradientBoostingRegressor

    #QuantileTransformer(output_distribution='normal')
    #fmaxpipe = Pipeline([('scaler', StandardScaler()), ('vt', VarianceThreshold(0.8*(1-0.8))), ("imputer", SimpleImputer()), ('regressor', LassoCV(alphas=[0.001, 0.01, 0.1, 0.5], cv=TimeSeriesSplit(n_splits=3)))])GradientBoostingRegressor( cv=TimeSeriesSplit(n_splits=3)))]
    fmaxpipe = Pipeline([ ('scaler', QuantileTransformer(output_distribution='normal')), ("imputer", SimpleImputer()),
                         ('regressor',  LassoCV(alphas=[0.001, 0.01, 0.1, 0.5], cv=TimeSeriesSplit(n_splits=3)))])
    siccpipe = Pipeline([ ('scaler', QuantileTransformer(output_distribution='normal')),("imputer", SimpleImputer()),
                         ('regressor',  LassoCV(alphas=[0.001, 0.01, 0.1, 0.5], cv=TimeSeriesSplit(n_splits=3)))])




    sort_dt = (alldata['TEST_END_DATE_x'] - alldata['TEST_END_DATE_y']).median()

    prediction_data = etest_dataset.load_etest(etest_dir, devices, etest_op,
                                               data['TEST_END_DATE'].max() - sort_dt, datetime.now())
    prediction_data['SORT_DATE'] = prediction_data['TEST_END_DATE'] + sort_dt

    sort_timedata = data.loc[:, ['LOT7', 'WAFER3','TEST_END_DATE']]
    sort_timedata = sort_timedata.rename(columns={'TEST_END_DATE': "SORT_DATE"})
    mintime = sort_timedata['SORT_DATE'].min()

    edata = pd.merge(edata, sort_timedata, on=['LOT7', 'WAFER3'], how='left')
    #replace_mask = edata['SORT_DATE'].isnull()
    edata['SORT_DATE'] = edata['SORT_DATE'].fillna(edata['TEST_END_DATE'] + sort_dt)
    #edata.loc[replace_mask, ['SORT_DATE']] = edata.loc[replace_mask, ['TEST_END_DATE']] + sort_dt

    fcols = []
    for col in alldata.columns:
        if 'fcol`' in col:
            fcols.append(col)

    nfcols = []
    for col in prediction_data.columns:
        if 'fcol' in col:
            nfcols.append(col)

    fcols = list(set(fcols).intersection(set(nfcols)))

    fmask = alldata[fcols].notna().sum()/alldata.shape[0] > 0.8
    fcols = np.asarray(fcols)[fmask]

    alldata = alldata.dropna(subset=[fmax_token, sicc_token])
    alldata = alldata.set_index("TEST_END_DATE_y")
    alldata[fmax_token] = standardize(alldata[fmax_token])
    alldata[sicc_token] = standardize(alldata[sicc_token])
    data[fmax_token] = standardize(data[fmax_token])
    data[sicc_token] = standardize(data[sicc_token])

    train_mask = np.random.default_rng().choice([True, False], size=len(alldata), p=[0.7, 0.3])

    fmaxpipe.fit(alldata.loc[train_mask, fcols], alldata.loc[train_mask, [fmax_token]]/alldata[fmax_token].max())
    siccpipe.fit(alldata.loc[train_mask, fcols], alldata.loc[train_mask, [sicc_token]]/alldata[sicc_token].max())


    def get_feature_importance(pipe, fcols):
        #mask = pipe.named_steps["xfr_select"].get_support()
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
    Xfmax_T = pd.DataFrame(data = Xfmax_T, columns=[x.replace(".", '`') for x in fcols])
    Xsicc_T = pd.DataFrame(data=Xsicc_T, columns=[x.replace(".", '`') for x in fcols])

    from sklearn.linear_model import Lasso
    #fmax_lr = Lasso(alpha=fmaxpipe.named_steps["regressor"].alpha_)
    fmax_lr = GradientBoostingRegressor()
    fmax_lr.fit(Xfmax_T.loc[train_mask, :], (alldata.loc[train_mask, [fmax_token]][fmax_token]/alldata[fmax_token].max()).ravel())

    #sicc_lr = Lasso(alpha=siccpipe.named_steps["regressor"].alpha_)
    sicc_lr = GradientBoostingRegressor()
    sicc_lr.fit(Xsicc_T.loc[train_mask, :], (alldata.loc[train_mask, [sicc_token]][sicc_token] / alldata[sicc_token].max()).ravel())
    
    fi_fmax = get_feature_importance(fmaxpipe, fcols)
    fi_sicc = get_feature_importance(siccpipe, fcols)


    prediction_data['FMAX_Predict'] = fmaxpipe.predict(prediction_data.loc[:, fcols])*alldata[fmax_token].max()
    prediction_data['SICC_Predict'] = siccpipe.predict(prediction_data.loc[:, fcols])*alldata[sicc_token].max()

    alldata['FMAX_PREDICT'] = fmaxpipe.predict(alldata.loc[:, fcols])*alldata[fmax_token].max()
    alldata['SICC_PREDICT'] = siccpipe.predict(alldata.loc[:, fcols]) * alldata[sicc_token].max()

    #prediction_data = prediction_data.set_index('SORT_DATE')
    prediction_data = prediction_data.sort_values('SORT_DATE').set_index("SORT_DATE")
    prediction_data['FMAX_EWMA'] = prediction_data.rolling(window = timedelta(days=14))['FMAX_Predict'].mean()
    prediction_data['SICC_EWMA'] = prediction_data.rolling(window = timedelta(days=14))['SICC_Predict'].mean()


    prediction_data = prediction_data.reset_index()

    scat = go.Figure([go.Scatter(x=data['TEST_END_DATE'], y=data[fmax_token], mode="markers", name="SORT_FMAX")])
    scat.add_scatter(x = alldata['TEST_END_DATE_x'], y=alldata[fmax_token], name='Measured@Etest',
                     mode="markers", hovertext=list(zip(alldata['LOT7'], alldata['WAFER3'])))
    scat.add_scatter(x = prediction_data['SORT_DATE'], y=prediction_data['FMAX_Predict'], mode="markers", name='Predicted')
    scat.add_scatter(x = prediction_data['SORT_DATE'], y=prediction_data['FMAX_EWMA'], name='EWMA')

    scat2 = go.Figure([go.Scatter(x=data['TEST_END_DATE'], y=data[sicc_token], mode="markers", name="SORT_SICC")])
    scat2.add_scatter(x=alldata['TEST_END_DATE_x'], y=alldata[sicc_token], name='Measured@Etest',
                     mode="markers", hovertext=list(zip(alldata['LOT7'], alldata['WAFER3'])))
    scat2.add_scatter(x=prediction_data['SORT_DATE'], y=prediction_data['SICC_Predict'], mode="markers",
                      hovertext=list(zip(alldata['LOT7'], alldata['WAFER3'])), name='Predicted')
    scat2.add_scatter(x=prediction_data['SORT_DATE'], y=prediction_data['SICC_EWMA'], name='EWMA')

    # corr = go.Figure([go.Scatter(x=data[sicc_token], y=data[fmax_token],mode="kde")])
    def densmap(x, y):
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=x, y=y, mode='markers'))
        fig.add_trace(go.Histogram2d(x=x, y=y))
        fig.add_trace(go.Histogram2dContour(x=x, y=y))
        return fig


    #corr = ff.create_2d_density(alldata[fmax_token], alldata['FMAX_PREDICT'])
    corr = densmap(alldata[fmax_token], alldata['FMAX_PREDICT'])
    #corr2 = ff.create_2d_density(alldata[sicc_token], alldata['SICC_PREDICT'])
    corr2 = densmap(alldata[sicc_token], alldata['SICC_PREDICT'])

    #corr3 = densmap(alldata[fmax_token], alldata[sicc_token])

    print(edata.head())

    fig_children = []
    for feats in range(3):
        fi_fig = make_subplots(1,2)
        fi_fig.add_trace(go.Scatter(x=alldata[fi_fmax[feats][0]], y=alldata[fmax_token], name=fi_fmax[feats][0], mode='markers'), row=1, col=1)
        fi_fig.add_trace(go.Scatter(x=alldata[fi_sicc[feats][0]], y=alldata[sicc_token], name=fi_sicc[feats][0], mode='markers'),
                      row=1, col=2)

        fig_children.append(dcc.Graph(id=f"feature_importance_{feats}", figure=fi_fig))


    return [
        html.Div(
            id="svm-graph-container",
            children=dcc.Loading(
                className="graph-wrapper",
                children=[dcc.Graph(id="graph-sklearn-scatter", figure=scat), dcc.Graph(id="graph-sicc", figure=scat2)],
                style={"display": "none"},
            ),
        ),
        html.Div(
            id="graphs-container",
            children=[
                dcc.Loading(
                    className="graph-wrapper",
                    children=dcc.Graph(id="graph-line-roc-curve", figure=corr),
                ),
                dcc.Loading(
                    className="graph-wrapper",
                    children=dcc.Graph(
                        id="graph-pie-confusion-matrix", figure=corr2
                    ),
                ),
            ],
        )
        # html.Div(
        #     id='feature-graphs-container',
        #     children=fi_children
        # )
    ]



@app.callback(
    Output("dropdown-select-device", "options"),
    Input("dropdown-select-process", 'value')
)
def get_devices(process):
    print(process)
    if process is None:
        return []
    if process == '1274':
        devices = list(sortparam_dataset.get_loaded_devices(sort_param_dir))
        print(devices)
        return [{'label': str(x), 'value': str(x)} for x in devices]

    if process == '1272':
        devices = sortparam_dataset.get_loaded_devices(sort_param_dir)
        return [{'label': str(x), 'value': str(x)} for x in devices]

@app.callback(
    [Output("fmax-token-select", 'options'), Output("sicc-token-select", 'options')],
    [Input("dropdown-select-device", 'value')]
)
def get_fmax_token(device):
    print(f"get fmax token {device}")
    if device is None:
        return [], []
    tokens = list(sortparam_dataset.get_loaded_tokens(sort_param_dir, device))
    print(f"tokens: {tokens}")
    return [{'label': x, 'value': x} for x in tokens], [{'label': x, 'value': x} for x in tokens]


@app.callback(
    [Output("fmax-operation-select", 'options'), Output("sicc-operation-select", 'options')],
    Input("dropdown-select-device", 'value')
)
def get_operations(device):
    if device is None:
        return [], []
    ops = list(sortparam_dataset.get_loaded_operations(sort_param_dir, device))
    print(ops)
    return [{'label': x, 'value': x} for x in ops], [{'label': x, 'value': x} for x in ops]

@app.callback(
    Output("etest-operation-select", 'options'),
    Input("dropdown-select-device", 'value')
)
def get_etest_operation(device):
    if device is None:
        return []

    ops = list(etest_dataset.get_loaded_operations(etest_dir, device))
    print(f"etest ops {ops}")
    return [{'label': x, 'value': x} for x in ops]

# @app.callback(
#     Output("div-graphs", "children"),
#     [
#         Input("dropdown-svm-parameter-kernel", "value"),
#         Input("slider-svm-parameter-degree", "value"),
#         Input("slider-svm-parameter-C-coef", "value"),
#         Input("slider-svm-parameter-C-power", "value"),
#         Input("slider-svm-parameter-gamma-coef", "value"),
#         Input("slider-svm-parameter-gamma-power", "value"),
#         Input("dropdown-select-dataset", "value"),
#         Input("slider-dataset-noise-level", "value"),
#         Input("radio-svm-parameter-shrinking", "value"),
#         Input("slider-threshold", "value"),
#         Input("slider-dataset-sample-size", "value"),
#     ],
# )
# def update_svm_graph(
#     kernel,
#     degree,
#     C_coef,
#     C_power,
#     gamma_coef,
#     gamma_power,
#     dataset,
#     noise,
#     shrinking,
#     threshold,
#     sample_size,
# ):
#     t_start = time.time()
#     h = 0.3  # step size in the mesh
#
#     # Data Pre-processing
#     X, y = generate_data(n_samples=sample_size, dataset=dataset, noise=noise)
#     X = StandardScaler().fit_transform(X)
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.4, random_state=42
#     )
#
#     x_min = X[:, 0].min() - 0.5
#     x_max = X[:, 0].max() + 0.5
#     y_min = X[:, 1].min() - 0.5
#     y_max = X[:, 1].max() + 0.5
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#
#     C = C_coef * 10 ** C_power
#     gamma = gamma_coef * 10 ** gamma_power
#
#     if shrinking == "True":
#         flag = True
#     else:
#         flag = False
#
#     # Train SVM
#     clf = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, shrinking=flag)
#     clf.fit(X_train, y_train)
#
#     # Plot the decision boundary. For that, we will assign a color to each
#     # point in the mesh [x_min, x_max]x[y_min, y_max].
#     if hasattr(clf, "decision_function"):
#         Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
#     else:
#         Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
#
#     prediction_figure = figs.serve_prediction_plot(
#         model=clf,
#         X_train=X_train,
#         X_test=X_test,
#         y_train=y_train,
#         y_test=y_test,
#         Z=Z,
#         xx=xx,
#         yy=yy,
#         mesh_step=h,
#         threshold=threshold,
#     )
#
#     roc_figure = figs.serve_roc_curve(model=clf, X_test=X_test, y_test=y_test)
#
#     confusion_figure = figs.serve_pie_confusion_matrix(
#         model=clf, X_test=X_test, y_test=y_test, Z=Z, threshold=threshold
#     )
#
#     return [
#         html.Div(
#             id="svm-graph-container",
#             children=dcc.Loading(
#                 className="graph-wrapper",
#                 children=dcc.Graph(id="graph-sklearn-svm", figure=prediction_figure),
#                 style={"display": "none"},
#             ),
#         ),
#         html.Div(
#             id="graphs-container",
#             children=[
#                 dcc.Loading(
#                     className="graph-wrapper",
#                     children=dcc.Graph(id="graph-line-roc-curve", figure=roc_figure),
#                 ),
#                 dcc.Loading(
#                     className="graph-wrapper",
#                     children=dcc.Graph(
#                         id="graph-pie-confusion-matrix", figure=confusion_figure
#                     ),
#                 ),
#             ],
#         ),
#     ]


# Running the server
if __name__ == "__main__":
    app.run_server(debug=False)
