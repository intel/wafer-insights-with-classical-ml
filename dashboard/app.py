import time
from datetime import datetime, timedelta
import importlib

#get my imports
import analytics.datasets.sort_parametrics as sortparam_dataset
import analytics.datasets.etest as etest_dataset

sort_param_dir = "C:/Users/eander2/PycharmProjects/WaferInsights/data/sort_parametric"
etest_dir = "C:/Users/eander2/PycharmProjects/WaferInsights/data/inline_etest"

import dash
# import dash_core_components as dcc
from dash import dcc

from dash import html
import numpy as np
from dash.dependencies import Input, Output, State
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.svm import SVC

import utils.dash_reusable_components as drc

from dash import dcc
import utils.figures as figs

app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)
app.title = "Support Vector Machine"
server = app.server


def generate_data(n_samples, dataset, noise):
    if isinstance(dataset, list):
        dataset = dataset[0]
    if dataset == "moons":
        return datasets.make_moons(n_samples=n_samples, noise=noise, random_state=0)

    elif dataset == "circles":
        return datasets.make_circles(
            n_samples=n_samples, noise=noise, factor=0.5, random_state=1
        )

    elif dataset == "linear":
        X, y = datasets.make_classification(
            n_samples=n_samples,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            random_state=2,
            n_clusters_per_class=1,
        )

        rng = np.random.RandomState(2)
        X += noise * rng.uniform(size=X.shape)
        linearly_separable = (X, y)

        return linearly_separable

    else:
        raise ValueError(
            "Data type incorrectly specified. Please choose an existing dataset."
        )


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
                                    href="https://github.com/plotly/dash-svm",
                                    style={
                                        "text-decoration": "none",
                                        "color": "inherit",
                                    },
                                )
                            ],
                        ),
                        html.A(
                            id="banner-logo",
                            children=[
                                html.Img(src=app.get_asset_url("dash-logo-new.png"))
                            ],
                            href="https://plot.ly/products/dash/",
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
                                        )]),
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
                                drc.Card(
                                    id="first-card",
                                    children=[
                                        drc.NamedDropdown(
                                            name="Select Dataset",
                                            id="dropdown-select-dataset",
                                            options=[
                                                {"label": "Moons", "value": "moons"},
                                                {
                                                    "label": "Linearly Separable",
                                                    "value": "linear",
                                                },
                                                {
                                                    "label": "Circles",
                                                    "value": "circles",
                                                },
                                            ],
                                            clearable=False,
                                            searchable=False,
                                            value="moons",
                                            multi=True
                                        ),
                                        drc.NamedSlider(
                                            name="Sample Size",
                                            id="slider-dataset-sample-size",
                                            min=100,
                                            max=500,
                                            step=100,
                                            marks={
                                                str(i): str(i)
                                                for i in [100, 200, 300, 400, 500]
                                            },
                                            value=300,
                                        ),
                                        drc.NamedSlider(
                                            name="Noise Level",
                                            id="slider-dataset-noise-level",
                                            min=0,
                                            max=1,
                                            marks={
                                                i / 10: str(i / 10)
                                                for i in range(0, 11, 2)
                                            },
                                            step=0.1,
                                            value=0.2,
                                        ),
                                    ],
                                ),
                                drc.Card(
                                    id="button-card",
                                    children=[
                                        drc.NamedSlider(
                                            name="Threshold",
                                            id="slider-threshold",
                                            min=0,
                                            max=1,
                                            value=0.5,
                                            step=0.01,
                                        ),
                                        html.Button(
                                            "Reset Threshold",
                                            id="button-zero-threshold",
                                        ),
                                    ],
                                ),
                                drc.Card(
                                    id="last-card",
                                    children=[
                                        drc.NamedDropdown(
                                            name="Kernel",
                                            id="dropdown-svm-parameter-kernel",
                                            options=[
                                                {
                                                    "label": "Radial basis function (RBF)",
                                                    "value": "rbf",
                                                },
                                                {"label": "Linear", "value": "linear"},
                                                {
                                                    "label": "Polynomial",
                                                    "value": "poly",
                                                },
                                                {
                                                    "label": "Sigmoid",
                                                    "value": "sigmoid",
                                                },
                                            ],
                                            value="rbf",
                                            clearable=False,
                                            searchable=False,
                                        ),
                                        drc.NamedSlider(
                                            name="Cost (C)",
                                            id="slider-svm-parameter-C-power",
                                            min=-2,
                                            max=4,
                                            value=0,
                                            marks={
                                                i: "{}".format(10 ** i)
                                                for i in range(-2, 5)
                                            },
                                        ),
                                        drc.FormattedSlider(
                                            id="slider-svm-parameter-C-coef",
                                            min=1,
                                            max=9,
                                            value=1,
                                        ),
                                        drc.NamedSlider(
                                            name="Degree",
                                            id="slider-svm-parameter-degree",
                                            min=2,
                                            max=10,
                                            value=3,
                                            step=1,
                                            marks={
                                                str(i): str(i) for i in range(2, 11, 2)
                                            },
                                        ),
                                        drc.NamedSlider(
                                            name="Gamma",
                                            id="slider-svm-parameter-gamma-power",
                                            min=-5,
                                            max=0,
                                            value=-1,
                                            marks={
                                                i: "{}".format(10 ** i)
                                                for i in range(-5, 1)
                                            },
                                        ),
                                        drc.FormattedSlider(
                                            id="slider-svm-parameter-gamma-coef",
                                            min=1,
                                            max=9,
                                            value=5,
                                        ),
                                        html.Div(
                                            id="shrinking-container",
                                            children=[
                                                html.P(children="Shrinking"),
                                                dcc.RadioItems(
                                                    id="radio-svm-parameter-shrinking",
                                                    labelStyle={
                                                        "margin-right": "7px",
                                                        "display": "inline-block",
                                                    },
                                                    options=[
                                                        {
                                                            "label": " Enabled",
                                                            "value": "True",
                                                        },
                                                        {
                                                            "label": " Disabled",
                                                            "value": "False",
                                                        },
                                                    ],
                                                    value="True",
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            id="div-graphs",
                            children=dcc.Graph(
                                id="graph-sklearn-svm",
                                figure=dict(
                                    layout=dict(
                                        plot_bgcolor="#282b38", paper_bgcolor="#282b38"
                                    )
                                ),
                            ),
                        ),
                    ],
                )
            ],
        ),
    ]
)


@app.callback(
    Output("slider-svm-parameter-gamma-coef", "marks"),
    [Input("slider-svm-parameter-gamma-power", "value")],
)
def update_slider_svm_parameter_gamma_coef(power):
    scale = 10 ** power
    return {i: str(round(i * scale, 8)) for i in range(1, 10, 2)}


@app.callback(
    Output("slider-svm-parameter-C-coef", "marks"),
    [Input("slider-svm-parameter-C-power", "value")],
)
def update_slider_svm_parameter_C_coef(power):
    scale = 10 ** power
    return {i: str(round(i * scale, 8)) for i in range(1, 10, 2)}

@app.callback(
    Output("slider-threshold", "value"),
    [Input("button-zero-threshold", "n_clicks")],
    [State("graph-sklearn-svm", "figure")],
)
def reset_threshold_center(n_clicks, figure):
    if n_clicks:
        Z = np.array(figure["data"][0]["z"])
        value = -Z.min() / (Z.max() - Z.min())
    else:
        value = 0.4959986285375595
    return value


# Disable Sliders if kernel not in the given list
@app.callback(
    Output("slider-svm-parameter-degree", "disabled"),
    [Input("dropdown-svm-parameter-kernel", "value")],
)
def disable_slider_param_degree(kernel):
    return kernel != "poly"


@app.callback(
    Output("slider-svm-parameter-gamma-coef", "disabled"),
    [Input("dropdown-svm-parameter-kernel", "value")],
)
def disable_slider_param_gamma_coef(kernel):
    return kernel not in ["rbf", "poly", "sigmoid"]


@app.callback(
    Output("slider-svm-parameter-gamma-power", "disabled"),
    [Input("dropdown-svm-parameter-kernel", "value")],
)
def disable_slider_param_gamma_power(kernel):
    return kernel not in ["rbf", "poly", "sigmoid"]

@app.callback(
    Output("dummy", "id"),
    [Input('query-data-button', "n_clicks"),
     Input("dropdown-select-device", 'value'),
     Input("dropdown-select-process", 'value'),
     Input('sort-date-picker-range', 'start_date'),
     Input('sort-date-picker-range', 'end_date'),
     Input('fmax-token-select', 'value'),
     Input('sicc-token-select', 'value'),
     Input('fmax-operation-select', 'value'),
     Input('sicc-operation-select', 'value')]
)
def query_data(n_clicks, devices, process, start_date, end_date, fmax_token, sicc_token, fmax_op, sicc_op):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    print(changed_id)
    if 'query-data-button' not in changed_id:
        print("not querying")
        return "dummpy2"

    data = sortparam_dataset.load_sort_parametric(sort_param_dir, devices, start_date, end_date, fmax_token, fmax_op, sicc_token, sicc_op)
    print(data.head())
    return "dummy"

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
    Output("div-graphs", "children"),
    [
        Input("dropdown-svm-parameter-kernel", "value"),
        Input("slider-svm-parameter-degree", "value"),
        Input("slider-svm-parameter-C-coef", "value"),
        Input("slider-svm-parameter-C-power", "value"),
        Input("slider-svm-parameter-gamma-coef", "value"),
        Input("slider-svm-parameter-gamma-power", "value"),
        Input("dropdown-select-dataset", "value"),
        Input("slider-dataset-noise-level", "value"),
        Input("radio-svm-parameter-shrinking", "value"),
        Input("slider-threshold", "value"),
        Input("slider-dataset-sample-size", "value"),
    ],
)
def update_svm_graph(
    kernel,
    degree,
    C_coef,
    C_power,
    gamma_coef,
    gamma_power,
    dataset,
    noise,
    shrinking,
    threshold,
    sample_size,
):
    t_start = time.time()
    h = 0.3  # step size in the mesh

    # Data Pre-processing
    X, y = generate_data(n_samples=sample_size, dataset=dataset, noise=noise)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    x_min = X[:, 0].min() - 0.5
    x_max = X[:, 0].max() + 0.5
    y_min = X[:, 1].min() - 0.5
    y_max = X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    C = C_coef * 10 ** C_power
    gamma = gamma_coef * 10 ** gamma_power

    if shrinking == "True":
        flag = True
    else:
        flag = False

    # Train SVM
    clf = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, shrinking=flag)
    clf.fit(X_train, y_train)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    prediction_figure = figs.serve_prediction_plot(
        model=clf,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        Z=Z,
        xx=xx,
        yy=yy,
        mesh_step=h,
        threshold=threshold,
    )

    roc_figure = figs.serve_roc_curve(model=clf, X_test=X_test, y_test=y_test)

    confusion_figure = figs.serve_pie_confusion_matrix(
        model=clf, X_test=X_test, y_test=y_test, Z=Z, threshold=threshold
    )

    return [
        html.Div(
            id="svm-graph-container",
            children=dcc.Loading(
                className="graph-wrapper",
                children=dcc.Graph(id="graph-sklearn-svm", figure=prediction_figure),
                style={"display": "none"},
            ),
        ),
        html.Div(
            id="graphs-container",
            children=[
                dcc.Loading(
                    className="graph-wrapper",
                    children=dcc.Graph(id="graph-line-roc-curve", figure=roc_figure),
                ),
                dcc.Loading(
                    className="graph-wrapper",
                    children=dcc.Graph(
                        id="graph-pie-confusion-matrix", figure=confusion_figure
                    ),
                ),
            ],
        ),
    ]


# Running the server
if __name__ == "__main__":
    app.run_server(debug=True)
