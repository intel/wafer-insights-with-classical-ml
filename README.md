# Wafer Insights with Classical ML

WaferInsights is a python application that allows users to predict FMAX/IDV tokens based on multiple data sources measured in fab.

## Table of Contents
- [Implementation Details](#implementation-details)
    - [Software Dependencies](#software-dependencies)
    - [Performance](#performance)
    - [Dataset](#dataset)
- [Getting Started](#getting-started)
- [Usage](#usage)
    - [Data Preprocessing](#data-preprocessing)
    - [Training](#training)
    - [Inference](#inference)
- [License](#license)

## Implementation Details 
 
### Software Dependencies
The software dependencies can be found in the `env.yaml` and `requirements.txt`.

### Performance 
Inference           |  Inference Transformed          | Pipeline Fit|
:-------------------------:|:-------------------------:|:-------------------------:
![](plots/inference.png)  |  ![](plots/inference_transform.png) | ![](plots/pipeline_fit.png)

They numbers shown in the plots are the timings for the different components with and without Intel optimizations, which can be produced with the benchmark script found in the `src` folder. Inference is the actual machine learning inference. Inference transformed is the data transforms associated with the inference pass. Pipeline fit are the data transform and machine learning model fit.

### Dataset
The actual measurement data from the fab are not allowed to be shared with the public, thus we provide a synthetic data loader to generate synthetic data using the `make_regression` function from sklearn, which has the following format:

| **Type**                 | **Format** | **Rows** | **Columns**|
| :---                     | :---       | :---      | :---   
| Feature Dataset          |  Parquet | 25000 | 2000
| Response Dataset         |  Parquet | 25000 | 1

The generated features and responses are saved separately in different folders under `data/synthetic_etest` and `data/synthetic_response`.


## Getting Started 
First, set up the environment with conda using 
```bash
conda create -n WI 
conda activate WI
pip install dash scikit-learn pandas pyarrow colorlover
````
Then pull the repo and get started to use it

```bash
git clone https://github.com/intel/wafer-insights-with-classical-ml.git
cd wafer-insights-with-classical-ml
```

## Usage
To generate synthetic data for testing from root directory:
```bash
cd src/loaders/synthetic/loader
python loader.py
```

To run the dashboard:

```bash
export PYTHONPATH=$PYTHONPATH:$PWD
python src/dashboard/app.py
```

The default dashboard URL is:
http://0.0.0.0:8050/


## License
[License](LICENSE)

