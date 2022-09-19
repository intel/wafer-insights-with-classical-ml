# applications.ai.appliedml.workflow.waferinsights

Wafer insights is a python application that allows users to predict FMAX/IDV tokens based on multiple data sources measured in fab.

## Installation

The environment can be created by using the env.yaml file in the root directory.

Linux
Note: production data extract currently only supports PyUber.
```bash
conda create -f env.yaml
```

Windows
Download zip file from folliwng URL:
https://github.com/intel-innersource/applications.manufacturing.intel.yield.pyuber/archive/refs/heads/master.zip

```bash
conda create -f env.yaml
pip install <your-path-to>/Downloads/applications.manufacturing.intel.yield.pyuber-master.zip
```

## Bare Metal Usage

Environment Creation:

```bash
conda create -n WI scikit-learn pandas pyrrow
conda activate WI
pip install dash
````

To generate synthetic data for testing from root directory:

```bash
conda activate WI
python src/loaders/synthetic/loader/loader.py
```

To run the dashboard:

```bash
export PYTHONPATH=$PYTHONPATH:$PWD
conda activate WI
python src/dashboard/app.py
```

The default dashboard URL is:

http://127.0.0.1:8050/


