# applications.ai.appliedml.workflow.waferinsights

Wafer insights is a python application that allows users to predict FMAX/IDV tokens based on multiple data sources measured in fab.

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

http://0.0.0.0:8050/


