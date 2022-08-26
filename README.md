# applications.ai.appliedml.workflow.waferinsights

Wafer insights is a python application that allows users to predict FMAX/IDV tokens based on multiple data sources measured in fab.

## Installation

The environment can be created by using the env.yaml file in the root directory.

```bash
conda create -f env.yaml
```

## Usage

To start the dashboard
```bash
conda activate WI
cd dashboard
python app.py
```

To run loaders to update local caches from production database:
```bash
python <loader_name>/loader.py
```

## External Dependencies
