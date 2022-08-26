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
Download zip:
https://github.com/intel-innersource/applications.manufacturing.intel.yield.pyuber/archive/refs/heads/master.zip

```bash
conda create -f env.yaml
pip install <your-path-to>/Downloads/applications.manufacturing.intel.yield.pyuber-master.zip
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
