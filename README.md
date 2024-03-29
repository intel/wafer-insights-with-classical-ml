# DISCONTINUATION OF PROJECT #
This project will no longer be maintained by Intel.  
This project has been identified as having known security escapes.  
Intel has ceased development and contributions including, but not limited to, maintenance, bug fixes, new releases, or updates, to this project.  
Intel no longer accepts patches to this project.  
  

# T6: WaferInsights Workflow
## Overview
Wafer Insights is a python application that allows users to predict FMAX/IDV tokens based on multiple data sources measured in fab.

## How it Works
Wafer Insights is an interactive data-visualization web application based on Dash and Plotly. It includes 2 major components: a data loader, which generates synthetic fab data for visualization, and a dash app that provides an interface for users to analyze the data and gain insight. Dash is written on top of Plotly.js and React.js, providing an ideal framework for building and deploying data apps with customized user interfaces. The `src/dashboard` folder contains the code for the dash app and the `src/loaders` folder contains the code for the data loader.

## Get Started 

### **Prerequisites**

#### Dependencies
The following libraries are required before you get started:
1. Git
2. Anaconda/Miniconda
3. Docker
4. Python3

#### Download the repo
Clone [Wafer Insights](https://github.com/intel/wafer-insights-with-classical-ml) repository.
```
git clone https://github.com/intel/wafer-insights-with-classical-ml
cd wafer-insights-with-classical-ml
git checkout v1.0.0
```

#### Download the Dataset
Actual measurement data from the Intel fab cannot be shared with the public. Therefore, we provide a synthetic data loader to generate synthetic data using the `make_regression` function from the sklearn library, which has the following format:
| **Type**         | **Format** | **Rows** | **Columns** |
| ---------------- | ---------- | -------- | ----------- |
| Feature Dataset  | Parquet    | 25000    | 2000        |
| Response Dataset | Parquet    | 25000    | 1            |

Refer to [How to Run](#how-to-run) to construct the dataset
### **Docker**
Below setup and how-to-run sessions are for users who want to use the provided docker image.
For bare metal environment, please go to [Bare Metal](#bare-metal).
#### Setup 


##### Pull Docker Image
```
docker pull intel/ai-workflows:wafer-insights
```

##### Set Up Synthetic Data
```
docker run -a stdout \
  -v $(pwd):/workspace \
  --workdir /workspace/src/loaders/synthetic_loader \
  --privileged --init --rm -it \
  intel/ai-workflows:wafer-insights \
  conda run --no-capture-output -n WI python loader.py
```

#### How to Run 
(Optional) Export related proxy into docker environment.
```
export DOCKER_RUN_ENVS="-e ftp_proxy=${ftp_proxy} \
  -e FTP_PROXY=${FTP_PROXY} -e http_proxy=${http_proxy} \
  -e HTTP_PROXY=${HTTP_PROXY} -e https_proxy=${https_proxy} \
  -e HTTPS_PROXY=${HTTPS_PROXY} -e no_proxy=${no_proxy} \
  -e NO_PROXY=${NO_PROXY} -e socks_proxy=${socks_proxy} \
  -e SOCKS_PROXY=${SOCKS_PROXY}"
```
To run the pipeline, follow the below instructions outside of the docker instance. 
```
export OUTPUT_DIR=/output
```

```
docker run -a stdout $DOCKER_RUN_ENVS \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env PYTHONPATH=$PYTHONPATH:$PWD \
  --volume ${OUTPUT_DIR}:/output \
  --volume $(pwd):/workspace \
  --workdir /workspace \
  -p 8050:8050 \
  --privileged --init --rm -it \
  intel/ai-workflows:wafer-insights \
  conda run --no-capture-output -n WI python src/dashboard/app.py
```

### **Bare Metal** 
Below setup and how-to-run sessions are for users who want to use the bare metal environment.
For docker environment, please go to [Docker](#docker).
#### Setup 
First, set up the environment with conda using:
```
conda create -n WI 
conda activate WI
pip install dash scikit-learn pandas pyarrow colorlover
```
#### How to Run 
To generate synthetic data for testing from the root directory:
```
cd src/loaders/synthetic_loader
python loader.py
```
To run the dashboard:
```
export PYTHONPATH=$PYTHONPATH:$PWD
python src/dashboard/app.py
```
The default dashboard URL is: http://0.0.0.0:8050/

## Recommended Hardware 
The hardware below is recommended for use with this reference implementation.   
| **Name**  | Description                                          |
| --------- | ---------------------------------------------------- |
| CPU       | Intel(R) Xeon(R) Gold 6252N CPU @ 2.30GHz (96 vCPUs) |
| Free RAM  | 367 GiB/376 GiB                                      |
| Disk Size | 2 TB                                                 | 

**Note:  The code was developed and tested on a machine with this configuration. However, it may be sufficient to use a machine that is much less powerful than the recommended configuration.**

## Useful Resources
[Intel AI Analytics Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html)<br>
[View All Containers and Solutions 🡢](https://www.intel.com/content/www/us/en/developer/tools/software-catalog/containers.html)<br>

## Support
[Report Issue](https://community.intel.com/t5/Intel-Optimized-AI-Frameworks/bd-p/optimized-ai-frameworks)<br>
