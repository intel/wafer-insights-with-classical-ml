FROM intel/oneapi-aikit:devel-ubuntu20.04 AS WaferInsights
RUN conda install dash
ADD . /opt/Dashboard
