FROM mambaorg/micromamba:0.24.0 as conda

USER root

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git pip swig cmake libpython3.8 golang && \
    rm -rf /var/lib/apt/lists/*

## Speed up the build, and avoid unnecessary writes to disk
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV PIPENV_VENV_IN_PROJECT=true PIP_NO_CACHE_DIR=false PIP_DISABLE_PIP_VERSION_CHECK=1

COPY --chown=$MAMBA_USER:$MAMBA_USER docker/environment_local.yml /tmp/environment.yml

# Take token arguments and set them as environment variables (needed to clone jumanji-internal)
ARG GITHUB_USER_TOKEN
ARG GITHUB_ACCESS_TOKEN
ENV GITHUB_USER_TOKEN=$GITHUB_USER_TOKEN
ENV GITHUB_ACCESS_TOKEN=$GITHUB_ACCESS_TOKEN

RUN micromamba create -y --file /tmp/environment.yml \
    && micromamba clean --all --yes \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete

FROM debian:bullseye-slim as test-image

# Let's have git installed in the container.
RUN apt update
RUN apt install -y git curl

COPY --from=conda /opt/conda/envs/. /opt/conda/envs/
ENV PATH=/opt/conda/envs/popcorn/bin/:$PATH APP_FOLDER=/app
ENV PYTHONPATH=$APP_FOLDER:$PYTHONPATH
ENV TZ=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR $APP_FOLDER

ARG USER_ID=1000
ARG GROUP_ID=1000
ENV USER=eng
ENV GROUP=eng
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/envs/popcorn/lib

COPY . $APP_FOLDER
RUN pip install --user -e ./

FROM test-image as run-image

# N.B: Had to relink this library so that TensorFlow has access to IFUNC symbol clock_gettime
# when cpu or tpu accelerator used.
ENV LD_PRELOAD="${LD_PRELOAD}:/lib/x86_64-linux-gnu/librt.so.1"

# The run-image (default) is the same as the dev-image with the some files directly
# copied inside
