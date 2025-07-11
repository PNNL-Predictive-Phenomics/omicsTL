FROM ubuntu:noble

ARG DEBIAN_FRONTEND=noninteractive

# Everything below make is for rpy2 to function. Id love to ditch it at some point, but here we are
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    libpython3.12-dev \
    python3.12-venv \
    python-is-python3 \
    git \
    vim \
    pandoc \
    make \
    gfortran \
    build-essential \
    gcc \
    g++ \
    cmake \
    libpcre2-dev \
    liblzma-dev \
    libbz2-dev \
    zlib1g-dev \
    libicu-dev \
    libblas-dev \
    liblapack-dev \
    libtirpc-dev \
    libzstd-dev \
    libcurl4-openssl-dev \
    libfontconfig1-dev \
    libxml2-dev \
    libssl-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    software-properties-common \
    dirmngr \
    wget \
    && apt-get clean

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && update-alternatives --set python /usr/bin/python3.12

RUN wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc && \
    add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/" && \
    apt-get install -y --no-install-recommends r-base-core=4.4.3-1.2404.0 r-base-dev=4.4.3-1.2404.0 r-cran-devtools

ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python -m pip install --upgrade pip setuptools wheel pre-commit

COPY .pre-commit-config.yaml /workspaces/omicsTL/.pre-commit-config.yaml
WORKDIR /workspaces/omicsTL
RUN git init && PRE_COMMIT_HOME=/root/.cache/pre-commit pre-commit install-hooks && rm -r .git

COPY requirements.txt /workspaces/omicsTL/requirements.txt
COPY requirements-dev.txt /workspaces/omicsTL/requirements-dev.txt
RUN pip install -r requirements.txt -r requirements-dev.txt

RUN R -e 'install.packages(c("BiocManager", "ggplot2", "here", "Matrix", "data.table", "survival", "Rcpp", "readr"), quiet=TRUE)'
RUN R -e 'BiocManager::install(version = "3.20", ask=FALSE, force=TRUE)'
RUN R -e 'BiocManager::install("mvdalab", quiet=TRUE)'

COPY . /workspaces/omicsTL

RUN R -e 'devtools::install_local("/workspaces/omicsTL/src/omicstl/r/viRF_code/viRandomForests_1.0.tar.gz")'
