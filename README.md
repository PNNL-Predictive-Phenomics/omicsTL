# omicsTL

## Environment Setup

### Docker

The easiest way to start with the package is to use the Docker container.

```bash
docker build . -t omicstl
docker run -it omicstl /bin/bash
```

### Dev Container

If you are using Visual Studio Code, there is also a Dev Container config included for use with the Dev Containers extension.
To activate it, press Ctrl+Shift+P/Cmd+Shift+P and select `Dev Containers: Reopen in Container`.

### Local Setup

Local setup is not recommended due to the tight dependencies required by the package, but if you need to run it locally, make sure the following are installed:

```
Python == 3.12
  setuptools
  wheel

R >= 4.2.0
  BiocManager==3.20
```

Note that the package will install a number of additional packages which may not be compatible with other packages you have installed.
For this reason, you will likely want to run this inside a virtual environment for Python and use a custom library path for R using the R_LIBS_USER environment variable.

## Installation

The package can be installed by cloning the repository, navigating to the cloned directory, and installing via pip:

```bash
# Install package
pip install .

# Install in development mode
pip install -e .
```

## Vignettes

A vignette on simulating data and how to fit each machine learning model is located in docs/example.ipynb


----------------------------------------------------------------------
Disclaimer

This material was prepared as an account of work sponsored by an agency of the United States Government.  Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any of their employees, nor any jurisdiction or organization that has cooperated in the development of these materials, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process disclosed, or represents that its use would not infringe privately owned rights.
Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.

PACIFIC NORTHWEST NATIONAL LABORATORY 

operated by 

BATTELLE 

for the 

UNITED STATES DEPARTMENT OF ENERGY 

under Contract DE-AC05-76RL01830 
