# inference_modules for ANNETTE


<div align="center">
  <img src="./_img/eml_logo_and_text.png", width="500">
</div>

## Description

This repository contains tools necessary to optimize, execute and process neural networks for ANNETTE.

## Structure

The folder hw_modules contains inference_modules for supported platforms. Each supported platform has an inference and a parser script.
The inference script contains necessary functions to optimize and run neural networks. The parser script contains functions to process data gatherd during the neural network execution.

## Installation

The inference_modules can be installed as a pip package. It is recommended to work with a Python virtual environment.

```console
git clone git@github.com:embedded-machine-learning/inference_modules.git
cd ./inference_modules/

python3 -m venv venv_infmod
source venv_infmod/bin/activate
pip3 install --upgrade pip setuptools

pip3 install -e .
```

## Versions

The modules were tested on Ubuntu 18.04 and 20.04, with Python3.6 and Python3.8

## Requirements

Different parts of the repository work with different pip packages. To avoid filling the virtual environments with too many packages, the requirements can be installed separately. 