# daart: deep learning for animal action recognition toolbox
A collection of tools for analyzing behavioral data


## Installation

First you'll have to install the `git` package in order to access the code on github. Follow the directions [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) for your specific OS.
Then, in the command line, navigate to where you'd like to install the `daart` package and move into that directory:
```
$: git clone https://github.com/themattinthehatt/daart
$: cd daart
```

Next, follow the directions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) to install the `conda` package for managing development environments. 
Then, create a conda environment:

```
$: conda create --name=daart python=3.6
$: conda activate daart
(daart) $: pip install -r requirements.txt 
```

To make the package modules visible to the python interpreter, locally run pip 
install from inside the main `daart` directory:

```
(daart) $: pip install -e .
```

To be able to use this environment for jupyter notebooks:

```
(daart) $: python -m ipykernel install --user --name daart
```

## Set paths

Next, you should create a file in the `daart` package named `paths.py` that looks like the following:

```python
DATA_PATH = '/top/level/data/path/'
RESULTS_PATH = '/top/level/results/path/'
```

This file contains the local paths on your machine, and will not be synced with github.
