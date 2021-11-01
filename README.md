# daart: deep learning for animal action recognition toolbox
A collection of tools for the discrete classification of animal behaviors using low-dimensional representations of videos (such as skeletons provided by tracking algorithms). Our approach combines strong supervision, weak supervision, and self-supervision to improve model performance. See the preprint [here](https://www.biorxiv.org/content/10.1101/2021.06.16.448685v1) for more details. This repo currently supports fitting the 
following types of base models on behavioral time series data:
* Dense MLP network with initial 1D convolutional layer
* RNNs - both LSTMs and GRUs
* Temporal Convolutional Networks (TCNs)


## Installation

First you'll have to install the `git` package in order to access the code on github. Follow the 
directions [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) 
for your specific OS.
Then, in the command line, navigate to where you'd like to install the `daart` package and move 
into that directory:
```
$: git clone https://github.com/themattinthehatt/daart
$: cd daart
```

Next, follow the directions 
[here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) 
to install the `conda` package for managing development environments.
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

## Getting started

To fit models from the command line using [test-tube](https://williamfalcon.github.io/test-tube/) 
for hyperparameter searching and model fitting, see `fit_models.py` in the `examples` directory.
This script fits one or more models based on three yaml configuration files: one describing the 
data, one describing the model, and one describing the training procedure. Example configuration
files can be found in the `configs` directory.

**_Note:_** Test-tube will automatically perform a hyperparameter search over any field that is 
provided as a list; for example, in the `model.yaml` file, change `n_hid_layers: 1` to 
`n_hid_layers: [1, 2, 3]` to search over the number of hidden layers in the model.
 
Once you have set the desired parameters in these files (see comment on data paths below), you can 
then fit models like so:

```
(daart) $: python fit_models.py --data_config /path/to/data.yaml 
       --model_config /path/to/model.yaml --train_config /path/to/train.yaml
```

#### Data paths

The `data.yaml` file has a field for listing experiment/session/video ids (`expt_ids`), as well as 
a `data_dir` field. The `fit_models.py` script assumes data is stored in the following way, though 
this can easily be adapted by changing the appropriate lines in a copy of the `fit_example.py` 
script:

* markers: `data_dir/markers/[expt_id]_labeled.csv` or `data_dir/markers/[expt_id]_labeled.h5`; 
the standard file formats used by DLC/DGP are currently supported.
 
* hand labels: `data_dir/labels-hand/[expt_id]_labels.csv`; a binary matrix of shape 
`(T, n_classes + 1)`, where the first column represents the `background` class; the gradients 
contributed by these time points are zeroed out during training.

* heuristic labels: `data_dir/labels-heuristic/[expt_id]_labels.csv`; same format as the hand 
labels

See the directory `daart/data` for example fly data used in the preprint.
