.. _user_guide_configs:

#######################
The configuration files
#######################

Users interact with daart through a set of configuration (yaml) files.
These files point to the data directories, define the type of model to fit, and specify a wide
range of hyperparameters.

An example set of configuration files can be found
`here <https://github.com/themattinthehatt/daart/tree/main/data/configs>`_.
When training a model on a new dataset, you must copy/paste these templates onto your local
machine and update the arguments to match your data.

There are three configuration files:

* :ref:`data <config_data>`: where data is stored and model input type
* :ref:`model <config_model>`: model class and various network hyperparameters
* :ref:`train <config_train>`: training epochs, batch size, etc.

The sections below describe the most important parameters in each file;
see the example configs for all possible options.

.. _config_data:

Data
====

* **input_type**: name of directory containing input data: 'markers' | 'features' | ...
* **expt_ids**: list of experiment ids used for training the model
* **ignore_class**: specifies index of the column in hand/heuristic label files that should be ignored when computing the loss function. 1s in this column mean "this frame has not been scored"; if every frame has been scored, set this to a negative value like -100.
* **weight_classes**: false to weight each class equally in loss function; true to weight each class inversely proportional to its frequency
* **data_dir**: absolute path to directory that contains the data
* **results_dir**: absolute path to directory that stores model fitting results

.. _config_model:

Model
=====

* **labmda_weak**: weight on heuristic/pseudo label classification loss
* **lambda_strong**: weight on hand label classification loss (can always leave this as 1)
* **lambda_recon**: weight on input reconstruction loss
* **lambda_pred**: weight on next-step-ahead prediction loss

So, for example, to fit a fully supervised classification model, set ``lambda_strong: 1`` and
all other "lambda" options to 0.

To fit a model that uses heuristic labels, set ``lambda_strong: 1``, ``lambda_weak: 1``, and
all other "lambda" options to 0. You can try several values of ``lambda_weak`` to see what works
best for your data.

.. _config_train:

Train
=====

* **min/max_epochs**: control length of training
* **enable_early_stop**: exit training early if validation loss begins to increase
* **trial_splits**: fraction of data to use for train;val;test;gap; you can always set "gap" to 0 as long as you validate your model on completely held-out videos
