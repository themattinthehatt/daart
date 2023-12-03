.. _user_guide_training:

########
Training
########

daart provides several tools for training models:

1. A set of high-level functions used for creating data loaders, models, trainers, etc.
   You can combine these to create your own custom training script.
2. An `example training script <https://github.com/themattinthehatt/daart/blob/main/examples/fit_models.py>`_
   that demonstrates how to combine the high-level functions for model training and evaluation.
   This is a complete training script and you may simply use it as-is.

Additionally, daart uses the `test-tube <https://williamfalcon.github.io/test-tube/>`_ package
for hyperparameter searching and model fitting.
``test-tube`` will automatically perform a hyperparameter search over any field that is provided as
a list;
for example, in the ``model.yaml`` file, change ``n_hid_layers: 1`` to ``n_hid_layers: [1, 2, 3]``
to search over the number of hidden layers in the model.

Once you have set the desired parameters in the :ref:`configuration files <user_guide_configs>`
(make sure to update the data paths!), move to the directory where your copy of ``fit_models.py``
is stored and run the following from the terminal:

.. code-block:: console

    python fit_models.py --data_config /path/to/data.yaml --model_config /path/to/model.yaml --train_config /path/to/train.yaml

You will see configuration details printed in the terminal, followed by a training progress bar.
Upon training completion the model will be saved in the location specified in the data config.
