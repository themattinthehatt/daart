.. _user_guide_inference:

#########
Inference
#########

Once you have trained a model you'll likely want to run inference on new videos.

Similar to training, there are a set of high-level functions used to perform inference and evaluate
performance; this page details some of the main steps.


Load model
==========

Using a provided model directory, construct a model and load the weights.

.. code-block:: python

    import os
    import torch
    import yaml

    from daart.models import Segmenter

    model_dir = /path/to/model_dir
    model_file = os.path.join(model_dir, 'best_val_model.pt')

    hparams_file = os.path.join(model_dir, 'hparams.yaml')
    hparams = yaml.safe_load(open(hparams_file, 'rb'))

    model = Segmenter(hparams)
    model.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))
    model.to(hparams['device'])
    model.eval()

Build data generator
====================

To run inference on a new session, you must provide a csv file that contains markers or features
from a new session (you must use the same type of inputs the model was trained on).

.. code-block:: python

    from daart.data import DataGenerator
    from daart.transforms import ZScore

    sess_id = <name_of_session>
    input_file = /path/to/markers_or_features_csv

    # define data generator signals
    signals = ['markers']  # same for markers or features
    transforms = [ZScore()]
    paths = [input_file]

    # build data generator
    data_gen_test = DataGenerator(
        [sess_id], [signals], [transforms], [paths], device=hparams['device'],
        sequence_length=hparams['sequence_length'], batch_size=hparams['batch_size'],
        trial_splits=hparams['trial_splits'],
        sequence_pad=hparams['sequence_pad'], input_type=hparams['input_type'],
    )

Run inference
=============

Inference can be performed by passing the newly constructed data generator to the model's
``predict_labels`` method:

.. code-block:: python

    import numpy as np

    # predict probabilities from model
    print('computing states for %s...' % sess_id, end='')
    tmp = model.predict_labels(data_gen_test, return_scores=True)
    probs = np.vstack(tmp['labels'][0])
    print('done')

    # get discrete state by taking argmax over probabilities at each time point
    states = np.argmax(probs, axis=1)

