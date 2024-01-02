.. _user_guide_data:

####################
Organizing your data
####################

The daart models use a low-dimensional representation of behavioral videos (such as pose estimates)
to predict a set of discrete behavioral classes.
Heuristic labels may supplement hand labels and lead to better classification performance
(see the preprint referenced on the home page).
Each daart project will have a data directory that contains markers (or other features),
hand labels, and, if desired, heuristic labels.
See the
`example data <https://github.com/themattinthehatt/daart/tree/main/data>`_
directory which contains two sessions of the head-fixed fly experiment analyzed in the paper.

.. note::

    The base daart package does **not** contain tools for labeling data;
    we recommend the `DeepEthogram labeling GUI <https://github.com/jbohnslav/deepethogram>`_.

.. note::

    Currently the models perform multinomial classification, so that only a single behavior
    is predicted at each time step.

Data directory structure
------------------------

The data directory structure contains subdirectories for each data type.
At minimum, a supervised model requires model inputs (either ``markers`` or ``features``)
and model outputs (``labels-hand``).
Semi-supervised models may also require heuristic labels (``labels-heuristic``).

The example directory structure below shows the naming convention for the different data types.
For each data type the data must be separated by experimental session, and each session must have a
unique ID.

Multiple types of features can be stored, and the set of features desired for a particular model
can be specified in the :ref:`configuration files <user_guide_configs>`.
In the example below, there are two sets of features: ``features-base`` and ``features-simba``.
The naming convention is the same for both.

Videos are not required for fitting the daart models, but may be useful for downstream analysis.

.. code-block::

    data_directory
    ├── features-base
    │   ├── <sess_id_0>.csv
    │   └── <sess_id_1>.csv
    ├── features-simba
    │   ├── <sess_id_0>.csv
    │   └── <sess_id_1>.csv
    ├── labels-hand
    │   ├── <sess_id_0>.csv
    │   └── <sess_id_1>.csv
    ├── labels-heuristic
    │   ├── <sess_id_0>.csv
    │   └── <sess_id_1>.csv
    ├── markers
    │   ├── <sess_id_0>.csv
    │   └── <sess_id_1>.csv
    └── videos
        ├── <sess_id_0>.mp4
        └── <sess_id_1>.mp4


Data formats
------------

Each data type requires its own (quite general) format for use with the daart code.

Markers format
**************

The current code accepts either csv or h5 files that are output by DLC or Lightning Pose.
The csv files must look like the following:

.. list-table:: markers/<sess_id>.csv
   :widths: 25 25 25 25 25 25 25
   :header-rows: 3

   * - scorer
     - scorer_name
     - scorer_name
     - scorer_name
     - scorer_name
     - scorer_name
     - scorer_name
   * - bodyparts
     - bodypart 1
     - bodypart 1
     - bodypart 1
     - bodypart 2
     - bodypart 2
     - bodypart 2
   * - coords
     - x
     - y
     - likelihood
     - x
     - y
     - likelihood
   * - 0
     - 274.3
     - 184.5
     - 0.87
     - 23.4
     - 13.0
     - 0.99
   * - 1
     - 275.6
     - 183.0
     - 0.88
     - 23.3
     - 13.0
     - 0.99
   * - 2
     - 276.9
     - 182.5
     - 0.87
     - 23.3
     - 12.9
     - 0.99
   * - 3
     - 278.4
     - 181.0
     - 0.87
     - 23.4
     - 13.1
     - 0.99

Features format
***************

Features should also be stored in csv files, with a single header row giving the feature name for
each column. The first column denotes the frame number.

.. list-table:: features/<sess_id>.csv
   :widths: 10 25 25 25
   :header-rows: 1

   * -
     - feature0_name
     - feature1_name
     - feature2_name
   * - 0
     - 458.3
     - 0.12
     - 13.8
   * - 1
     - 500.2
     - 0.06
     - 14.7
   * - 2
     - 523.8
     - -0.06
     - 15.6
   * - 3
     - 567.4
     - -0.08
     - 16.5

Hand labels format
******************

The hand labels are stored in a csv file; the first (header) row denotes the behavior class names
(with the first column containing an empty cell).
The remaining rows contain the hand labels for each time point.
The first column denotes the frame number.
The second column denotes the "background" class, and for each row the entry should be 1 if
no other behavior is labeled at that time point, or 0 if at least one other behavior is labeled at
that time point.
The remaining columns correspond to the dataset-specific behavioral classes, and are binary as well
(0s and 1s).
There should only be a single "1" per row.

.. list-table:: labels-hand/<sess_id>.csv
   :widths: 10 25 25 25 25
   :header-rows: 1

   * -
     - background
     - behavior0_name
     - behavior1_name
     - behavior2_name
   * - 0
     - 1
     - 0
     - 0
     - 0
   * - 1
     - 1
     - 0
     - 0
     - 0
   * - 2
     - 0
     - 0
     - 0
     - 1
   * - 3
     - 0
     - 0
     - 0
     - 1

For a complete example see the csv files in the `example data <https://github.com/themattinthehatt/daart/tree/main/data>`_.

Heuristic labels format
***********************

Same format as the hand labels.
