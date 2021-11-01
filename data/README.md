# daart example data

The daart code uses a low-dimensional representation of behavioral videos (such as pose estimates)
to predict a set of discrete behavioral classes. Heuristic labels may supplement hand labels and
lead to better classification performance (see the preprint referenced on the home page of the 
repo). This directory contains example data (markers from pose estimation, hand labels, and 
heuristic labels) for two sessions of the head-fixed fly experiment analyzed in the preprint.

**Note**: Currently the models perform multinomial classification, so that only a single behavior is 
predicted for each timestep. We are working on support for predicting multiple behaviors per
timestep.
 
In order to fit the daart models, data must be in the following formats.

### markers format
The current code accepts either csv or h5 files that are output by DLC/DGP.

### hand labels format
The current code accepts a csv file with the first row containing an empty cell and then the name
of each behavioral class. The first column denotes the frame number. The second column denotes the
"background" class, and entries should be 1 (no other behavior is labeled) or 0 (at least one other
behavior is labeled). The remaining columns correspond to the dataset-specific behavioral classes,
and are binary as well (0s and 1s). 

|            | background | still | walk | front_groom | back_groom | abdomen_move |
| ---------- | ---------- | ----- | ---- | ----------- | ---------- | ------------ |
|          0 |          1 |     0 |    0 |           0 |          0 |            0 |
|          1 |          1 |     0 |    0 |           0 |          0 |            0 |
|          2 |          0 |     0 |    0 |           1 |          0 |            0 |
|          3 |          0 |     0 |    0 |           1 |          0 |            0 |

For a complete example see the csv files in the directories above. 
 
### heuristic labels format
Same format as the hand labels.
