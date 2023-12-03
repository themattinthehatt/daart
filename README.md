# daart: deep learning for animal action recognition toolbox

[![Documentation Status](https://readthedocs.org/projects/daart/badge/?version=latest)](https://daart.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/334987729.svg)](https://zenodo.org/badge/latestdoi/334987729)

A collection of tools for the discrete classification of animal behaviors using low-dimensional representations of videos (such as skeletons provided by tracking algorithms). Our approach combines strong supervision, weak supervision, and self-supervision to improve model performance. See the preprint [here](https://www.biorxiv.org/content/10.1101/2021.06.16.448685v1) for more details. 

This repo currently supports fitting the following types of base models on behavioral time series data:
* Dense MLP network with initial 1D convolutional layer
* RNNs - both LSTMs and GRUs
* Temporal Convolutional Networks (TCNs)

See the [documentation](https://daart.readthedocs.io/) to get started!

If you use daart in your analysis of behavioral data, please cite our preprint!

    @inproceedings{whiteway2021semi,
      title={Semi-supervised sequence modeling for improved behavioral segmentation},
      author={Whiteway, Matthew R and Schaffer, Evan S and Wu, Anqi and Buchanan, E Kelly and Onder, Omer F and Mishra, Neeli and Paninski, Liam},
      journal={bioRxiv},
      year={2021},
      publisher={Cold Spring Harbor Laboratory}
    }
