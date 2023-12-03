.. daart documentation master file, created by
   sphinx-quickstart on Wed Nov 29 20:41:57 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to daart's documentation!
=================================

The ``daart`` package is a collection of tools for the discrete classification of animal behaviors
using low-dimensional representations of videos (such as skeletons provided by tracking algorithms).
``daart`` combines strong supervision, weak supervision, and self-supervision to improve model
performance.
See the `preprint <https://www.biorxiv.org/content/10.1101/2021.06.16.448685v1>`_ for more details.

This repo currently supports fitting the following types of base models on behavioral time series
data:

* dense MLP network with initial 1D convolutional layer
* RNNs - both LSTMs and GRUs
* temporal convolutional networks (TCNs)

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   source/installation
   source/user_guide
   source/api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
