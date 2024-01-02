Installation
============

First you'll have to install the ``git`` package in order to access the code on github.
Follow the
`git installion instructions <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`_
for your specific OS.

**Install ffmpeg**

First, check to see you have ``ffmpeg`` installed by typing the following into the terminal:

.. code-block:: console

    ffmpeg -version

If not, install:

.. code-block:: console

    sudo apt install ffmpeg

**Set up a conda environment**

Next, follow the
`conda installation instructions <https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`_
to install the ``conda`` package for managing development environments.
Then, create a conda environment:

.. code-block:: console

    conda create --name daart python=3.6

Active the new environment:

.. code-block:: console

    conda activate daart

**Install daart**

.. note::

    Make sure your conda environment is activated during the following steps.

1. Move into the directory where you want to place the repository folder, and download it from
   github.

   .. code-block:: console

       git clone https://github.com/themattinthehatt/daart

2. Move into the newly created folder:

   .. code-block:: console

       cd daart

   and install the package and all dependencies:

   .. code-block::

       pip install -e .

3. Verify that all the unit tests are passing on your machine by running

   .. code-block:: console

       pytest

   The tests will take a few minutes to run. You should not see any fails; warnings are ok.

4. To be able to use this environment for jupyter notebooks:

   .. code-block:: console

       python -m ipykernel install --user --name daart
