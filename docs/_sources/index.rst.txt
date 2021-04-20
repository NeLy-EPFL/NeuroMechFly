.. NEUROMECHFLY documentation master file, created by
   sphinx-quickstart on Thu Nov 21 19:42:22 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Neuromechfly's documentation!
===========================================

.. warning::  Documentation is work in progress!!

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   NeuroMechFly
   scripts
   config
   data


NeuroMechFly
============

..
   Functions for running NeuroMechFly simulation

   Installation
   ------------

   To install NeuroMechFly, run the following commands in the command line:

   .. code:: bash

      $ conda create -n neuromechfly python=3.6 numpy Cython
      $ git clone https://github.com/NeLy-EPFL/NeuroMechFly.git
      $ pip install -e .

   Running the experiments
   -----------------------

   Before running the scripts please make sure that you activated the conda
   environment by running ``$ conda activate neuromechfly`` on the command
   line.

   **1. Kinematic Matching** - Make sure in */data* folder, there are
   */walking/df3d* and *grooming/df3d* folders that contain a .pkl file
   starting with “joint_angles_..”. - Navigate to */scripts/KM* folder. -
   Run ``$ python kinematicMatching_noSelfCollisions.py`` for simulating
   the locomotion behavior. - Change the behavior in ``main()``\ to
   “grooming” to simulate the grooming behavior. Note: Collisions are
   disabled in this script, but use this for now because
   *kinematicMatching.py* needs to be modified for the new scale.

   **2. Optimization** - To simulate the results of the last evolution, run
   ``$ python drosophila_simulation_opt.py``. This script will run the best
   result from *FUN.ged3* and *VAR.ged3*. - To formulate the objective
   functions and run the evolution, run
   ``$ python drosophila_evolution.py``. This is where the jMetal framework
   is set.

   Miscellaneous
   -------------

   -  To see the CPG network, navigate to */scripts/Network* and run
      ``$ python locomotion.py``


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
