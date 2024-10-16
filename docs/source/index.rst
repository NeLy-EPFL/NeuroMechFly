.. neuromechfly documentation master file, created by
   sphinx-quickstart on Wed May 12 09:24:55 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

NeuroMechFly
============

.. warning::
   
   |
   
   **IMPORTANT INFORMATION**

   This website contains documentation for legacy code related to `Lobato-Rios et al, Nature Methods, 2022 <https://doi.org/10.1038/s41592-022-01466-7>`__.

   **NeuroMechFly has since been updated, and this website is no longer actively maintained.**
   
   **For most up-to-date information, please visit** `neuromechfly.org <https://neuromechfly.org/>`__.
   
   |


.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
.. image:: https://badge.fury.io/gh/tterb%2FHyde.svg

**NeuroMechFly** is a data-driven computational simulation of adult
*Drosophila melanogaster* designed to synthesize rapidly growing
experimental datasets and to test theories of neuromechanical behavioral
control. For the technical background and details, please refer to our
`paper <https://www.biorxiv.org/content/10.1101/2021.04.17.440214v1>`__.


.. only:: html

   .. figure:: ../images/NeuroMechFly.gif

If you use NeuroMechFly in your research, you can cite us:

.. code:: latex

   @article {Lobato-Rios2021.04.17.440214,
       author = {Lobato-Rios, Victor and {\"O}zdil, Pembe Gizem and Ramalingasetty, Shravan Tata and Arreguit, Jonathan and Clerc Rosset, St{\'e}phanie and Knott, Graham and Ijspeert, Auke Jan and Ramdya, Pavan},
       title = {NeuroMechFly, a neuromechanical model of adult Drosophila melanogaster},
       elocation-id = {2021.04.17.440214},
       year = {2021},
       doi = {10.1101/2021.04.17.440214},
       publisher = {Cold Spring Harbor Laboratory},
       URL = {https://www.biorxiv.org/content/early/2021/04/18/2021.04.17.440214},
       eprint = {https://www.biorxiv.org/content/early/2021/04/18/2021.04.17.440214.full.pdf},
       journal = {bioRxiv}
   }

Installation and Getting started
================================

.. toctree::
   :maxdepth: 2

   installation

Reproducing the experiments
===========================

.. toctree::
   :maxdepth: 2

   replication
   

Customizing the biomechanical model
===================================

.. toctree::
   :maxdepth: 2

   biomechanical_tutorial.rst
   
Changing the neural controller
==============================

.. toctree::
   :maxdepth: 2

   controller_tutorial.rst
   
Changing the simulation parameters
==================================

.. toctree::
   :maxdepth: 2

   environment_tutorial.rst
   
Changing the muscle model
=========================

.. toctree::
   :maxdepth: 2

   muscles_tutorial.rst
   
Miscellaneous
=============

.. toctree::
   :maxdepth: 2

   misc.rst

Documentation
==============

.. toctree::
   :maxdepth: 2

   reference/index
