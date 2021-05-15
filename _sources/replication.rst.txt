Replicating our results
-----------------------

**Note:** before running the following scripts, please be sure to
activate the virtual environment (see the `installation
guide <docs/installation.md>`__)

NeuroMechFly is run in
`PyBullet <https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet>`__.
In the Graphical User Interface, you can use the following keyboard and
mouse combinations to control the camera’s viewpoint: - ALT/CONTROL &
Left Mouse Button: Rotate - ALT/CONTROL & Scroll Mouse Button: Pan -
Scroll Mouse Button: Zoom

**1. Kinematic replay**

.. only:: html

   .. figure:: ../images/km_walking.gif

   .. figure:: ../images/km_grooming.gif

Run the following commands on the terminal to reproduce the kinematic
replay experiments: - ``$ run_kinematic_replay --behavior walking`` for
locomotion on the spherical treadmill. To simulate foreleg/antennal
grooming, change ``walking`` at the end of the command to ``grooming``.
**Note:** Locomotion begins ~2.5 seconds into the simulation. Until
then, the fly stands still.

-  ``$ run_kinematic_replay_ground --perturbation`` to simulate
   locomotion on the ground with perturbations enabled. Remove
   ``--perturbation`` to disable perturbations. To change the behavior
   to grooming, append ``--behavior grooming`` to the command.

.. only:: html

   .. figure:: ../images/perturbation.gif


**NOTE:** At the end of each simulation run, a folder called
\*kinematic_replay\_\_\* containing the physical quantities (joint
angles, torques etc.) will be created under the
*scripts/kinematic_replay* folder.

**NOTE:** To obtain new pose estimates from the `DeepFly3D
Database <https://dataverse.harvard.edu/dataverse/DeepFly3D>`__, please
refer to `DeepFly3D
repository <https://github.com/NeLy-EPFL/DeepFly3D>`__. After running
the pose estimator on the recordings, you can follow the instructions
for computing joint angles to control NeuroMechFly
`here. <https://github.com/NeLy-EPFL/NeuroMechFly/blob/km-refactor/docs/angleprocessing.md>`__

--------------

**2. Gait optimization**

.. only:: html

   .. figure:: ../images/optimization.gif

Run the following commands on the terminal to reproduce the locomotor
gait optimization experiments: - ``$ run_neuromuscular_control`` to run
the latest generation of the last optimization run. By default, this
script will read and run the files *FUN.txt* and *VAR.txt* under the
*scripts/neuromuscular_optimization/* folder. To run different files,
simply run
``$ run_neuromuscular_control -p <'path-of-the-optimization-results'> -g <'generation-number'> -s <'solution-type'>``
(solution type being fastest, medium, slowest, or a specific index).
**The results path should be relative to the scripts folder.** To see
the results that are already provided, go to the folder
*scripts/neuromuscular_optimization/* and run:
``$ run_neuromuscular_control  -p optimization_results/run_Drosophila_example/ -g 50``.

**NOTE:** At the end of each simulation run, a folder named according to
the chosen optimization run will be created under the
*scripts/neuromuscular_optimization* folder which contains the network
parameters and physical quantities.

-  ``$ run_multiobj_optimization`` to run locomotor gait optimization
   from scratch. This script will create new files named *FUN.txt* and
   *VAR.txt* as well as a new folder containing the results from each
   generation in a folder named *optimization_results*. After
   optimization has completed, run ``$ run_neuromuscular_control`` to
   visualize the results from the last generation. To see different
   generations, follow the instructions above and select a different
   file.

**NOTE:** Optimization results will be stored under
*scripts/neuromuscular_optimization/optimization_results* inside a
folder named according to the chosen optimization run.

**NOTE:** The code and results in this repository are improved compared
with the results in our original
`paper <https://www.biorxiv.org/content/10.1101/2021.04.17.440214v1>`__.

**NOTE:** To formulate new objective functions and penalties, please
refer to the *NeuroMechFly/experiments/network_optimization*.

--------------

**3. Sensitivity Analysis**

-  First, download the simulation data pertaining to the sensitivity
   analyses from
   `here <https://drive.google.com/drive/folders/1H0G3mdeKLyGkS1DYxbOeOCXgywJmwfs9?usp=sharing>`__
   and place these files in the folder, *data/sensitivity_analysis*
-  To reproduce the sensitivity analysis figures,
   ``$ run_sensitivity_analysis``. Make sure that the downloaded files
   are in the correct location.
