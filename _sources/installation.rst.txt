Installation
============

To avoid any conflicts of python packages with your existing python
environment, we highly recommend to use virtualenv or conda env. To
create a conda environment, follow the following steps:

**First, make sure that you have git-lfs (large file storage) installed
on your local machine. Otherwise, please refer to
this**\ `link <https://github.com/git-lfs/git-lfs>`__\ **to learn more
about how to install the Git LFS.**

Then, you can download the repository on your local machine by running
the following line in the terminal:

.. code:: bash

   $ git clone https://github.com/NeLy-EPFL/NeuroMechFly.git

After the download is complete, navigate to the NeuroMechFly folder:

.. code:: bash

   $ cd NeuroMechFly

In this folder, run the following commands to create a virtual
environment and activate it:

.. code:: bash

   $ conda create -n neuromechfly python=3.6 numpy Cython shapely
   $ conda activate neuromechfly

First, install the FARMS Container by running:

.. code:: bash

   $ pip install git+https://gitlab.com/FARMSIM/farms_container.git

Finally, install all the dependencies by running:

.. code:: bash

   $ pip install -e .

Once you complete all the steps, NeuroMechFly is ready to use!

Alternatively, you can use virtualenv. For instructions on how to setup
and use virtual environments please refer to
`Virtualenv <https://realpython.com/python-virtual-environments-a-primer>`__.

After setting up your virtualenv, to install and use the NeuroMechFly
library follow the abovementioned procedure in your active python
environment.

Angle Processing (Obtaining angles from pose estimation)
========================================================

An interesting application for NeuroMechFly is what we call **kinematic
replay**, i.e., reproducing real fly movements with our model. This can
be done by feeding the model with the joint angles calculated from 3D
pose estimations. Any method can be used for obtaining such estimations.

We used `DeepFly3D <https://github.com/NeLy-EPFL/DeepFly3D>`__ data and
software for tracking all leg joints in 3D. Then, we obtained the joint
angles from the estimated poses using the package
`df3dPostProcessing <https://github.com/NeLy-EPFL/df3dPostProcessing>`__
(included in the NeuroMechFly installation).

We provide the `pose estimation
results <https://github.com/NeLy-EPFL/NeuroMechFly/tree/master/data/joint_tracking>`__
from two experiments including walking and grooming behaviors. We
interpolated the original data because it was recorded at 100 fps and
our simulation runs with a time step of 1ms. Joint angles can be
calculated as follows:

.. code:: python

   from df3dPostProcessing import df3dPostProcess

   exp = 'path/to/NeuroMechFly/data/joint_tracking/walking or grooming/df3d/pose_result*.pkl'

   # Read pose results and calculate 3d positions from 2d estimations
   df3d = df3dPostProcess(exp, calculate_3d=True)

   # Align and scale 3d positions using the NeuroMechFly skeleton as template, data is interpolated
   align = df3d.align_to_template(interpolate=True)

   # Calculate joint angles from the leg (ThC-3DOF, CTr-2DOF, FTi-1DOF, TiTa-1DOF)
   angles = df3d.calculate_leg_angles()

Alternately, we provide the joint angles for both behaviors which were
calculated as shown above. This files are used when running the scripts
*run_kinematic replay* and *run_kinematic_replay_ground*.
