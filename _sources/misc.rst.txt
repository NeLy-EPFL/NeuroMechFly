Miscellaneous
-------------

**1. Central Pattern Generator Controller** - To see the CPG network,
navigate to *data/locomotion_network/* and run
``$ python locomotion.py`` - Please refer to `FARMS
Network <https://gitlab.com/farmsim/farms_network>`__ to learn more
about how to design new neural network controllers.

--------------

**2. Blender Model** - To visualize the biomechanical model, first
install `Blender <https://www.blender.org/download/>`__. - After
installation, navigate to *data/design/blender* and open
``neuromechfly_full_model.blend`` with Blender.

--------------

**3. Reproducing the Figures** - All of the plotting functions used in
the paper can be found in
`NeuroMechFly/utils/plotting.py <NeuroMechFly/utils/plotting.py>`__.
Please refer to the docstrings provided in the code for the details
about how to plot your simulation data. - For example, for reproducing
plots on Fig. 5 and 6 panel E, first, run the script
*run_kinematic_replay* or *run_kinematic_replay_ground*, and then use:

.. code:: python

   from NeuroMechFly.utils import plotting
   import pickle

   path_data = /path/to/kinematic/replay/results/folder

   # Selecting right front leg for plotting (other options are LF, RM, LM, LH, or RH)
   leg = 'RF'

   # Read angles from file
   with open(path/to/angles, 'rb') as f:
       angles = pickle.load(f)

   # Defining time limits for the plot (seconds)
   start_time = 3.5 # 0.5 for grooming
   stop_time = 4.6 # 2.5 for grooming

   plotting.plot_data(path_data,
              leg,
              angles=angles,
              plot_angles=True,
              plot_torques=True,
              plot_grf=True,
              plot_collisions=True, # For grooming example
              collisions_across=True,
              begin=start_time,
              end=stop_time)

-  For reproducing gait/collision diagrams from Fig. 5 and 6, first, run
   the script *run_kinematic_replay* or *run_kinematic_replay_ground*,
   and then use:

.. code:: python

   from NeuroMechFly.utils import plotting

   path_data = /path/to/kinematic/replay/results/folder

   # Selecting walking behavior
   behavior = 'walking'

   # Defining time limits for the plot (seconds)
   start_time = 3.5 # 0.5 for grooming
   stop_time = 4.6 # 2.5 for grooming

   plotting.plot_collision_diagram(path_data,
                           behavior,
                           begin=start_time,
                           end=stop_time)


-  For reproducing plots from Fig. 7 panel E, first, run the script
   *run_neuromuscular_control*, and then use:

.. code:: python

   from NeuroMechFly.utils import plotting

   path_data = /path/to/neuromuscular/control/results/folder

   # Selecting right front leg for plotting (other options are LF, RM, LM, LH, or RH)
   leg = 'RF'

   # Defining time limits for the plot (seconds)
   start_time = 1.0
   stop_time = 1.5

   plotting.plot_data(path_data,
              leg,
              plot_angles=False,
              plot_torques=False,
              plot_grf=False,
              collisions_across=False,
              plot_muscles_act=True,
              plot_torques_muscles=True,
              plot_angles_sim=True,
              begin=start_time,
              end=stop_time)
