Environment module
==================

In this tutorial you can learn how to: - `Manage the simulation
options <#managing-the-simulation-options>`__ - `Initialize the
simulation <#initializing-the-simulation>`__ - `Add objects to the
environment <#adding-objects-to-the-environment>`__

Manage the simulation options
-----------------------------

There are 40 simulation options that can be changed from the scripts.
Please refer to ``NeuroMechFly/simulation/bullet_simulation.py`` to
check each of the options. Here we list the ones used in our scripts to
manage the environment:

-  ``headless``: The GUI is not shown if *True*. This speeds up the
   simulation.
-  ``model``: Global path to the sdf file with the desired model
   description.
-  ``pose``: Global path to the yaml file with the desired initial pose.
-  ``results_path``: Global path where the results will be stored.
-  ``model_offset``: Model's base offset *[x, y, z]* (in meters) when
   the simulation starts.
-  ``run_time``: Total time of the simulation in seconds.
-  ``time_step``: Time step of the simulation in seconds.
-  ``base_link``: Link defined as the starting point of the kinematic
   chain.
-  ``ground_contacts``: List of link names from which ground reactions
   forces will be obtained.
-  ``self_collisions``: List of link names from which self-collision
   forces will be obtained.
-  ``draw_collisions``: If *True*, links in contact with the ground or
   experiencing self-collisions (if defined in the self-collision
   variable) are painted green.
-  ``record``: If *True* a video from the simulation is recorded. This
   variable is accesible from our script's flags. ``headless`` variable
   should be *False*.
-  ``save_frames``: If *True* an image for each time step of the
   simulation will be stored in the folder specified in the
   ``results_path`` variable.
-  ``camera_distance``: Distance (in millimeters) from the rendering
   camera to the model's base link.
-  ``track``: If *True* the camera will follow the model as it moves in
   the environment.
-  ``moviename``: Name of the recorded video with the global path. If
   ``record`` is *True*.
-  ``moviespeed``: Speed for the recorded video, 1 corresponds to real
   time. If ``record`` is *True*.
-  ``slow_down``: If *True* the simulation is paused ``sleep_time``
   seconds after each time step.
-  ``sleep_time``: Sleep time when ``slow_down`` is *True*.
-  ``rot_cam``: If *True* the camera rotates around the model as in our
   Kinematic Replay videos.
-  ``behavior``: Specifies which behavior we are simulating (*walking,
   grooming, or None*) for selecting the treadmill position, if
   ``ball_info`` is *False*, and the ``rot_cam`` sequence.
-  ``ground``: Specifies what will be considered the ground during the
   simulation (*ball or floor*).
-  ``ground_friction_coef``: Specifies the lateral friction coefficient
   for the ground specified in ``ground``.
-  ``ball_info``: If *True* a file named \*treadmill\_info\_\_\*\* will
   be read to obtain the treadmill's position and size.
-  ``ball_mass``: Specifies the mass of the treadmill. If *None* the
   mass is calculated based on its size and the density of polyurethane
   foam.
-  ``solver_iterations``: Specifies the number of iterations used by the
   phisics engine solver during each time step. Default value is 1000,
   if you decrease it the simulation will run faster but the solver
   could not converge to a feasible solution.

You can refer to any of the scripts in the ``scripts/kinematic_replay``
folder to have an example of how to use them.

Initializing the simulation
---------------------------

When the simulation is initialized, you can set other parameters besides
the simulation options explained above. For example, as shown in the
snippet below, we can (i) set the controller gains (*kp* and *kv*) if we
are using PyBullet's PD controller, (ii) specify the position of joints
that should remain fixed during the simulation, and (iii) define the
paths where data can be found. We use this kind of initialization for
the Kinematic Replay scripts. You can find examples of usage in
``scripts/kinematic_replay/``. If you want to add other variables during
initialization you need to modify the class ``DrosophilaSimulation``
found in any script in the folder ``NeuroMechFly/experiments``.

.. code:: python

    animal = kinematic_replay.DrosophilaSimulation(
            container,
            sim_options,
            kp=0.4, kv=0.9,
            angles_path=angles_path,
            velocity_path=velocity_path,
            starting_time=starting_time,
            fixed_positions=fixed_positions
        )

Adding objects to the environment
---------------------------------

Please refer to the `PyBullet
documentation <https://pybullet.org/wordpress/>`__ for a complete guide
on how to include objects in your simulation. We include objects in the
simulation in three ways. The NeuroMechFly model is imported from a
*sdf* file which contains the model's description (see the `biomechanics
tutorial <biomechanical_tutorial.md>`__). External perturbations shown
in Video 10 from our related
`publication <https://www.biorxiv.org/content/10.1101/2021.04.17.440214v2>`__
were added from a `*urdf* <https://wiki.ros.org/urdf/Tutorials>`__ file.
This is another format supported by PyBullet that is used to describe
objects (you can find this example in the file
``NeuroMechFly/experiments/kinematic_replay/kinematic_replay_no_support.py``).
Finally, we included the spherical treadmill in the simulation using the
*createMultiBody* built-in function from PyBullet. You can refer to
``NeuroMechFly/simulation/bullet_simulation.py`` to see an example of
how we use this function.
