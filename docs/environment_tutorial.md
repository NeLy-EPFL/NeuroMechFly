## Managing the simulation options

There are 40 simulation options that can be changed from the scripts. Please refer to ```NeuroMechFly/simulation/bullet_simulation.py``` for checking all the options. Here we enlisted the options we use in our scripts for managing the environment:

- ```headless```: The GUI is not shown if *True*. This speed-up the simulation. 
- ```model```: Global path to the sdf file with the desired model description.
- ```pose```: Global path to the yaml file with the desired initial pose.
- ```results_path```: Global path where the results will be stored.
- ```model_offset```: Model's base offset *[x, y, z]* (in meters) when the simulation starts.
- ```run_time```: Total time of the simulation in seconds.
- ```time_step```: Time step of the simulation in seconds.
- ```base_link```: Link defined as the starting point of the kinematic chain.
- ```ground_contacts```: List of link's names from which ground reactions forces will be obtained.
- ```self_collisions```: List of link's names from which self-collision forces will be obtained.
- ```draw_collisions```: If *True*, links in contact with the ground or in self-collision (if defined in self-collision variable) are painted green.
- ```record```: If *True* a video from the simulation is recorded. This variable is accesible from our script's flags. ```headless``` variable should be *False*.
- ```save_frames```: If *True* an image for each time step of the simulation is stored in the folder specified in the ```results_path``` variable.
- ```camera_distance```: Distance (in millimeters) from the rendering camera to the model's base link.
- ```track```: If *True* the camera will follow the model as it moves in the environment.
- ```moviename```: Name of the recorded video with the global path. If ```record``` is *True*.
- ```moviespeed```: Speed for the recorded video, 1 corresponds to real time. If ```record``` is *True*.
- ```slow_down```: If *True* the simulation is paused ```sleep_time``` seconds after each time step.
- ```sleep_time```: Sleep time when ```slow_down``` is *True*.
- ```rot_cam```: If *True* the camera rotates around the model as in our kinematic replay videos.
- ```behavior```: Specifies which behavior we are simulating (*walking, grooming, or None*) for selecting the treadmill position, if ```ball_info``` is *False*, and the ```rot_cam``` sequence.
- ```ground```: Specifies what will be consider as ground during the simulation (*ball or floor*).
- ```ground_friction_coef```: Specifies the lateral friction coefficient for the ground specified in ```ground```.
- ```ball_info```: If *True* a file named *treadmill_info__** will be read to obtain the treadmill position and size.
- ```ball_mass```: Specifies the mass of the treadmill. If *None* the mass is calculated based on its size and the polyurethane foam density.

You can refer to any of the scripts in the ```scripts/kinematic_replay``` folder to have an example of how to use any of these options. 

## Initializing the simulation

```python
animal = kinematic_replay.DrosophilaSimulation(
        container,
        sim_options,
        kp=0.4, kv=0.9,
        angles_path=angles_path,
        velocity_path=velocity_path,
        starting_time=starting_time,
        fixed_positions=fixed_positions,
    )
```

## Adding objects to the environment

