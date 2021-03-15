# NeuroMechFly
Functions for running NeuroMechFly simulation

## Installation 
To install NeuroMechFly, run the following commands in the command line:
```bash
$ conda create -n neuromechfly python=3.6 numpy Cython
$ git clone https://github.com/NeLy-EPFL/NeuroMechFly.git
$ pip install -e .
```

## Running the experiments 
Before running the scripts please make sure that you activated the conda environment by running ```bash $ conda activate neuromechfly``` on the command line. 

**1. Kinematic Matching**
- Make sure in */data* folder, there are */walking/df3d* and *grooming/df3d* folders that contain a .pkl file starting with "joint_angles_..".
- Navigate to */scripts/KM* folder.
- Run ```bash $ python kinematicMatching_noSelfCollisions.py``` for simulating the locomotion behavior. 
- Change the behavior in ```main()```to "grooming" to simulate the grooming behavior. Note: Collisions are disabled in this script, but use this for now because *kinematicMatching.py* needs to be modified for the new scale. 

**2. Optimization** 
- To simulate the results of the last evolution, run ```bash $ python drosophila_simulation_opt.py```. This script will run the best result from *FUN.ged3* and *VAR.ged3*. 
- To formulate the objective functions and run the evolution, run ```bash $ python drosophila_evolution.py```. This is where the jMetal framework is set. 

## Miscellaneous 
- To see the CPG network, navigate to */scripts/Network* and run ```bash $ python locomotion.py```

