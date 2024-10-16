> [!IMPORTANT]  
> ⚠️⚠️⚠️⚠️⚠️
> 
> **This GitHub repository contains documentation for legacy code related to [Lobato-Rios et al, Nature Methods, 2022](https://www.nature.com/articles/s41592-022-01466-7). NeuroMechFly has since been updated, and this repository is no longer actively maintained. For most up-to-date information, please visit [neuromechfly.org](https://neuromechfly.org/).**

# NeuroMechFly
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Version](https://badge.fury.io/gh/tterb%2FHyde.svg)](https://badge.fury.io/gh/tterb%2FHyde)

<p align="center">
  <img align="center" width="600" src="docs/images/NeuroMechFly.gif">
</p>

**NeuroMechFly** is a data-driven computational simulation of adult *Drosophila melanogaster* designed to synthesize rapidly growing experimental datasets and to test theories of neuromechanical behavioral control. For the technical background and details, please refer to our [paper](https://www.nature.com/articles/s41592-022-01466-7).

If you use NeuroMechFly in your research, you can cite us:

```Latex
@article{LobatoRios2022,
  doi = {10.1038/s41592-022-01466-7},
  url = {https://doi.org/10.1038/s41592-022-01466-7},
  year = {2022},
  month = May,
  publisher = {Springer Science and Business Media {LLC}},
  volume = {19},
  number = {5},
  pages = {620--627},
  author = {Victor Lobato-Rios and Shravan Tata Ramalingasetty and Pembe Gizem \"{O}zdil and Jonathan Arreguit and Auke Jan Ijspeert and Pavan Ramdya},
  title = {{NeuroMechFly},  a neuromechanical model of adult Drosophila melanogaster},
  journal = {Nature Methods}
}
```

A Gym environment of NeuroMechFly is under development [here](https://github.com/NeLy-EPFL/nmf-gym).

## Content

- [Starting](#starting)
- [Reproducing the experiments](#reproducing-the-experiments)
- [Customizing NeuroMechFly](#customizing-neuromechfly)
- [Miscellaneous](#miscellaneous)


## Starting
* [Installation](docs/installation.md)
* [Angle Processing](docs/angleprocessing.md)


## Reproducing the experiments
**Note:** before running the following scripts, please be sure to activate the virtual environment (see the [installation guide](docs/installation.md))

NeuroMechFly is run in [PyBullet](https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet). In the Graphical User Interface, you can use the following keyboard and mouse combinations to control the camera's viewpoint:
- ALT/CONTROL & Left Mouse Button: Rotate
- ALT/CONTROL & Scroll Mouse Button: Pan
- Scroll Mouse Button: Zoom

**1. Kinematic replay**
<p align="center">
  <img src="docs/images/km_walking.gif" width="330" />
  <img src="docs/images/km_grooming.gif" width="330" />
</p>

Run the following commands on the terminal to reproduce the kinematic replay experiments:
- ```$ run_kinematic_replay -b walking```  for walking behavior on the spherical treadmill. Replace ```walking``` for ```grooming``` to simulate the foreleg/antennal grooming example. 

- ```$ run_kinematic_replay_ground``` for replaying tethered walking kinematics on the floor. Add ```--perturbation``` to enable perturbations. For changing the behavior to grooming, append ```-b grooming``` to the command.

Furthermore, for both commands above, you can add the flag ```-fly #``` to run the simulation with other walking behaviors, # can be 1, 2, or 3 (default is 1). The flag ```--show_collisions``` will colored in green the segments in collision. Finally, the flag ```--record``` will save a video from the simulation in the folder *scripts/kinematic_replay/simulation_results*. The video will be recorded at 0.2x real-time (refer to the [environment tutorial](docs/environment_tutorial.md) to learn how to change this value). 

<p align="center">
  <img src="docs/images/perturbation.gif" width="450" />
</p>

- ```$ run_morphology_experiment``` for replaying grooming kinematics changing the legs and antennae morphology. Add ```--model model_name``` to select the morphology. ```model_name``` can be ```nmf```, ```stick_legs```, or ```stick_legs_antennae```. This command also support ```--record``` and ```--show_collisions``` flags.

**NOTE:** At the end of each simulation run, a folder called *kinematic_replay_<behavior>_<time-stamp>* containing the physical quantities (joint angles, torques etc.) will be created under the *scripts/kinematic_replay/simulation_results* folder.

**NOTE:** Flags ```--show_collisions``` and ```--record``` will slow down your simulation.

**NOTE:** To obtain new pose estimates from the [DeepFly3D Database](https://dataverse.harvard.edu/dataverse/DeepFly3D), please refer to [DeepFly3D repository](https://github.com/NeLy-EPFL/DeepFly3D). After running the pose estimator on the recordings, you can follow the instructions for computing joint angles to control NeuroMechFly [here.](https://github.com/NeLy-EPFL/NeuroMechFly/blob/master/docs/angleprocessing.md)

---

**2. Gait optimization**

<p align="center">
  <img align="center" width="420" src="docs/images/optimization.gif">
</p>

Run the following commands on the terminal to reproduce the locomotor gait optimization experiments:
- ```$ run_neuromuscular_control --gui``` to run the latest generation of the last optimization run. By default, this script will read and run the files *FUN.txt* and *VAR.txt* under the *scripts/neuromuscular_optimization/* folder. To run different files, simply run ```$ run_neuromuscular_control --gui -p <'path-of-the-optimization-results'> -g <'generation-number'> -s <'solution-type'>``` (solution type being 'fastest', 'tradeoff', 'most_stable', or a specific index). **The results path should be relative to the *scripts* folder.** 
- To see the results that are already provided, go to the folder *scripts/neuromuscular_optimization/* and run: 
	```$ run_neuromuscular_control --gui  -p optimization_results/run_Drosophila_example/ -g 59```. 
- Append ```--plot``` to the command to visualize the Pareto front and the gait diagram of the solution. To record the simulation, append ```--record``` to the command you run. To log the penalties separately from the objective functions, append ```--log_penalties``` to the command you run, penalties will be logged in a new file named *PENALTIES.<gen>* in the provided path.

**NOTE:** At the end of each simulation run, a folder named according to the chosen optimization run will be created under the *scripts/neuromuscular_optimization* folder which contains the network parameters and physical quantities.

- ```$ run_multiobj_optimization``` to run locomotor gait optimization from scratch. This script will create new files named *FUN.txt* and *VAR.txt* as well as a new folder containing the results from each generation in a folder named *optimization_results*. After optimization has completed, run ```$ run_neuromuscular_control --gui``` to visualize the results from the last generation. To see different generations, follow the instructions above and select a different file.

**NOTE:** Optimization results will be stored under *scripts/neuromuscular_optimization/optimization_results* inside a folder named according to the chosen optimization run.

**NOTE:** To formulate new objective functions and penalties, please refer to the [neural controller tutorial](docs/controller_tutorial.md).

---

**3. Sensitivity Analysis**

- First, download the data from sensitivity analyses [here](https://drive.google.com/file/d/10XfMkMY0nhDABekzQ7wVid9hVI5C4Xiz/view?usp=sharing). Place these files into the folder, *data/sensitivity_analysis*
- To reproduce the sensitivity analysis figures, ```$ run_sensitivity_analysis```. Make sure that the downloaded files are in the correct location.
	
## Customizing NeuroMechFly
	
Each module in NeuroMechFly can be modified to create a customized simulation. Here are some tutorials explaining how to do this:
	
* [Biomechanical model](docs/biomechanical_tutorial.md)
	- Modify body segments.
	- Modify joints.
	- Change the pose.
	
* [Neural controller](docs/controller_tutorial.md)
	- Modify the PyBullet joint controller.
	- Modify our neural controller.
	- Incorporate customized controllers.
	
* [Muscle model](docs/muscles_tutorial.md)
	- Modify our muscle model.
	- Incorporate customized muscle models.

* [Environment](docs/environment_tutorial.md)	
	- Manage the simulation options.
	- Initialize the simulation.
	- Add objects to the environment.
	
---

## Miscellaneous

**1. Central Pattern Generator Controller**
- To see the CPG network, navigate to *data/locomotion_network/* and run ```$ python locomotion.py```
- Please refer to [FARMS Network](https://gitlab.com/farmsim/farms_network) to learn more about how to design new neural network controllers.

---

**2. Blender Model**
- To visualize the biomechanical model, first install [Blender](https://www.blender.org/download/).
- After installation, navigate to *data/design/blender* and open ```neuromechfly_full_model.blend``` with Blender.

---

**3. Reproducing the Figures**
-  All of the plotting functions used in the paper can be found in [*NeuroMechFly/utils/plotting.py*](NeuroMechFly/utils/plotting.py). Please refer to the docstrings provided in the code for details on how to plot your simulation data.
-  For example, to reproduce the plots on Figs. 4 and 5 panel E, first, run the script *run_kinematic_replay* or *run_kinematic_replay_ground*, and then use:
```python
from NeuroMechFly.utils import plotting
from pathlib import Path
import pickle
import glob
import os

path_data = '~/NeuroMechFly/scripts/kinematic_replay/simulation_results/<name-of-the-results-folder>'

# Selecting a behavior (walking or grooming)
behavior = 'walking'

# Selecting a fly
fly_number = 1

# Selecting the right front leg for plotting (other options are LF, RM, LM, LH, or RH)
leg = 'LF' # 'RF' for grooming

# Reading angles from a file
angles_path = os.path.join(str(Path.home()),f'NeuroMechFly/data/joint_tracking/{behavior}/fly{fly_number}/df3d/')
file_path = glob.glob(f'{angles_path}/joint_angles*.pkl')[0]
with open(file_path, 'rb') as f:
    angles = pickle.load(f)

# Defining time limits for a plot (in seconds)
start_time = 3.0 # 0.5 for grooming
stop_time = 5.0 # 2.5 for grooming

plotting.plot_data(path_data,
		   leg,
		   sim_data=behavior,
		   angles=angles,
		   plot_angles_intraleg=True,
		   plot_torques=True,
		   plot_grf=True,
		   plot_collisions=True,
		   collisions_across=True,
		   begin=start_time,
		   end=stop_time)
```

- To reproduce gait/collision diagrams from Figs. 4 and 5, first, run the script *run_kinematic_replay* or *run_kinematic_replay_ground*, and then use:
```python
from NeuroMechFly.utils import plotting

path_data = '~/NeuroMechFly/scripts/kinematic_replay/simulation_results/<name-of-the-results-folder>'

# Selecting walking behavior
behavior = 'walking'

# Defining time limits for the plot (seconds)
start_time = 3.0 # 0.5 for grooming
stop_time = 5.0 # 2.5 for grooming

plotting.plot_collision_diagram(path_data,
		                behavior,
		                begin=start_time,
		                end=stop_time)

```

- For reproducing plots from Fig. 6 panel E, and F, first, run the script *run_neuromuscular_control*, and then use:
```python
from NeuroMechFly.utils import plotting

# e.g. type: fastest, tradeoff, most_stable, or the individual, number: generation number
path_data = '~/NeuroMechFly/scripts/neuromuscular_optimization/simulation_last_run/gen_<number>/sol_<type>'

# Selecting the joint of interest (Coxa-Trochanter/Femur)
link = 'Femur'

# Defining time limits for the plot (in seconds)
start_time = 1.0
stop_time = 1.5

plotting.plot_network_activity(
    results_path=path_data,
    link=link,
    beg=start_time,
    end=stop_time
)

```
---
**4. The CT-scan Data**
	
File containing the raw X-ray microtomography data could be downloaded [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/PEOVAV).
	
---

## License
[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)
	
---
## README for the GitHub Pages Branch
This branch is simply a cache for the website served from https://nely-epfl.github.io/NeuroMechFly/,
and is  not intended to be viewed on github.com.
