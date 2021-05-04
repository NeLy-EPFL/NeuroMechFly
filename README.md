# NeuroMechFly
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) 
[![Version](https://badge.fury.io/gh/tterb%2FHyde.svg)](https://badge.fury.io/gh/tterb%2FHyde)

<p align="center">
  <img align="center" width="600" src="docs/NeuroMechFly.gif">
</p>

**NeuroMechFly** is a data-driven computational simulation of adult *Drosophila* designed to synthesize rapidly growing experimental datasets and to test theories of neuromechanical behavioral control. Specifically, you can use NeuroMechFly to:
* (A) estimate expected contact reaction forces, torques, and tactile signals during replayed *Drosophila* walking and grooming to expand your experimental data
* (B) discover neural network topologies that can drive different gaits
* (C) use machine learning methods to create a link between neuroscience and artificial intelligence. 

For the technical background and details, please refer to our [paper](https://www.biorxiv.org/content/10.1101/2021.04.17.440214v1).

If you find NeuroMechFly useful in your research, please consider citing us!

```Latex
@article {R{\'\i}os2021.04.17.440214,
	author = {R{\'\i}os, Victor Lobato and {\"O}zdil, Pembe Gizem and Ramalingasetty, Shravan Tata and Arreguit, Jonathan and Clerc Rosset, St{\'e}phanie and Knott, Graham and Ijspeert, Auke Jan and Ramdya, Pavan},
	title = {NeuroMechFly, a neuromechanical model of adult Drosophila melanogaster},
	elocation-id = {2021.04.17.440214},
	year = {2021},
	doi = {10.1101/2021.04.17.440214},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2021/04/18/2021.04.17.440214},
	eprint = {https://www.biorxiv.org/content/early/2021/04/18/2021.04.17.440214.full.pdf},
	journal = {bioRxiv}
}
```

## Content

- [Starting](#starting)
- [Reproducing the experiments](#reproducing-the-experiments)
- [Miscellaneous](#miscellaneous)


## Starting
* [Installation](docs/installation.md)
* [Angle Processing](docs/angleprocessing.md)

## Reproducing the experiments 
Before running the scripts, please make sure that you activate the virtual environment (see the [installation guide](docs/installation.md))

NeuroMechFly is run in [PyBullet](https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet). In the Graphical User Interface, you can use the following keyboard and mouse combinations to control the camera viewpoint:
- ALT/CONTROL & Left Mouse Button: Rotate
- ALT/CONTROL & Scroll Mouse Button: Pan
- Scroll Mouse Button: Zoom

**1. Kinematic Matching**
<p align="center">
  <img src="docs/km_walking.gif" width="330" />
  <img src="docs/km_grooming.gif" width="330" />
</p>

Run the following commands on the terminal to reproduce the kinematic replay experiments:
- ```$ run_kinematic_replay --behavior walking```  for the locomotion behavior on a spherical treadmill. To simulate the foreleg/antennal grooming behavior, change ```walking``` at the end of the command to ```grooming```.
**Note:** Locomotion behavior starts approximately 2.5 seconds in the simulation. Until then, the fly stands still on the spherical treadmill. 

- ```$ run_kinematic_replay_ground --perturbation``` for simulating the locomotion behavior on the ground with perturbations enabled. Remove ```--perturbation``` to disable perturbations. To change the behavior type, append ```--behavior grooming``` to the command

<p align="center">
  <img src="docs/perturbation.gif" width="450" />
</p>

**NOTE:** To obtain new pose estimates from the [DeepFly3D Database](https://dataverse.harvard.edu/dataverse/DeepFly3D), please refer to [DeepFly3D repository](https://github.com/NeLy-EPFL/DeepFly3D). After running the pose estimator on the recordings, you can follow the instructions for computing joint angles to control NeuroMechFly [here.](https://github.com/NeLy-EPFL/NeuroMechFly/blob/km-refactor/docs/angleprocessing.md)

---

**2. Optimization** 

Run the following commands on the terminal to reproduce the optimization experiments:
- ```$ run_neuromuscular_control``` to run the latest generation of the last optimization run. By default, this script will read and run the files *FUN.txt* and *VAR.txt* under the *scripts* folder. To run different files, simply run ```$ run_neuromuscular_control -v <path-of-the-var-file> -f <path-of-the-fun-file>```. **These paths should be relative to the *scripts* folder.**

- ```$ run_multiobj_optimization``` to run the optimization from scratch. This script will create new files named *FUN.txt* and *VAR.txt* as well as a new folder containing the results from each generation under a folder named *optimization_results*. After the optimization is over, run ```$ run_neuromuscular_control``` to see the results from the last generation. To see different generations, follow the instructions above to select a different file. 

**IMPORTANT NOTE:** We are currently improving the optimization approach, therefore, the code and results in this repository do not reflect the results in our [paper](https://www.biorxiv.org/content/10.1101/2021.04.17.440214v1). 

**NOTE:** To formulate new objective functions and penalties, please refer to the *NeuroMechFly/experiments/network_optimization*. 

---

**3. Sensitivity Analysis** 

- First, please download the simulation data obtained from sensitivity analysis from [here](https://drive.google.com/drive/folders/1H0G3mdeKLyGkS1DYxbOeOCXgywJmwfs9?usp=sharing) and place these files under *data/sensitivity_analysis*
- To reproduce the sensitivity analysis figures, ```$ run_sensitivity_analysis```. Please make sure that the downloaded files are located correctly. 

## Miscellaneous

**1. Central Pattern Generator Controller**
- To see the CPG network, navigate to *data/locomotion_network/* and run ```$ python locomotion.py```
- Please refer to [FARMS Network](https://gitlab.com/farmsim/farms_network) to learn more about designing new neural network controllers.

---

**2. Blender Model**
- To visualize the biomechanical model, first install [Blender](https://www.blender.org/download/). 
- After the installation, navigate to *data/design/blender* and open ```neuromechfly_full_model.blend``` with Blender. 

---

**3. Reproducing the Figures**
-  All the plotting functions used in the paper are provided in [*NeuroMechFly/utils/plotting.py*](NeuroMechFly/utils/plotting.py). Please refer to the docstrings provided in the code to learn how to plot your simulation data.

---

## License
[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)
