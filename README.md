# NeuroMechFly

<p align="center">
  <img align="center" width="600" src="https://github.com/NeLy-EPFL/NeuroMechFly/blob/675d9ae07db1c3899926b1bbaf447230f7737ad0/docs/NeuroMechFly.gif">
</p>

**NeuroMechFly** is a data-driven computational simulation of adult *Drosophila* designed to synthesize rapidly growing experimental datasets and to test theories of neuromechanical behavioral control. Specifically, you can use NeuroMechFly to:
* (A) estimate expected contact reaction forces, torques, and tactile signals during replayed *Drosophila* walking and grooming to expand your experimental data
* (B) discover neural network topologies that can drive different gaits
* (C) use machine learning methods to create a link between neuroscience and artificial intelligence. 

For more details, please refer to our paper (link here).

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
- [Running the experiments](#running-the-experiments)
- [Miscellaneous](#miscellaneous)


## Starting
* [Installation](https://github.com/NeLy-EPFL/NeuroMechFly/blob/km-refactor/docs/installation.md)
* [Angle Processing](https://github.com/NeLy-EPFL/NeuroMechFly/blob/km-refactor/docs/angleprocessing.md)

## Running the experiments 
Before running the scripts please make sure that you activate the virtual environment by running the following on the command line:
```$ conda activate neuromechfly```

**1. Kinematic Matching**
- Make sure in */data* folder, there are */walking/df3d* and *grooming/df3d* folders that contain a .pkl file starting with "joint_angles_..".
- Navigate to */scripts/KM* folder.
- Run ```$ python kinematicMatching_noSelfCollisions.py``` for simulating the locomotion behavior. 
- Run ```$ python kinematicMatching.py``` for simulating the forleg/antennal grooming behavior. 

**NOTE:** To obtain new pose estimates from the [DeepFly3D Database](https://dataverse.harvard.edu/dataverse/DeepFly3D), please refer to [DeepFly3D repository](https://github.com/NeLy-EPFL/DeepFly3D). After running the pose estimator on the recordings, you can follow the instructions for computing joint angles to control NeuroMechFly [here.](https://github.com/NeLy-EPFL/NeuroMechFly/blob/km-refactor/docs/angleprocessing.md)

**2. Optimization** 
- To simulate the results of the last evolution, run ```$ python drosophila_simulation_opt.py```. This script will run the best result from *FUN.ged3* and *VAR.ged3*. 
- To formulate the objective functions and run the evolution, run ```$ python drosophila_evolution.py```. This is where the jMetal framework is set. 

## Miscellaneous 
- To see the CPG network, navigate to */scripts/Network* and run ```$ python locomotion.py```

