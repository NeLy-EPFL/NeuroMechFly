# NeuroMechFly

<p align="center">
  <img align="center" width="600" src="https://github.com/NeLy-EPFL/NeuroMechFly/blob/f6464a158958077c37845695ba37204fbc8f062a/docs/NeuroMechFly.gif">
</p>

**NeuroMechFly** is a data-driven computational simulation of adult *Drosophila* designed to synthesize rapidly growing experimental datasets and to test theories of neuromechanical behavioral control. Specifically, you can use NeuroMechFly to:
* (A) estimate expected contact reaction forces, torques, and tactile signals during replayed *Drosophila* walking and grooming to expand your experimental data
* (B) discover neural network topologies that can drive different gaits
* (C) use machine learning methods to create a link between neuroscience and artificial intelligence. 

For more details, please refer to our [paper](https://www.biorxiv.org/content/10.1101/2021.04.17.440214v1).

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
* [Installation](https://github.com/NeLy-EPFL/NeuroMechFly/blob/81f22ddf165940434fe55b67008647171940b1b9/docs/installation.md)
* [Angle Processing](https://github.com/NeLy-EPFL/NeuroMechFly/blob/81f22ddf165940434fe55b67008647171940b1b9/docs/angleprocessing.md)

## Reproducing the experiments 
Before running the scripts please make sure that you activate the virtual environment by running the following on the command line:

```$ conda activate neuromechfly```

**1. Kinematic Matching**
<p align="center">
  <img src="https://github.com/NeLy-EPFL/NeuroMechFly/blob/f6464a158958077c37845695ba37204fbc8f062a/docs/km_grooming.gif" width="330" />
  <img src="https://github.com/NeLy-EPFL/NeuroMechFly/blob/f6464a158958077c37845695ba37204fbc8f062a/docs/km_walking.gif" width="330" />
</p>

- Navigate to */scripts/* folder.
- Run ```$ python run_kinematic_matching.py --behavior walking``` for simulating the locomotion behavior. 
- Run ```$ python run_kinematic_matching.py --behavior grooming``` for simulating the foreleg/antennal grooming behavior. 
- Run ```$ python run_kinematic_matching_ground.py``` for simulating the locomotion behavior on the ground. 

<p align="center">
  <img src="https://github.com/NeLy-EPFL/NeuroMechFly/blob/81f22ddf165940434fe55b67008647171940b1b9/docs/perturbation.gif" width="450" />
</p>

**NOTE:** To obtain new pose estimates from the [DeepFly3D Database](https://dataverse.harvard.edu/dataverse/DeepFly3D), please refer to [DeepFly3D repository](https://github.com/NeLy-EPFL/DeepFly3D). After running the pose estimator on the recordings, you can follow the instructions for computing joint angles to control NeuroMechFly [here.](https://github.com/NeLy-EPFL/NeuroMechFly/blob/km-refactor/docs/angleprocessing.md)

**NOTE 2:** After obtaining new pose estimates, you can follow this guide(link here) to produce your experiments.

**2. Optimization** 

- Run ```$ python run_neuromuscular_control.py``` to run the results from the paper. This script will read and run the files *FUN.txt* and *VAR.txt*.
- Run ```$ python run_multiobj_optimization.py``` to run the obtimization from stratch. This script will create a new files named *FUN.txt* and *VAR.txt*.

**NOTE 2:** To formulate new objective functions and design a different controller, please read this guide(link).

## Miscellaneous 
- To see the CPG network, navigate to */scripts/Network* and run ```$ python locomotion.py```

