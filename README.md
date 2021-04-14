# NeuroMechFly

<p align="center">
  <img align="center" width="600" src="https://github.com/NeLy-EPFL/NeuroMechFly/blob/675d9ae07db1c3899926b1bbaf447230f7737ad0/docs/NeuroMechFly.gif">
</p>

NeuroMechFly is a data-driven computational simulation of adult *Drosophila* designed to synthesize rapidly growing experimental datasets and to test theories of neuromechanical behavioral control. Specifically, you can use NeuroMechFly to:
* (A) estimate expected contact reaction forces, torques, and tactile signals during replayed *Drosophila* walking and grooming to expand your experimental data
* (B) discover neural network topologies that can drive different gaits
* (C) use machine learning methods to create a link between neuroscience and artificial intelligence. 

For more details, please refer to our paper (link here).

If you find NeuroMechFly useful in your research, please consider citing us!

```Latex
@article{lobato2021neuromechfly,
	title = {NeuroMechFly, a neuromechanical model of adult Drosophila melanogaster},
	journal = {In preparation},
	author = {Lobato Rios, Victor and Ozdil, Pembe Gizem and Ramalingasetty, Shravan and Arreguit, Jonathan and Rosset, Stéphanie and Knott, Graham and Ijspeert, Auke and Ramdya, Pavan},
	year = {2021}
}
```


## Content

- [Installation](#installation)
- [Running the experiments](#running-the-experiments)

## Installation 

First, you can download the repository on your local machine by running the following line in the terminal:
```bash
$ git clone https://github.com/NeLy-EPFL/NeuroMechFly.git
```
After the download is complete, navigate to the NeuromMechFly folder:
```bash
$ cd NeuroMechFly
```
In this folder, run the following commands to create a virtual environment:
```bash
$ conda create -n neuromechfly python=3.6 numpy Cython
```
Finally, install all the dependencies by running:
```bash
$ pip install -e .
```
Once you complete all the steps, NeuroMechFly is ready to use!

## Running the experiments 
Before running the scripts please make sure that you activated the conda environment by running ```$ conda activate neuromechfly``` on the command line. 

**1. Kinematic Matching**
- Make sure in */data* folder, there are */walking/df3d* and *grooming/df3d* folders that contain a .pkl file starting with "joint_angles_..".
- Navigate to */scripts/KM* folder.
- Run ```$ python kinematicMatching_noSelfCollisions.py``` for simulating the locomotion behavior. 
- Change the behavior in ```main()```to "grooming" to simulate the grooming behavior. Note: Collisions are disabled in this script, but use this for now because *kinematicMatching.py* needs to be modified for the new scale. 

**2. Optimization** 
- To simulate the results of the last evolution, run ```$ python drosophila_simulation_opt.py```. This script will run the best result from *FUN.ged3* and *VAR.ged3*. 
- To formulate the objective functions and run the evolution, run ```$ python drosophila_evolution.py```. This is where the jMetal framework is set. 

## Miscellaneous 
- To see the CPG network, navigate to */scripts/Network* and run ```$ python locomotion.py```

