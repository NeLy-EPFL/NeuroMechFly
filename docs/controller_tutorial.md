# Controller module

In this tutorial you can learn how to:
- [Modify the PyBullet joint controller](#modifying-PyBullet-joint-controller)
- [Modify our neural controller](#Modifying-our-neural-controller)
- [Incorporate customized controllers](#incorporating-customized-controllers)

## Modifying PyBullet joint controller

We use two control modes in our experiments for actuating NeuroMechFly joints: 

**Position control**

This mode is used in our kinematic replay experiments. It takes angular positions for controlling each joint. There are certain parameters that can be modified such as the controller gains (Kp and Kd). Please refer to the [environment tutorial](environment_tutorial.md) to learn how to change these parameters while inizializing the simulation.

**Torque control**

This mode is used in our optimization experiments. It takes torques computed from our muscles model to actuated each joint. Please refer to the [muscles tutorial](muscles_tutorial.md) to learn how these model can be modified.

## Modifying our neural controller

Our neural controller relies on a coupled oscillators network which parameters are optimized through a multi-objective genetic algorithm. You can either modify the network architecture to explore different behaviors, or reformulate the objective functions and penalties to evaluate their relevance and evolution over generations with our optimization framework.

**Modifying the oscillators network**

Our network is created using the [NetworkX](https://networkx.org/) Python package. Please refer to their documentation for a full description of how to create a network. We define our architecture in the script ```data/locomotion_network/locomotion.py```. When running this script a configuration file named ```locomotion_network.graphml``` is created in the ```config/network/``` folder. This file contains the network description defining its nodes, edges, phases, and weights. You can refer to the comments in the ```locomotion.py``` script to learn how to modify the network's architecture.

*NOTE:* If you modify the architecture, make sure to update its parameters accordingly in the script ```NeuroMechFly/experiments/network_optimization/neuromuscular_control.py```. Also, update the optimization framework if neccesary in the script ```NeuroMechFly/experiments/network_optimization/multiobj_optimization.py```.

**Formulating new objective functions and penalties**

We defined two objective functions: locomotor speed and static stability. Furthermore, we added four penalties to those functions: a moving threshold, angular velocity limit, range of motion, and duty-factor. Please refer to our related [publication](https://www.biorxiv.org/content/10.1101/2021.04.17.440214v2) for a full description. 

These objective functions and penalties are implemented in the script ```NeuroMechFly/experiments/network_optimization/neuromuscular_control.py``` you can refer to the docstrings in that script for learning how to modify them.

On the other hand, you can always define completely new objective functions and penalties. In that case, you would need to update add them to the optimization framework in the script ```NeuroMechFly/experiments/network_optimization/multiobj_optimization.py```.

## Incorporating customized controllers

You can also incorporate a completely different neural controller in NeuroMechFly, e.g., a Hodgkin-Huxley model. This implies to implement your own script for computing your prefered model, or you can also check [FARMS Network](https://gitlab.com/farmsim/farms_network) to learn how to design new neural network controllers. However, once you have your neural controller, you can use our other three modules (muscles, biomechanical, and environment) to run a complete simulation. The requirements for using NeuroMechFly are:

**Using muscles, biomechanical, and environment modules**

- The output of your neural controller should be a **motor neuron-like** activity function: our muscle model computes torques based on this kind of signal such as our CPG-like function from our coupled oscillators. This output is stored using the *Container class* and loaded to the muscles during its initialization in the script ```NeuroMechFly/experiments/network_optimization/neuromuscular_control.py```. Please, refer to [FARMS container](https://gitlab.com/farmsim/farms_container) to learn how to use the *Container class*.

**Using biomechanical, and environment modules**

- The output of your neural controller should be **torques or angular positions** for each joint: in this case you can choose between both control modes explained above (position and torque modes), refer to those sections to learn how to test your neural controller with our biomechanical model and its environment.
