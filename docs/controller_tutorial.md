# Controller module

In this tutorial you will learn how to:
- [Modify the PyBullet joint controller](#modifying-PyBullet-joint-controller)
- [Modify our neural controller](#Modifying-our-neural-controller)
- [Incorporate customized controllers](#incorporating-customized-controllers)

## Modifying the PyBullet joint controller

We use two control modes in our experiments to actuate NeuroMechFly's joints: 

**Position control**

This mode is used in our Kinematic Replay experiments. It takes as inputs angular positions for each joint. There are certain parameters that can be modified such as the controller gains (Kp and Kd). Please refer to the [environment tutorial](environment_tutorial.md) to learn how to change these parameters when initializing the simulation.

**Torque control**

This mode is used in our Optimization experiments. It takes as inputs torques computed from our muscles model to actuate each joint. Please refer to the [muscles tutorial](muscles_tutorial.md) to learn how these models can be modified.

## Modifying our neural controller

Our neural controller is a network of coupled oscillators whose parameters are optimized using a multi-objective genetic algorithm. You can either modify the network architecture to explore different behaviors, or reformulate the objective functions and penalties in our optimization framework to evaluate their importance over generations.

**Modifying the network of coupled oscillators**

Our network was created using the [NetworkX](https://networkx.org/) Python package. Please refer to their documentation for a full description of how to create a network. We define our architecture in the script ```data/locomotion_network/locomotion.py```. When running this script, a configuration file named ```locomotion_network.graphml``` is created in the ```config/network/``` folder. This file contains the network description defining its nodes, edges, phases, and weights. You can refer to the comments in the ```locomotion.py``` script to learn how to modify the network's architecture.

*NOTE:* If you modify the architecture, make sure to update its parameters accordingly in the script ```NeuroMechFly/experiments/network_optimization/neuromuscular_control.py```. Also, be sure to update the optimization framework if neccesary in the script ```NeuroMechFly/experiments/network_optimization/multiobj_optimization.py```.

**Formulating new objective functions and penalties**

We defined two objective functions: locomotor speed and static stability. Furthermore, we added four penalties to those functions: a moving boundary, an angular velocity limit, a range of motion limit, and a duty-factor range. Please refer to our related [publication](https://www.biorxiv.org/content/10.1101/2021.04.17.440214v2) for a full description. 

These objective functions and penalties are implemented in the script ```NeuroMechFly/experiments/network_optimization/neuromuscular_control.py```. You can refer to the docstrings in that script to learn how to modify them.

On the other hand, you can always define completely new objective functions and penalties. In that case, you would need to add them to the optimization framework in the script ```NeuroMechFly/experiments/network_optimization/multiobj_optimization.py```.

## Incorporating customized controllers

You can also incorporate a completely different neural controller into NeuroMechFly (e.g., a Hodgkin-Huxley model). You would need to implement your own script to compute your preferred model, or you can also check [FARMS Network](https://gitlab.com/farmsim/farms_network) to learn how to design new neural network controllers. However, once you have your neural controller, you can use our other three modules (muscles, biomechanical, and environment) to run a complete simulation. The requirements for using NeuroMechFly are as follows:

**Using muscles, biomechanical, and environment modules**

- The output of your neural controller should be a **motor neuron-like** activity function: our muscle model computes torques based on this kind of signal. For example, the output of a CPG-like function from a network of coupled oscillators. This output is stored using the *Container class* and loaded to the muscles during initialization in the script ```NeuroMechFly/experiments/network_optimization/neuromuscular_control.py```. Please, refer to [FARMS container](https://gitlab.com/farmsim/farms_container) to learn how to use the *Container class*.

**Using biomechanical, and environment modules**

- The output of your neural controller should be **torques or angular positions** for each joint. In this case you can choose between either of the control modes explained above (i.e., position or torque mode). Refer to those sections to learn how to test your neural controller with our biomechanical model within its environment.
