# Muscles module

In this tutorial you can learn how to:
- [Modify our muscle model](#modifying-our-muscle-model)
- [Incorporate a customized muscle models](#incorporating-customized-muscle-models)

## Modifying our muscle model

We use a spring-and-damper (Ekeberg) muscle model to control NeuroMechFly using output signals from a network of coupled oscillators. Please refer to our related [publication](https://www.biorxiv.org/content/10.1101/2021.04.17.440214v2) for a complete description of this model. Each variable from this muscle model is determined using multi-objective evolutionary optimization. However, you can modify (or replace) these parameters from the scripts. 

The optimized variables are incorporated into the muscle model in the *update_parameters* function within the script ``` NeuroMechFly/experiments/network_optimization/neuromuscular_control.py```. The order for assigning these muscle's parameters to each joint is: α, β, γ, δ, and Δφ. We use the same muscle properties for flexors and extensors. This can also be changed using the same function.

The muscle model is implemented in the script ```NeuroMechFly/control/spring_damper_muscles.py```.

## Incorporating a customized muscle model

It is always possible to incorporate completely different muscle models in NeuroMechFly (e.g., Hill-type muscles). That requires implementing your own script to compute your preferred model. However, once you have your muscle model in hand, you can use our other three modules (neural controller, biomechanical, and environment) to run a complete simulation. The requirements for using NeuroMechFly would depend on the modules you want to use:

**Using neural controller, biomechanical, and environment modules**

1. The input to your muscle model should be the output of a **CPG-like** function. You can retrieve these values from our *Container class* as we do for initializing our muscles. For an example, see our implementation in the *init* function in the script ```NeuroMechFly/experiments/network_optimization/neuromuscular_control.py```.
2. The output of your muscle model should be **joint torques**. You can actuate each joint using *torque control* from PyBullet. For an example, see our implementation in the *muscle_controller* function in the script ```NeuroMechFly/experiments/network_optimization/neuromuscular_control.py```.

*NOTE:* If you do not want to use our neural controller module, then you should only follow condition number two.
