## Modifying our muscle model

We use our Ekeberg muscle model when controlling NeuroMechfLy with our coupled oscillators network. Please refer to our related [publication](https://www.biorxiv.org/content/10.1101/2021.04.17.440214v2) to have a complete description of this model. Each variable from this muscle model is determined through our multi-objective optimization. However, you could modify (or replace) these parameters from the scripts. 

The optimized variables are incorporated into the muscle model in the *update_parameters* function in the script ``` NeuroMechFly/experiments/network_optimization/neuromuscular_control.py```. The order for assigning these muscle's parameters to each joint is: α, β, γ, δ, and Δφ. We use the same muscles properties for flexors and extensors, that's also something that you could change in the same function.

The muscle model is implemented in the script ```NeuroMechFly/control/spring_damper_muscles.py```.

## Incorporating customized muscle models

There is always the possibility to incorporate completely different muscle models in NeuroMechFly, e.g., Hill-type muscles. That implies to implement your own script computing your prefered model. However, once you have your muscle model, you can use our other three modules (neural controller, biomechanical, and environment) to run a complete simulation. The requirements for using NeuroMechFly would depend on the modules you want to use:

**Using neural controller, biomechanical, and environment modules**

1. The input of your muscle model should be a **CPG-like** function: you can retrieve these values from our *Container class* as we do for initializing our muscles. For an example, see our implementation in the *init* function from the script ```NeuroMechFly/experiments/network_optimization/neuromuscular_control.py```.
2. The output of your muscle model should be **joint torques**: you can actuated each joint using the *torque control* from PyBullet. For an example, see our implementation in the *muscle_controller* function from the script ```NeuroMechFly/experiments/network_optimization/neuromuscular_control.py```.

*NOTE:* If you do not want to use our neural controller module, then you should only follow condition number 2.
