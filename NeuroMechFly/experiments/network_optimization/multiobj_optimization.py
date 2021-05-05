""" Drosophila Evolution. """

import logging
import os
from pathlib import Path
import pkgutil

import farms_pylog as pylog
import numpy as np
from jmetal.core.observer import Observer
from jmetal.core.problem import DynamicProblem, FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.util.solution import (
    print_function_values_to_file,
    print_variables_to_file,
)
from farms_container import Container
from NeuroMechFly.experiments.network_optimization.neuromuscular_control import \
    DrosophilaSimulation


LOGGER = logging.getLogger('jmetal')

neuromechfly_path = Path(pkgutil.get_loader("NeuroMechFly").get_filename()).parents[1]



class WriteFullFrontToFileObserver(Observer):
    """ Write full front to file. """

    def __init__(self, output_directory: str) -> None:
        """ Write function values of the front into files.

        output_directory: <str>
            Output directory in which the optimization results will be saved.
            Objective functions will be saved on a file `FUN.x`.
            Variable values will be saved on a file `VAR.x`.
        """
        self.counter = 0
        self.directory = output_directory

        if Path(self.directory).is_dir():
            LOGGER.warning(
                'Directory {} exists. Removing contents.'.format(
                    self.directory,
                )
            )
            for file in os.listdir(self.directory):
                os.remove('{0}/{1}'.format(self.directory, file))
        else:
            LOGGER.warning(
                'Directory {} does not exist. Creating it.'.format(
                    self.directory,
                )
            )
            Path(self.directory).mkdir(parents=True)

    def update(self, *args, **kwargs):
        problem = kwargs['PROBLEM']
        solutions = kwargs['SOLUTIONS']

        if solutions:
            if isinstance(problem, DynamicProblem):
                termination_criterion_is_met = kwargs.get(
                    'TERMINATION_CRITERIA_IS_MET', None)
                if termination_criterion_is_met:
                    print_variables_to_file(
                        solutions,
                        '{}/VAR.{}'.format(
                            self.directory,
                            self.counter,
                        )
                    )
                    print_function_values_to_file(
                        solutions,
                        '{}/FUN.{}'.format(
                            self.directory,
                            self.counter,
                        )
                    )
                    self.counter += 1
            else:
                print_variables_to_file(
                    solutions,
                    '{}/VAR.{}'.format(
                        self.directory,
                        self.counter,
                    )
                )
                print_function_values_to_file(
                    solutions,
                    '{}/FUN.{}'.format(
                        self.directory,
                        self.counter,
                    )
                )
                self.counter += 1


def read_optimization_results(fun, var):
    """ Read optimization results. """
    return (np.loadtxt(fun), np.loadtxt(var))


class DrosophilaEvolution(FloatProblem):
    """ Class for Evolutionary Optimization. """

    def __init__(self):
        super(DrosophilaEvolution, self).__init__()
        #: Set number of variables, objectives, and contraints
        self.number_of_variables = 62
        self.number_of_objectives = 2
        self.number_of_constraints = 0
        #: Minimize the objectives
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["Distance (negative)", "Stability"]

        #: Bounds for the muscle parameters 3 muscles per leg
        #: Each muscle has 5 variables to be optimized corresponding to
        #: Alpha, beta, gamma, delta, and resting pose of the Ekeberg model
        lower_bound_active_muscles = (
                np.asarray(
                    [# Front
                    [1e-2, 1e-2, 1e-3, 1e-4, -0.22], # Coxa
                    [1e-2, 1e-2, 1e-3, 1e-4, -2.5], # Femur
                    [1e-2, 1e-2, 1e-3, 5e-4, 0.76], # Tibia
                    # Mid
                    [1e-2, 1e-2, 1e-3, 1e-4, -2.2], # Coxa_roll
                    [1e-2, 1e-2, 1e-3, 1e-4, -2.35], # Femur
                    [1e-2, 1e-2, 1e-3, 5e-4, 1.73], # Tibia
                    # Hind
                    [1e-2, 1e-2, 1e-3, 1e-4, -2.78], # Coxa_roll
                    [1e-2, 1e-2, 1e-3, 1e-4, -2.46], # Femur
                    [1e-2, 1e-2, 1e-3, 5e-4, 1.12], # Tibia
                    ]
                )
        ).flatten()

        upper_bound_active_muscles = (
                np.asarray(
                    [
                    # Front
                    [1e0, 1e0, 1e0, 1e-3, 0.49], # Coxa
                    [1e0, 1e0, 1e0, 1e-3, -1.3], # Femur
                    [1e-1, 1e-1, 1e-1, 1e-3, 2.19], # Tibia
                    # Mid
                    [1e0, 1e0, 1e0, 1e-3, -1.75], # Coxa_roll
                    [1e0, 1e0, 1e0, 1e-3, -1.84], # Femur
                    [1e-1, 1e-1, 1e-1, 1e-3, 2.63], # Tibia
                    # Hind
                    [1e0, 1e0, 1e0, 1e-3, -2.44], # Coxa_roll
                    [1e0, 1e0, 1e0, 1e-3, -1.31], # Femur
                    [1e-1, 1e-1, 1e-1, 1e-3, 2.79], # Tibia
                    ]
                )
        ).flatten()

        #: Bounds for the intraleg (12) and interleg (5) phases
        lower_bound_phases = np.ones(
            (17,)) * -np.pi
        upper_bound_phases = np.ones(
            (17,)) * np.pi

        self.lower_bound = np.hstack(
            (
                lower_bound_active_muscles,
                lower_bound_phases
            )
        )
        self.upper_bound = np.hstack(
            (
                upper_bound_active_muscles,
                upper_bound_phases
            )
        )

        #: Uncomment in case of warm start
        # fun, var = read_optimization_results(
        #     "./FUN_warm_start.3",
        #     "./VAR_warm_start.3"
        # )

        # self.initial_solutions =  list(var) # [var[np.argmin(fun[:, 0])]]
        self.initial_solutions = []
        self._initial_solutions = self.initial_solutions.copy()

    def create_solution(self):
        """ Creates a new solution based on the bounds and variables. """
        new_solution = FloatSolution(
            self.lower_bound,
            self.upper_bound,
            self.number_of_objectives,
            self.number_of_constraints
        )
        new_solution.variables = np.random.uniform(
            self.lower_bound,
            self.upper_bound,
            self.number_of_variables
        ).tolist() if not self._initial_solutions else self._initial_solutions.pop()
        return new_solution

    def evaluate(self, solution):
        """ Evaluates the optimization solution in PyBullet without GUI.

        Parameters
        ----------
        solution : <FloatSolution>
            Solution obtained from the optimizer.

        Returns
        -------
        solution : <FloatSolution>
            Evaluated solution.
        """
        #: Set how long the simulation will run to evaluate the solution
        run_time = 3.0
        #: Set a time step for the physics engine
        time_step = 0.001
        #: Setting up the paths for the SDF and POSE files
        model_path = os.path.join(
            neuromechfly_path,
            'data/design/sdf/neuromechfly_limitsFromData.sdf',
        )
        pose_path = os.path.join(
            neuromechfly_path,
            'data/config/pose/pose_tripod.yaml',
        )
        controller_path = os.path.join(
            neuromechfly_path,
            'data/config/network/locomotion_network.graphml',
        )
        #: Simulation options
        sim_options = {
            "headless": True,
            "model": model_path,
            "model_offset": [0., 0., 11.2e-3],
            "pose": pose_path,
            "run_time": run_time,
            "base_link": 'Thorax',
            "controller": controller_path
        }
        #: Create the container instance that the simulation results will be dumped
        container = Container(run_time / time_step)
        #: Create the simulation instance with the specified options and container
        fly = DrosophilaSimulation(container, sim_options)
        #: Update the parameters (i.e. muscle, phases)
        fly.update_parameters(solution.variables)
        #: Check if any of the termination criteria is met
        _successful = fly.run(optimization=True)

        #: Objectives
        #: Minimize activations
        m_out = np.asarray(container.muscle.outputs.log)
        m_names = container.muscle.outputs.names
        act = np.asarray([
            m_out[:, j]
            for j, name in enumerate(m_names)
            if 'flexor_act' in name or 'extensor_act' in name
        ])
        #: Normalize it by the maximum activation possible [0- ~2]
        act = np.sum(act**2) / 1e5

        #: Forward distance (backward rotation of the ball)
        distance = -np.array(
            fly.ball_rotations
        )[0] * fly.ball_radius

        #: Stability coefficient
        stability = fly.opti_stability

        #: Penalties
        #: Penalty long stance periods

        expected_stance_legs = 3.8
        min_legs = 3
        mean_stance_legs = fly.stance_count * fly.time_step / fly.time
        penalty_time_stance = (
            0.0
            if min_legs <= mean_stance_legs < expected_stance_legs
            else abs(mean_stance_legs - min_legs)
        )
        distance_weight = -1e1
        stability_weight = -1e-1
        movement_weight = 1e-2
        touch_weight = 1e-2
        velocity_weight = 1e-1
        stance_weight = 1e2
        penalties = (
            movement_weight * fly.opti_lava + \
            velocity_weight * fly.opti_velocity + \
            stance_weight * penalty_time_stance
        )

        #: Print penalties and objectives
        print(
            f"OBJECTIVES\n===========\n\
                Distance: {distance_weight * distance} \n \
                Stability: {stability_weight * stability} \n \
                Work: {fly.mechanical_work} \n \
                PENALTIES\n=========\n \
                Penalty lava: {movement_weight * fly.opti_lava} \n \
                Penalty velocity: {velocity_weight*fly.opti_velocity} \n \
                Penalty stance: {stance_weight * penalty_time_stance} \n \
            "
        )

        solution.objectives[0] = distance_weight * distance + penalties
        solution.objectives[1] = stability_weight * stability + penalties

        print(
            "OBJECTIVE FUNCTION EVALUATION:\n===========\n\
                First: {} \n \
                Second: {} \n \
            ".format(
                solution.objectives[0],
                solution.objectives[1]
            )
        )

        return solution

    def get_name(self):
        """ Name of the simulation. """
        return 'Drosophila'

    def __del__(self):
        """ Delete the simulation at the end of each run. """
        print('Deleting fly simulation....')
