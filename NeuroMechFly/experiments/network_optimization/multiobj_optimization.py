""" Drosophila Evolution. """

import logging
import os
from pathlib import Path
import pkgutil
import random
import yaml
from typing import Tuple

from typing import Tuple
import numpy as np
from jmetal.core.observer import Observer
from jmetal.core.problem import DynamicProblem, FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.util.solution import (
    print_function_values_to_file,
    print_variables_to_file,
)

import farms_pylog as pylog
from farms_container import Container
from NeuroMechFly.experiments.network_optimization.neuromuscular_control import \
    DrosophilaSimulation


LOGGER = logging.getLogger('jmetal')

neuromechfly_path = Path(pkgutil.get_loader(
    "NeuroMechFly").get_filename()).parents[1]

pylog.set_level('error')

# NOTE: Uncomment these lines to always get the same optimization results
# random.seed(0)
# np.random.seed(0)


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
            # for file in os.listdir(self.directory):
            #     os.remove('{0}/{1}'.format(self.directory, file))
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


def generate_config_file(log: dict) -> None:
    """ Generates a config file of the weights used in the optimization. """

    output_file_directory = os.path.join(
        neuromechfly_path,
        'scripts/neuromuscular_optimization',
        'optimization_results',
        'CONFIG.yaml'
    )

    pylog.info('Output config file : ' + output_file_directory)

    with open(output_file_directory, 'w') as of:
        yaml.dump(log, of, default_flow_style=False)


class DrosophilaEvolution(FloatProblem):
    """ Class for Evolutionary Optimization. """

    def __init__(self):
        super(DrosophilaEvolution, self).__init__()
        # Set number of variables, objectives, and contraints
        self.number_of_variables = 63
        self.number_of_objectives = 2
        self.number_of_constraints = 0
        # Minimize the objectives
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["Distance (negative)", "Stability"]

        # Bounds for frequency
        lower_bound_frequency = 6  # Hz
        upper_bound_frequency = 10  # Hz

        # Bounds for the muscle parameters 3 muscles per leg
        # Each muscle has 5 variables to be optimized corresponding to
        # Alpha, beta, gamma, delta, and resting pose of the Ekeberg model
        lower_bound_active_muscles = (
            np.asarray(
                [  # Front
                    [1e-10, 1e-10, 1.0, 5e-13, 0.0],  # Coxa
                    [1e-10, 1e-10, 1.0, 5e-13, -2.0],  # Femur
                    [1e-10, 1e-10, 1.0, 5e-13, 1.31],  # Tibia
                    # Mid
                    [1e-10, 1e-10, 1.0, 5e-13, 2.18],  # Coxa_roll
                    [1e-10, 1e-10, 1.0, 5e-13, -2.14],  # Femur
                    [1e-10, 1e-10, 1.0, 5e-13, 1.96],  # Tibia
                    # Hind
                    [1e-10, 1e-10, 1.0, 5e-13, 2.69],  # Coxa_roll
                    [1e-10, 1e-10, 1.0, 5e-13, -2.14],  # Femur
                    [1e-10, 1e-10, 1.0, 5e-13, 1.43],  # Tibia
                ]
            )
        ).flatten()

        upper_bound_active_muscles = (
            np.asarray(
                [
                    # Front
                    [5e-9, 1e-9, 10.0, 1e-11, 0.47],  # Coxa
                    [1e-9, 1e-9, 10.0, 1e-11, -1.68],  # Femur
                    [1e-9, 1e-9, 10.0, 1e-11, 2.05],  # Tibia
                    # Mid
                    [5e-9, 1e-9, 10.0, 1e-11, 2.01],  # Coxa_roll
                    [1e-9, 1e-9, 10.0, 1e-11, -2.0],  # Femur
                    [1e-9, 1e-9, 10.0, 1e-11, 2.22],  # Tibia
                    # Hind
                    [5e-9, 1e-9, 10.0, 1e-11, 2.53],  # Coxa_roll
                    [1e-9, 1e-9, 10.0, 1e-11, -1.55],  # Femur
                    [1e-9, 1e-9, 10.0, 1e-11, 2.26],  # Tibia
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
                lower_bound_frequency,
                lower_bound_active_muscles,
                lower_bound_phases
            )
        )
        self.upper_bound = np.hstack(
            (
                upper_bound_frequency,
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

        # Initial solutions: upper, mid, and lower bounds
        self.initial_solutions = [
            self.upper_bound.tolist(),
            ((self.upper_bound + self.lower_bound) * 0.5).tolist(),
        ]
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
        run_time = 2.0
        #: Set a time step for the physics engine
        time_step = 1e-4
        #: Setting up the paths for the SDF and POSE files
        model_path = neuromechfly_path.joinpath(
            "data/design/sdf/neuromechfly_locomotion_optimization.sdf"
        )
        pose_path = neuromechfly_path.joinpath(
            "data/config/pose/pose_tripod.yaml"
        )
        controller_path = neuromechfly_path.joinpath(
            "data/config/network/locomotion_network.graphml"
        )
        # Set collision segments
        ground_contacts = tuple(
            f"{side}{leg}{segment}"
            for side in ('L', 'R')
            for leg in ('F', 'M', 'H')
            for segment in tuple(f"Tarsus{i}" for i in range(1, 6))
        )

        # Set the ball specs based on your experimental setup
        ball_specs = {
            'ball_mass': 54.6e-6,
            'ground_friction_coef': 1.3
        }
        # Simulation options
        sim_options = {
            "headless": True,
            "model": str(model_path),
            "time_step": time_step,
            "solver_iterations": 100,
            "model_offset": [0., 0., 11.2e-3],
            "pose": pose_path,
            "run_time": run_time,
            "controller": controller_path,
            "base_link": 'Thorax',
            "ground_contacts": ground_contacts,
            "camera_distance": 4.5,
            "track": False,
            "globalCFM": 5.,
            **ball_specs
        }
        #: Create the container instance that the simulation results will be dumped
        container = Container(run_time / time_step)
        #: Create the simulation instance with the specified options and container
        fly = DrosophilaSimulation(container, sim_options)
        #: Update the parameters (i.e. muscle, phases)
        fly.update_parameters(solution.variables)
        #: Check if any of the termination criteria is met
        _successful = fly.run(optimization=True)

        #: OBJECTIVES
        objectives = {}

        #: Forward distance (backward rotation of the ball)
        objectives['distance'] = np.array(fly.ball_rotations)[
            0] * fly.ball_radius * fly.units.meters
        objectives['stability'] = fly.opti_stability
        # objectives['mechanical_work'] = np.sum(fly.mechanical_work)

        #: PENALTIES
        penalties = {}

        duty_factor = fly.duty_factor
        # Keep the duty factor between 40% and 90%
        # Taken from Mendes et al. 2012
        penalties['duty_factor'] = np.count_nonzero(
            duty_factor < 0.4) + np.count_nonzero(duty_factor > 0.90)

        #: Penalty long stance periods
        # constraints = {}
        # constraints['max_legs'] = 5
        # constraints['min_legs'] = 2
        # mean_stance_legs = fly.stance_count * fly.time_step / fly.time
        # penalties['stance'] = (
        #     0.0
        #     if constraints['min_legs'] <= mean_stance_legs <= constraints['max_legs']
        #     else 1.0
        # )

        penalties['lava'] = fly.opti_lava
        penalties['velocity'] = fly.opti_velocity
        penalties['joint_limits'] = fly.opti_joint_limit

        weights = {
            'distance': -1e1,
            'stability': -1e-2,
            'mechanical_work': 1e-2,
            'stance': 1e0,
            'lava': 1e-1,
            'velocity': 1e-1,
            'joint_limits': 5e-2,
            'duty_factor': 1e2
        }

        objectives_weighted = {
            obj_name: obj_value * weights[obj_name]
            for obj_name, obj_value in objectives.items()
        }

        penalties_weighted = {
            pen_name: pen_value * weights[pen_name]
            for pen_name, pen_value in penalties.items()
        }

        #: Print penalties and objectives
        print('\nObjectives\n==========')
        for name, item in objectives_weighted.items():
            print(
                '{}: {}'.format(name, item)
            )
        print('\nPenalties\n=========')
        for name, item in penalties_weighted.items():
            print(
                '{}: {}'.format(name, item)
            )

        solution.objectives[0] = objectives_weighted['distance'] + \
            sum(penalties_weighted.values())
        solution.objectives[1] = objectives_weighted['stability'] + \
            sum(penalties_weighted.values())

        print(
            f'\nObjective func eval:\nFirst: {solution.objectives[0]}\nSecond: {solution.objectives[1]}\n'
        )

        config_file = {
            weight_name: weight for weight_name, weight in weights.items()
            if weight_name in {**objectives, **penalties}
        }

        config_file = {**config_file, **constraints} if 'stance' in penalties else config_file
        generate_config_file(config_file)

        return solution

    def get_name(self):
        """ Name of the simulation. """
        return 'Drosophila'

    def __del__(self):
        """ Delete the simulation at the end of each run. """
        print('Deleting fly simulation....')
