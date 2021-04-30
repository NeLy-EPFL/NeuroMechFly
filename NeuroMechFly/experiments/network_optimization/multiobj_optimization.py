""" Drosophila Evolution. """
import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import farms_pylog as pylog
import numpy as np
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.core.observer import Observer
from jmetal.core.problem import DynamicProblem, FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.lab.visualization import InteractivePlot, Plot
from jmetal.operator import PolynomialMutation, SBXCrossover
from jmetal.util.evaluator import MultiprocessEvaluator
from jmetal.util.observer import ProgressBarObserver, VisualizerObserver
from jmetal.util.ranking import FastNonDominatedRanking
from jmetal.util.solution import (print_function_values_to_file,
                                  print_variables_to_file)
from jmetal.util.termination_criterion import StoppingByEvaluations

from NeuroMechFly.experiments.network_optimization.neuromuscular_control import DrosophilaSimulation
from NeuroMechFly.container import Container

LOGGER = logging.getLogger('jmetal')


class WriteFullFrontToFileObserver(Observer):
    """ Write full front to file """

    def __init__(self, output_directory: str) -> None:
        """ Write function values of the front into files.

        :param output_directory: Output directory.
        Each front will be saved on a file `FUN.x`.
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
    """Documentation for DrosophilaEvolution"""
    def __init__(self):
        super(DrosophilaEvolution, self).__init__()
        self.number_of_variables = 62
        self.number_of_objectives = 2
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["Distance (negative)", "Stability"]

        #: Bounds for the muscle parameters
        noscillators = 36
        N = int(noscillators/4)

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

        #: Bounds for the phases
        lower_bound_phases = np.ones(
            (17,))*-np.pi
        upper_bound_phases = np.ones(
            (17,))*np.pi

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

        #fun, var = read_optimization_results(
        #     "./FUN_mixed_retrain.ged3",
        #     "./VAR_mixed_retrain.ged3"
        #)

        #fun, var = read_optimization_results(
        #     "./optimization_results/run_Drosophila_var_71_obj_2_pop_20_gen_100_1106_0257/FUN.12",
        #     "./optimization_results/run_Drosophila_var_71_obj_2_pop_20_gen_100_1106_0257/VAR.12",
        #)

        #self.initial_solutions =  list(var) # [var[np.argmin(fun[:, 0])]]
        self.initial_solutions = []
        self._initial_solutions = self.initial_solutions.copy()

    def create_solution(self):
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
        #: SIMULATION RUN time, decreasing will lower the computation time
        run_time = 5.
        time_step = 0.001
        sim_options = {
            "headless": True,
            "model": "../data/design/sdf/neuromechfly_limitsFromData_minMax.sdf",
            "model_offset": [0., 0., 11.2e-3],
            "pose": "../data/config/pose/test_pose_tripod.yaml",
            "run_time": run_time,
            "base_link": 'Thorax',
            "controller": '../data/config/network/locomotion_ball.graphml',
        }

        container = Container(run_time/time_step)
        fly = DrosophilaSimulation(container, sim_options)

        # Set the variables
        fly.update_parameters(solution.variables)
        successful = fly.run(optimization=True)

        # Objectives
        # Minimize activations
        m_out = np.asarray(container.muscle.outputs.log)
        m_names = container.muscle.outputs.names

        # Activations
        act = np.asarray([
            m_out[:, j]
            for j, name in enumerate(m_names)
            if 'flexor_act' in name or 'extensor_act' in name
        ])
        # normalize it by the maximum activation possible [0- ~2]
        act = np.sum(act**2)/1e5

        # Forward distance
        distance = -np.array(
            fly.ball_rotations
        )[0]*fly.ball_radius  

        # Stability
        stability = fly.stability_coef*fly.TIME_STEP/fly.TIME_STEP

        if not successful:
            lava = fly.is_lava()
            flying = fly.is_flying()
            touch = fly.is_touch()
            velocity_cap = fly.is_velocity_limit()
        else:
            lava = False
            flying = False
            touch = False
            velocity_cap = False

        #: Penalties
        # Penalty time
        penalty_time = (
            1e0 + 1e0*(fly.RUN_TIME - fly.TIME)/fly.RUN_TIME
            if (lava or flying or touch or velocity_cap)
            else 0.0
        )

        #: Penalty distance
        expected_dist = 2*np.pi*fly.ball_radius
        penalty_dist = 0.0 if expected_dist < distance else (
            1e1 + 40*abs(distance-expected_dist))

        #: Penalty linearity
        penalty_linearity = 2e3*fly.ball_radius * \
            (abs(np.array(fly.ball_rotations))[
                1]+abs(np.array(fly.ball_rotations))[2])

        #: Penalty long stance periods
        expected_stance_legs = 4
        min_legs = 3
        mean_stance_legs = fly.stance_count*fly.RUN_TIME/fly.TIME
        penalty_time_stance = (
            0.0
            if min_legs <= mean_stance_legs <= expected_stance_legs
            else 1e2 * abs(mean_stance_legs - min_legs)
        )

        ### PRINT PENALTIES AND OBJECTIVES ###
        pylog.debug(
            "OBJECTIVES\n===========\n\
                Distance: {} \n \
                Activations: {} \n \
                Stability: {} \n \
            PENALTIES\n=========\n \
                Penalty linearity: {} \n \
                Penalty time: {} \n \
                Penalty distance: {} \n \
                Penalty time stance: {} \n \
            ".format(
                -distance,
                act,
                stability,
                penalty_linearity,
                penalty_time,
                penalty_dist,
                penalty_time_stance,
            )
            )

        solution.objectives[0] = (
            -distance
            + penalty_time
        )
        solution.objectives[1] = (
            stability
            + penalty_time_stance
        )
        # solution.objectives[1] = (
        #     2e3*stability
        #     + penalty_dist
        #     + penalty_time_stance
        # )
        print(solution.objectives)

        return solution

    def get_name(self):
        return 'Drosophila'

    def __del__(self):
        print('Deleting fly simulation....')
