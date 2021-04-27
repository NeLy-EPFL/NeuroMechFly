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
from farms_container import Container
from NeuroMechFly.experiments.network_optimization.neuromuscular_control import \
    DrosophilaSimulation

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
        self.number_of_variables = 75
        self.number_of_objectives = 2
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["Distance (negative)", "Stability"]

        #: Bounds
        noscillators = 36
        N = int(noscillators/4)

        # Muscle params with rest position being optimized
        lower_bound_active_muscles = (
                np.asarray(
                    [# Front
                    [1e-2, 1e-2, 1e-3, 1e-4, -0.22, 0.0, 0.0], # Coxa
                    [1e-2, 1e-2, 1e-3, 1e-4, -2.5, 0.0, 0.0], # Femur
                    [1e-2, 1e-2, 1e-3, 5e-4, 0.76, 0.0, 0.0], # Tibia
                    # Mid
                    [1e-2, 1e-2, 1e-3, 1e-4, -2.2, 0.0, 0.0], # Coxa_roll
                    [1e-2, 1e-2, 1e-3, 1e-4, -2.35, 0.0, 0.0], # Femur
                    [1e-2, 1e-2, 1e-3, 5e-4, 1.73, 0.0, 0.0], # Tibia
                    # Hind
                    [1e-2, 1e-2, 1e-3, 1e-4, -2.78, 0.0, 0.0], # Coxa_roll
                    [1e-2, 1e-2, 1e-3, 1e-4, -2.46, 0.0, 0.0], # Femur
                    [1e-2, 1e-2, 1e-3, 5e-4, 1.12, 0.0, 0.0], # Tibia
                    ]
                )
        ).flatten()

        upper_bound_active_muscles = (
                np.asarray(
                    [
                    # Front
                    [1e0, 1e0, 1e0, 1e-3, 0.69, 1.75, 1.75], # Coxa
                    [1e0, 1e0, 1e0, 0.92e-3, -1.3, 1.75, 1.75], # Femur
                    [1e-1, 1e-1, 1e-1, 1e-3, 2.19, 1.75, 1.75], # Tibia
                    # Mid
                    [1e0, 1e0, 1e0, 1e-3, -1.95, 1.75, 1.75], # Coxa_roll
                    [1e0, 1e0, 1e0, 0.92e-3, -1.84, 1.75, 1.75], # Femur
                    [1e-1, 1e-1, 1e-1, 1e-3, 2.63, 1.75, 1.75], # Tibia
                    # Hind
                    [1e0, 1e0, 1e0, 1e-3, -2.44, 1.75, 1.75], # Coxa_roll
                    [1e0, 1e0, 1e0, 1e-3, -1.31, 1.75, 1.75], # Femur
                    [1e-1, 1e-1, 1e-1, 1e-3, 2.79, 1.75, 1.75], # Tibia
                    ]
                )
        ).flatten()

        #: Phases
        lower_bound_phases = np.ones(
            (12,))*-np.pi
        upper_bound_phases = np.ones(
            (12,))*np.pi

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
        #: SIMULATION RUN time
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

        # Velocity
        # velocity = (
        #    np.sum(np.asarray(container.physics.joint_velocities.log)**2)
        # )*fly.time_step/fly.run_time

        use_penalties = True
        if use_penalties:
            if not successful:
                lava = fly.is_lava()
                flying = fly.is_flying()
                #bbox = fly.is_in_not_bounds()
                touch = fly.is_touch()
                velocity_cap = fly.is_velocity_limit()
            else:
                lava = False
                flying = False
                #bbox = False
                touch = False
                velocity_cap = False

            # Penalties
            # Penalty time
            penalty_time = (
                1e0 + 1e0*(fly.RUN_TIME - fly.TIME)/fly.RUN_TIME
                if (lava or flying or touch or velocity_cap)
                else 0.0
            )

            # Penalty distance
            expected_dist = 2*np.pi*fly.ball_radius
            penalty_dist = 0.0 if expected_dist < distance else (
                1e1 + 40*abs(distance-expected_dist))

            # Penalty linearity
            penalty_linearity = 2e3*fly.ball_radius * \
                (abs(np.array(fly.ball_rotations))[
                 1]+abs(np.array(fly.ball_rotations))[2])

            # Penalty if all legs are in contact with the ground
            penalty_all_legs = 5.0 if (
                np.any(fly.check_is_all_legs == False)) else (0.0)

            # Penalty long stance periods
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
                PENALTIES\n=========\n \
                    Penalty linearity: {} \n \
                    Penalty time: {} \n \
                    Penalty distance: {} \n \
                    Penalty time stance: {} \n \
                    Penalty all legs: {} \n \
                ".format(
                    -distance,
                    act,
                    penalty_linearity,
                    penalty_time,
                    penalty_dist,
                    penalty_time_stance,
                    penalty_all_legs
                )
                )

            solution.objectives[0] = (
                -distance
                + penalty_all_legs
               # + penalty_linearity
               # + penalty_time
            )
            solution.objectives[1] = (
                act
                # + penalty_all_legs**2
                + penalty_time
            )
            # solution.objectives[1] = (
            #     2e3*stability
            #     + penalty_dist
            #     + penalty_time_stance
            # )
            print(solution.objectives)
        else:
            # Torques
            # torque_sum = (np.sum(
            #     np.asarray(container.physics.joint_torques.log)**2
            # ))*fly.time_step/fly.run_time
            active_torque_sum = (np.sum(
                np.asarray(container.muscle.active_torques.log)**2
            ))*fly.TIME_STEP/fly.RUN_TIME

            # Objectives
            #solution.objectives[0] = -distance + abs(distance_lateral)
            #solution.objectives[1] = active_torque_sum
            # solution.objectives[1] = torque_sum

        return solution

    def get_name(self):
        return 'Drosophila'

    def __del__(self):
        print('Deleting fly simulation....')

def parse_args(problem):
    """Argument parser"""
    parser = argparse.ArgumentParser(
        description='FARMS simulation with Pybullet',
        formatter_class=(
            lambda prog:
            argparse.HelpFormatter(prog, max_help_position=50)
        ),
    )
    parser.add_argument(
        '--n_pop',
        type=int,
        default=10,
        help='Population size',
    )
    parser.add_argument(
        '--n_gen',
        type=int,
        default=4,
        help='Number of generations',
    )
    parser.add_argument(
        '--n_cpu',
        type=int,
        default=4,
        help='Number of CPUs',
    )
    parser.add_argument(
        '--output_directory',
        type=str,
        default=(
            # './optimization_results/run_{}_var_{}_obj_{}_pop_{}_gen_{}_{}'
            './optimization_results/run_{}_var_{}_obj_{}_{}'
        ).format(
            problem.get_name(),
            problem.number_of_variables,
            problem.number_of_objectives,
            # n_pop,
            # n_gen,
            datetime.now().strftime('%m%d_%H%M'),
        ),
        help='Output directory',
    )
    parser.add_argument(
        '--output_fun',
        type=str,
        default='FUN.txt',
        help='Results output of functions',
    )
    parser.add_argument(
        '--output_var',
        type=str,
        default='VAR.txt',
        help='Results output of variables',
    )
    parser.add_argument(
        '--mutation_probability',
        type=float,
        default=1.0 / problem.number_of_variables,
        help='Mutation probability',
    )
    parser.add_argument(
        '--mutation_distribution',
        type=float,
        default=0.2,  # 20
        help='Mutation probability',
    )
    parser.add_argument(
        '--crossover_probability',
        type=float,
        default=1.0,
        help='Crossover probability',
    )
    parser.add_argument(
        '--crossover_distribution',
        type=float,
        default=20,
        help='Crossover probability',
    )
    return parser.parse_args()


def main():
    """ Main """

    # Problem
    problem = DrosophilaEvolution()

    # Parse command line arguments
    clargs = parse_args(problem=problem)
    max_evaluations = clargs.n_pop*clargs.n_gen

    # Algorithm
    algorithm = NSGAII(
        problem=problem,
        population_size=clargs.n_pop,
        offspring_population_size=clargs.n_pop,
        mutation=PolynomialMutation(
            probability=clargs.mutation_probability,
            distribution_index=clargs.mutation_distribution,
        ),
        crossover=SBXCrossover(
            probability=clargs.crossover_probability,
            distribution_index=clargs.crossover_distribution,
        ),
        population_evaluator=MultiprocessEvaluator(clargs.n_cpu),
        termination_criterion=StoppingByEvaluations(
            max_evaluations=max_evaluations
        ),
        # dominance_comparator=DominanceComparator()
    )

    # Results Dumping
    algorithm.observable.register(
        observer=WriteFullFrontToFileObserver(
            output_directory=clargs.output_directory,
        )
    )

    # Visualisers
    algorithm.observable.register(
        observer=ProgressBarObserver(max=max_evaluations)
    )
    algorithm.observable.register(
        observer=VisualizerObserver(
            reference_front=problem.reference_front
        )
    )

    # Run optimisation
    algorithm.run()

    # Get results
    front = algorithm.get_result()
    ranking = FastNonDominatedRanking()
    # pareto_fronts = ranking.compute_ranking(front)

    # Plot front
    plot_front = Plot(
        title='Pareto front approximation',
        reference_front=problem.reference_front,
        axis_labels=problem.obj_labels)
    plot_front.plot(front, filename=algorithm.get_name())

    # Plot interactive front
    plot_front = InteractivePlot(
        title='Pareto front approximation',
        reference_front=problem.reference_front,
        axis_labels=problem.obj_labels)
    plot_front.plot(front, filename=algorithm.get_name())

    # Save results to file
    print_function_values_to_file(front, clargs.output_fun)
    print_variables_to_file(front, clargs.output_var)

    print('Algorithm (continuous problem): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time*1/60))


if __name__ == '__main__':
    main()
