""" Drosophila Evolution. """
from datetime import datetime
import logging
import os
from pathlib import Path

import numpy as np

import pybullet as p
from NeuroMechFly.container import Container
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.core.observer import Observer
from jmetal.core.problem import FloatProblem
from jmetal.core.problem import DynamicProblem
from jmetal.core.solution import FloatSolution
from jmetal.lab.visualization import InteractivePlot, Plot
from jmetal.operator import (DifferentialEvolutionCrossover,
                             PolynomialMutation, SBXCrossover)
from jmetal.util.evaluator import MultiprocessEvaluator
from jmetal.util.observer import ProgressBarObserver, VisualizerObserver
from jmetal.util.ranking import FastNonDominatedRanking
from jmetal.util.solution import \
    print_function_values_to_file  # read_solutions,
from jmetal.util.solution import print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations
from drosophila_simulation_opt import DrosophilaSimulation

LOGGER = logging.getLogger('jmetal')

class WriteFullFrontToFileObserver(Observer):

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
                    self.directory))
            for file in os.listdir(self.directory):
                os.remove('{0}/{1}'.format(self.directory, file))
        else:
            LOGGER.warning(
                'Directory {} does not exist. Creating it.'.format(
                    self.directory))
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
                            self.directory, self.counter)
                    )
                    print_function_values_to_file(
                        solutions,
                        '{}/FUN.{}'.format(
                            self.directory, self.counter)
                    )
                    self.counter += 1
            else:
                print_variables_to_file(
                    solutions,
                    '{}/VAR.{}'.format(
                        self.directory, self.counter)
                )
                print_function_values_to_file(
                    solutions,
                    '{}/FUN.{}'.format(
                        self.directory, self.counter)
                )
                self.counter += 1

def read_optimization_results(fun, var):
    """ Read optimization results. """
    return (np.loadtxt(fun), np.loadtxt(var))

class DrosophilaEvolution(FloatProblem):
    """Documentation for DrosophilaEvolution"""
    def __init__(self):
        super(DrosophilaEvolution, self).__init__()
        self.number_of_variables = 79
        self.number_of_objectives = 2
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["Distance (negative)", "Activations"]

        #: Bounds
        noscillators = 36
        N = int(noscillators/4)

        #: Muscle parameters
        lower_bound_active_muscles = (
        np.ones((N, 6))*np.array([1e-3, 1e-3, 1e-4, 1e-5, 0.5, 0.0]
            )).flatten()
        #upper_bound_active_muscles = (
        #    np.ones((N, 6))*np.array([7e-1, 7e-1, 7e-2, 7e-3, 0.0, 0.0]
        #    )).flatten()
        
        #upper_bound_active_muscles = [8e-3, 8e-3, 8e-4, 8e-5, 1.5, 2,
        #                              8e-3, 8e-3, 8e-4, 8e-5, 1.5, 2,
        #                              8e-3, 8e-3, 8e-4, 8e-5, 1.5, 2,
        #                              5e-2, 5e-2, 5e-3, 5e-4, 1.5, 2,
        #                              5e-2, 5e-2, 5e-3, 5e-4, 1.5, 2,
        #                              5e-2, 5e-2, 5e-3, 5e-4, 1.5, 2,
        #                              7e-2, 7e-2, 7e-3, 7e-4, 1.5, 2,
        #                              7e-2, 7e-2, 7e-3, 7e-4, 1.5, 2,
        #                              7e-2, 7e-2, 7e-3, 7e-4, 1.5, 2]

        #lower_bound_active_muscles = np.array([3e-3, 8e-3, 8e-4, 8e-5, 1.5, 2]*3 +
        #                                      [5e-2, 5e-2, 5e-3, 5e-4, 1.5, 2]*3 +
        #                                      [7e-2, 7e-2, 7e-3, 7e-4, 1.5, 2]*3)
        
        upper_bound_active_muscles = np.array([2e-2, 2e-2, 3e-3, 1e-3, 1.5, 2]*3 +
                                              [3e-2, 3e-2, 3e-3, 1e-3, 1.5, 2]*3 +
                                              [3e-2, 3e-2, 1e-3, 8e-4, 1.5, 2]*3)
        
        #: Phases
        lower_bound_phases = np.ones(
            (25,))*-np.pi
        upper_bound_phases = np.ones(
            (25,))*np.pi

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

        #fun, var = read_optimization_results(
        #     "./FUN_mixed_retrain.ged3",
        #     "./VAR_mixed_retrain.ged3"
        #)

        #fun, var = read_optimization_results(
        #     "./optimization_results/run_Drosophila_var_48_obj_2_pop_12_gen_30_03_19_12/FUN.19",
        #     "./optimization_results/run_Drosophila_var_48_obj_2_pop_12_gen_30_03_19_12/VAR.19",
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
        #: SIMULATION RUN TIME
        run_time = 5.
        time_step = 0.001
        sim_options = {
            "headless": True,
            "model": "../../design/sdf/drosophila_100x_Limits_strict_offset.sdf",
            "model_offset": [0., 0., 1.12],
            "pose": "../../config/pose.yaml",
            "run_time": run_time,
            "base_link": 'Thorax',
            "controller": '../../config/locomotion_ball.graphml',
        }
        container = Container(run_time/time_step)
        fly = DrosophilaSimulation(container, sim_options)
        #: Set the variables
        fly.update_parameters(solution.variables)
        successful = fly.run(optimization=True)

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
        
        #: Objectives
        #: Minimize activations
        m_out = np.asarray(container.muscle.outputs.log)
        m_names = container.muscle.outputs.names
        act = np.asarray(
            [m_out[:, j] for j, name in enumerate(m_names) if 'active' in name]
        )
        act = np.sum(act**2)*fly.TIME_STEP/fly.RUN_TIME
        #: Distance
        distance = -np.array(fly.ball_rotations())[0]*fly.ball_radius #fly.distance_y
        
        #: Velocity
        #velocity = (
        #    np.sum(np.asarray(container.physics.joint_velocities.log)**2)
        #)*fly.TIME_STEP/fly.RUN_TIME

        #: Penalties
        penalty_time = 1e2 + 1e2*(fly.RUN_TIME - fly.TIME)/fly.RUN_TIME if (lava or flying or touch or velocity_cap) else 0.0
            
        expected_dist = 2*np.pi*fly.ball_radius
        penalty_dist = 0.0 if expected_dist < distance else (1e1 + 40*abs(distance-expected_dist))

        penalty_linearity = 1e1*fly.ball_radius*(abs(np.array(fly.ball_rotations()))[1]+abs(np.array(fly.ball_rotations()))[2])

        print(1e2*distance,1e4*act,penalty_linearity,penalty_dist,penalty_time)
        solution.objectives[0] = (-1e2*distance + penalty_linearity + penalty_time)
        solution.objectives[1] = (1e4*act + penalty_dist + penalty_time)
        # solution.objectives[2] = (velocity + penalty_dist + penalty_time)
        print(solution.objectives)
        return solution

    def get_name(self):
        return 'Drosophila'

    def __del__(self):
        print("Deleting fly simulation....")


def main():
    """ Main """

    n_pop = 20
    n_gen = 50

    max_evaluations = n_pop*n_gen

    problem = DrosophilaEvolution()

    algorithm = NSGAII(
        problem=problem,
        population_size=n_pop,
        offspring_population_size=n_pop,
        mutation=PolynomialMutation(
            probability=1.0 / problem.number_of_variables,
            distribution_index=0.20  # 20
        ),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        population_evaluator=MultiprocessEvaluator(8),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        # dominance_comparator=DominanceComparator()
    )

    # Results Dumping
    algorithm.observable.register(
        observer=WriteFullFrontToFileObserver(
            output_directory=(
                './optimization_results/run_{}_var_{}_obj_{}_pop_{}_gen_{}_{}'
            ).format(
                problem.get_name(),
                problem.number_of_variables,
                problem.number_of_objectives,
                n_pop,
                n_gen,
                datetime.now().strftime("%d_%H_%M"),
            )
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
    pareto_fronts = ranking.compute_ranking(front)

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
    print_function_values_to_file(front, 'FUN.' + 'ged3')
    print_variables_to_file(front, 'VAR.'+ 'ged3')

    print('Algorithm (continuous problem): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))


if __name__ == '__main__':
    main()
