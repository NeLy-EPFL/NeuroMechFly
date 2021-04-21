from NeuroMechFly.experiments.network_optimization import multiobj_optimization
from NeuroMechFly.experiments.network_optimization.multiobj_optimization import WriteFullFrontToFileObserver

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator import PolynomialMutation, SBXCrossover
from jmetal.util.evaluator import MultiprocessEvaluator
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.observer import ProgressBarObserver, VisualizerObserver
from jmetal.util.ranking import FastNonDominatedRanking
from jmetal.util.solution import (print_function_values_to_file,
                                  print_variables_to_file)
from jmetal.lab.visualization import Plot     
from datetime import datetime

if __name__ == "__main__":
    """ Main """
    n_pop = 10
    n_gen = 4
    max_evaluations = n_pop*n_gen
    # Problem
    problem = multiobj_optimization.DrosophilaEvolution()

    # Parse command line arguments
    max_evaluations = n_pop*n_gen

    # Algorithm
    algorithm = NSGAII(
        problem=problem,
        population_size=n_pop,
        offspring_population_size=n_pop,
        mutation=PolynomialMutation(
            probability=1.0 / problem.number_of_variables,
            distribution_index=0.20,
        ),
        crossover=SBXCrossover(
            probability=1.0,
            distribution_index=20,
        ),
        population_evaluator=MultiprocessEvaluator(4),
        termination_criterion=StoppingByEvaluations(
            max_evaluations=max_evaluations
        ),
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
                datetime.now().strftime('%m%d_%H%M'),
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
    # pareto_fronts = ranking.compute_ranking(front)

    # Plot front
    plot_front = Plot(
        title='Pareto front approximation',
        reference_front=problem.reference_front,
        axis_labels=problem.obj_labels)
    plot_front.plot(front, filename=algorithm.get_name())

    # Save results to file
    print_function_values_to_file(front, 'FUN.txt')
    print_variables_to_file(front, 'VAR.txt')

    print('Algorithm (continuous problem): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time*1/60))