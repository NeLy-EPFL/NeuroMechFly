#!/usr/bin/env python
""" Script to run neuromuscular control. """

import argparse
import os
import pkgutil
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool

from farms_container import Container
from NeuroMechFly.experiments.network_optimization import neuromuscular_control
from NeuroMechFly.utils.plotting import plot_collision_diagram, plot_data
from NeuroMechFly.utils.profiler import profile

neuromechfly_path = Path(pkgutil.get_loader("NeuroMechFly").get_filename()).parents[1]


def run_simulation(ind, gen, args):
    """ Main function """

    run_time = 3.0
    time_step = 1e-4

    #: Set collision segments
    ground_contact = [
            f"{side}{leg}{segment}"
            for side in ('L', 'R')
            for leg in ('F', 'M', 'H')
            for segment in tuple(f"Tarsus{i}" for i in range(1, 6))
    ]

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


    #: Get experiment information
    path = os.path.abspath(args.path)
    exp_name = path.split('/')[-1]
    exp_info = exp_name.split('_')

    exp = f"{exp_name}"
    gen = gen


    # Set the ball specs based on your experimental setup
    ball_specs = {
        'ball_mass': 54.6e-6,
        'ball_friction_coef': 1.3
    }
    # Simulation options
    sim_options = {
        "headless": not args.gui,
        "model": str(model_path),
        "model_offset": [0., 0., 11.2e-3],
        "run_time": run_time,
        "time_step": time_step,
        "pose": pose_path,
        "base_link": 'Thorax',
        "controller": controller_path,
        "ground_contacts": ground_contact,
        # 'self_collisions': self_collision,
        "draw_collisions": False, #: Set True to change the color of colliding segments
        "record": args.record, #: Set True to record the simulation
        'camera_distance': 4.5,
        'track': False,
        'moviename': f'{exp}_gen_{gen}_sol_{ind}.mp4',
        'moviespeed': 0.1, #: Speed of the recorded movie
        'slow_down': False,
        'sleep_time': 1e-3,
        'rot_cam': True, #: Set true to rotate the camera automatically
        'ground': 'ball',
        **ball_specs
    }
    #: Initialize the container
    container = Container(run_time / time_step)
    animal = neuromuscular_control.DrosophilaSimulation(container, sim_options)
    #: Load the results of the latest generation of the last optimization run
    fun_path = os.path.join(
        neuromechfly_path,
        'scripts/neuromuscular_optimization',
        args.path,
        'FUN.' + gen
        )
    var_path = os.path.join(
        neuromechfly_path,
        'scripts/neuromuscular_optimization',
        args.path,
        'VAR.' + gen
        )
    fun, var = np.loadtxt(fun_path), np.loadtxt(var_path)

    #: Select which solution to use (fastest, in this case)
    ind = animal.select_solution(ind, fun)
    params = var[ind]
    params = np.array(params)

    #: Run the selected parameter values in the simulation
    animal.update_parameters(params)
    animal.run(optimization=True)
    #: Dump the optimization results
    results_path = os.path.join(
        neuromechfly_path,
        'scripts/neuromuscular_optimization',
        f"simulation_{exp}",
        f"gen_{gen}",
    )

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    if args.plot:
        from NeuroMechFly.utils.plot_contacts import plot_gait_diagram
        contact_flag_data = animal.container.physics.contact_flag.log
        contact_flag_names = animal.container.physics.contact_flag.names
        contact_flag = {
            name: contact_flag_data[:,i] for i, name in enumerate(contact_flag_names)
        }

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16,4))
        ax1.scatter(fun[:,0], fun[:,1])
        ax1.scatter(fun[ind,0], fun[ind,1], label=f'Ind: {ind}')
        ax1.set_xlabel('Distance')
        ax1.set_ylabel('Stability')
        ax1.legend()

        # plt.savefig(os.path.join(results_path, f'selected_sol_{ind}.png'))
        duty_factor = [round(dut,2) for dut in animal.duty_factor]
        plot_gait_diagram(data=contact_flag, export_path=results_path, ax=ax2)
        ax2.set_title(f'Duty factor: {duty_factor}')
        ax1.set_title(f'Speed: {round(animal.ball_rotations[0]*5/3, 2)} mm/s')
        fig.savefig(os.path.join(results_path, f'../selected_sol_{ind}.png'))
        # plt.show()


def parse_args():
    """ Parse arguments from command line """
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', default='')
    parser.add_argument('--gui', dest='gui', action='store_true')
    parser.add_argument('--record', dest='record', action='store_true')
    parser.add_argument('--plot', dest='plot', action='store_true')
    parser.set_defaults(gui=False, profile=False)
    return parser.parse_args()


if __name__ == "__main__":
    """ Main """
    # parse cli arguments
    cli_args = parse_args()
    population = 200
    frequency = 5
    ncores = 10
    generations = np.arange(0,80,10)

    for gen in generations:
        #: Parallelize
        with Pool(processes=ncores) as pool:
            pool.starmap(
                run_simulation,
                [
                    (sol, str(gen), cli_args)
                    for sol in np.arange(0, population, frequency)
                ]
            )