#!/usr/bin/env python

""" Script to generate set of optimization videos """
import os
from argparse import ArgumentParser
from multiprocessing import Pool
from pathlib import Path

import farms_pylog as pylog
import numpy as np

from NeuroMechFly.simulation.bullet_simulation import DrosophilaSimulation
from NeuroMechFly.container import Container


def generate_videos(export_path, results_path, frequency, ncores):
    """ Generate a set of optimization videos

    Parameters
    ----------
    export_path: <str>
        Path to dump the videos. If folder already exists then overwrite videos
    results_path: <str>
        Path to the optimization results folder containing FUN and VAR files
    frequency: <int>
        Frequency of generations to generate videos for
    ncores : <int>
        Number of cores to use for parallelization
    """
    export_path = Path(export_path)
    # Create directory if it doesn't exist
    pylog.debug(f"Creating directory {export_path}")
    export_path.mkdir(exist_ok=False)
    # Results path
    results_folder_path = Path(results_path)
    if not results_folder_path.exists():
        raise ValueError(f"Path {results_folder_path} doesn't exist")
    # Get optimization info from folder name
    opti_options = results_folder_path.name.split('_')
    population = int(opti_options[7])
    generations = int(opti_options[9])

    #: Parallelize
    with Pool(processes=ncores) as pool:
        pool.starmap(
            run_simulation,
            [
                (get_params_from_optimization_results(
                    results_folder_path.joinpath(f"FUN.{gen}"),
                    results_folder_path.joinpath(f"VAR.{gen}"),
                ), export_path.joinpath(f"gen_{gen}.mp4"))
                for gen in np.arange(0, generations, frequency)
            ]
        )

    # # Run the simulations
    # for gen in np.arange(0, generations, frequency):
    #     fun, var = read_optimization_results(
    #         results_folder_path.joinpath(f"FUN.{gen}"),
    #         results_folder_path.joinpath(f"VAR.{gen}"),
    #     )
    #     params = np.asarray(var[np.argmin(fun[:,0]*fun[:,1])])
    #     run_simulation(
    #         params, export_path.joinpath(f"gen_{gen}.mp4")
    #     )


def get_params_from_optimization_results(fun, var):
    """ Read optimization results.
    Parameters
    ----------
    fun: <Path>
        Path to the fun file location
    var: <Path>
        Path to the var file location

    """
    fun, var = np.loadtxt(fun), np.loadtxt(var)
    params = np.asarray(var[np.argmin(fun[:,0]* fun[:,1])])
    return params


def run_simulation(params, export_path):
    """ Run the simulation for a given generations

    Parameters
    ----------
    params: <np.ndarray>
        Parameters for the simulation
    export_path: <str>
         export path for the video with name

    """
    run_time = 5.0
    time_step = 0.001

    side = ['L','R']
    pos = ['F','M','H']
    leg_segments = ['Femur','Tibia']+['Tarsus' + str(i) for i in range(1, 6)]

    ground_contact = [s+p+name for s in side for p in pos for name in leg_segments if 'Tarsus' in name]

    left_front_leg = ['LF'+name for name in leg_segments]
    left_middle_leg = ['LM'+name for name in leg_segments]
    left_hind_leg = ['LH'+name for name in leg_segments]

    right_front_leg = ['RF'+name for name in leg_segments]
    right_middle_leg = ['RM'+name for name in leg_segments]
    right_hind_leg = ['RH'+name for name in leg_segments]

    body_segments = ['A1A2','A3','A4','A5','A6','Thorax','Head']

    self_collision = []
    for link0 in left_front_leg:
        for link1 in left_middle_leg:
            self_collision.append([link0,link1])
    for link0 in left_middle_leg:
        for link1 in left_hind_leg:
            self_collision.append([link0,link1])
    for link0 in left_front_leg:
        for link1 in body_segments:
            self_collision.append([link0,link1])
    for link0 in left_middle_leg:
        for link1 in body_segments:
            self_collision.append([link0,link1])
    for link0 in left_hind_leg:
        for link1 in body_segments:
            self_collision.append([link0,link1])

    for link0 in right_front_leg:
        for link1 in right_middle_leg:
            self_collision.append([link0,link1])
    for link0 in right_middle_leg:
        for link1 in right_hind_leg:
            self_collision.append([link0,link1])
    for link0 in right_front_leg:
        for link1 in body_segments:
            self_collision.append([link0,link1])
    for link0 in right_middle_leg:
        for link1 in body_segments:
            self_collision.append([link0,link1])
    for link0 in right_hind_leg:
        for link1 in body_segments:
            self_collision.append([link0,link1])

    print(export_path)
    sim_options = {
        "headless": False,
        # Scaled SDF model
        "model": "../../design/sdf/neuromechfly_limitsFromData_minMax.sdf",
        "model_offset": [0., 0., 11.2e-3],
        "run_time": run_time,
        "pose": '../../config/test_pose_tripod.yaml',
        "base_link": 'Thorax',
        "controller": '../../config/locomotion_ball.graphml',
        "ground_contacts": ground_contact,
        'self_collisions': self_collision,
        "draw_collisions": True,
        "record": True,
        'camera_distance': 3.5,
        'track': False,
        'moviename': export_path,
        'rot_cam': True
        }

    container = Container(run_time/time_step)
    animal = DrosophilaSimulation(container, sim_options)
    # Update parameters
    animal.update_parameters(params)
    # Run simulation
    animal.run(optimization=False)


def parse_args():
    """ Pargse cli arguments """
    parser = ArgumentParser("Generate optimization videos")
    parser.add_argument(
        "--results_path", "-r", type=str, dest="results_path",
        required=True
    )
    parser.add_argument(
        "--export_path", "-e", type=str, dest="export_path",
        required=True
    )
    parser.add_argument(
        "--frequency", "-f", type=int, dest="frequency",
        required=True
    )
    parser.add_argument(
        "--ncores", "-n", type=int, dest="ncores",
        required=False, default=1
    )
    return vars(parser.parse_args())


if __name__ == '__main__':
    #: generate videos
    generate_videos(**parse_args())
