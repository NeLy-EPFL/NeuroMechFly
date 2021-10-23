import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from NeuroMechFly.utils.plotting import plot_data, plot_penalties, plot_pareto_front, plot_fly_path
from NeuroMechFly.utils.plot_contacts import plot_gait_diagram

joints = ['Coxa','Femur', 'Tibia']
legs = ['LF','LM','LH','RF','RM','RH']

def plot_gait_diagram(contact_data, title, export_path):
    """ Plot gait diagrams. """
    fig, ax = plt.subplots(figsize = (7,3))
    plot_gait_diagram(data=contact_data, ax=ax)
    ax.set_xlim(0.5,1.5)
    ax.set_title(title)
    save_name = title.split(' ')
    fig.savefig(os.path.join(export_path, '_'.join(save_name).lower()))
    plt.show()


def plot_pareto_with_selected_inds(fun_parent, var_parent, gen, solutions_to_plot, export_path):

    fun_path = os.path.join(fun_parent, f'FUN.{gen}')
    var_path = os.path.join(var_parent, f'VAR.{gen}')
    fun, var = np.loadtxt(fun_path), np.loadtxt(var_path)

    fig, ax = plt.subplots(figsize = (8,6))
    ax.scatter(fun[:,0], fun[:,1])
    for sol_criteria in solutions_to_plot:
        ind_number = _select_solution(sol_criteria, fun)
        ax.scatter(fun[ind_number,0], fun[ind_number,1], label=f'Sol {sol_criteria}')

    ax.set_xlabel('Distance')
    ax.set_ylabel('Stability')
    ax.legend()
    fig.savefig(os.path.join(export_path, 'pareto_selected_solutions'))
    plt.show()


def plot_ball_path(contact_data, title, export_path):
    plot_fly_path(
        path_data,
        generations=None,
        solutions=None,
        sequence=False,
        heading=True,
        ball_radius = 5.0,
        begin=0,
        end=0,
        time_step=0.001
        )

def plot_joint_info(path_data):

    plot_data(
            path_data,
            leg_key='RF',
            joint_key='ThC',
            sim_data='walking',
            angles={},
            plot_muscles_act=True,
            plot_torques_muscles=True,
            plot_angles_interleg=True,
            plot_angles_intraleg=True,
            plot_torques=False,
            plot_grf=False,
            plot_collisions=False,
            collisions_across=False,
            begin=0.0,
            end=0.0,
            time_step=5e-4,
            torqueScalingFactor=1e9,
            grfScalingFactor=1e6
        )


def _select_solution(criteria, fun):
    """ Selects a solution given a criteria.

    Parameters
    ----------
    criteria: <str>
        criteria for selecting a solution

    fun: <list>
        Solutions from optimization

    Returns
    -------
    out : Index of the solution fulfilling the criteria
    #TODO: Check this

    """
    norm_fun = (fun - np.min(fun, axis=0)) / \
        (np.max(fun, axis=0) - np.min(fun, axis=0))

    if criteria == 'fastest':
        return np.argmin(norm_fun[:, 0])
    if criteria == 'slowest':
        return np.argmax(norm_fun[:, 0])
    if criteria == 'tradeoff':
        return np.argmin(np.sqrt(norm_fun[:, 0]**2 + norm_fun[:, 1]**2))
    if criteria == 'medium':
        mida = mid(norm_fun[:, 0])
        midb = mid(norm_fun[:, 1])
        return np.argmin(
            np.sqrt((norm_fun[:, 0] - mida)**2 + (norm_fun[:, 1] - midb)**2))
    return int(criteria)

if __name__ == '__main__':
    plot_joint_info('/home/nely/nmf-revision/NeuroMechFly/scripts/neuromuscular_optimization/simulation_run_Drosophila_var_63_obj_2_pop_200_gen_40_211022_210331/gen_39/sol_fastest')