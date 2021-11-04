""" Script to plot the simulation results. """

import os
from typing import List
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats
import scipy.signal
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from .sensitivity_analysis import calculate_forces
from scipy.interpolate import pchip_interpolate


def plot_mu_sem(
    mu,
    error,
    conf=None,
    plot_label='Mean',
    x=None,
    alpha=0.3,
    color=None,
    ax=None,
    beg=0,
    time_step=0.001,
    end=100,
):
    """ Plots mean, confidence interval, and standard deviation (Author: JB)

    Parameters
    ----------
    mu: <np.array>
        Mean, shape [N_samples, N_lines] or [N_samples].
    error: <np.array>
        Error to be plotted, e.g. standard error of the mean, shape [N_samples, N_lines] or [N_samples].
    conf: <int>
        Confidence interval, if none, stderror is plotted instead of std.
    plot_label: <str>
        The label for each line either a string if only one line or list of strings if multiple lines.
    x: <np.array>
        shape [N_samples]. If not specified will be np.arange(mu.shape[0]).
    alpha: <float>
        Transparency of the shaded area. default 0.3.
    color:
        Pre-specify colour. if None, use Python default colour cycle.
    ax:
        axis to be plotted on, otherwise the current is axis with plt.gca().
    """
    if ax is None:
        ax = plt.gca()
    if x is None:
        x = np.arange(0, mu.shape[0], 1) * time_step
    p = ax.plot(x[beg:end], mu[beg:end], lw=1, color=color, label=plot_label)
    if len(mu.shape) == 1:
        if conf is not None:
            ax.plot(x[beg:end],
                    mu[beg:end] - conf * error[beg:end],
                    alpha=alpha,
                    linewidth=1.5,
                    linestyle=':',
                    color='black',
                    label="Confidence Interval {}%".format(conf))
            ax.plot(x[beg:end], mu[beg:end] + conf * error[beg:end],
                    alpha=alpha, linewidth=1.5, linestyle=':', color='black')
        ax.fill_between(x[beg:end],
                        mu[beg:end] - error[beg:end],
                        mu[beg:end] + error[beg:end],
                        alpha=alpha,
                        color=p[0].get_color())
    else:
        for i in np.arange(mu.shape[1]):
            if conf is not None:
                ax.plot(x[beg:end],
                        mu[beg:end,
                           i] - conf * error[beg:end,
                                             i],
                        alpha=alpha,
                        linewidth=1.5,
                        linestyle=':',
                        color='black',
                        label="Confidence Interval {}%".format(conf))
                ax.plot(x[beg:end], mu[beg:end, i] + conf * error[beg:end, i],
                        alpha=alpha, linewidth=1.5, linestyle=':', color='black')
            ax.fill_between(x[beg:end], mu[beg:end, i] -
                            error[beg:end, i], mu[beg:end, i] +
                            error[beg:end, i], alpha=alpha, color=p[i].get_color())


def plot_kp_joint(
    *args,
    show_vector=False,
    calc_force=False,
    full_name='joint_LMTibia',
    gain_range=np.arange(0.1, 1.1, 0.2),
    scaling_factor=1,
    ax=None,
    constant='kv0.9',
    condition='kp0.4_kv0.9',
    beg=2000,
    intv=250,
    time_step=0.001,
    ground_truth=None
):
    """Plot the joint info of one specific leg versus independent variable.

    Parameters
    ----------
    *args: <np.array>
        Force to be plotted, i.e. grf, lateral friction, thorax.
    multiple: <bool>
        Plots vectors instead of norm.
    data: <dictionary>
        Dictionary to be plotted, i.e. joint torques.
    full_name: <str>
        Key name, e.g., 'joint_LMTibia'.
    gain_range: <np.array>
        Range of gains to be plotted, i.e. np.arange(0.1,1.4,0.2).
    scaling_factor: <int>
        Scale to change the units.
    ax:
        Axis to be plotted on, otherwise the current is axis with plt.gca().
    beg: <int>
        Beginning of the data to be plotted. the entire data is long.
    intv: <int>
        Int of the data to be plotted.
    ground_truth: <np.array>
        Ground truth for position or velocity.
    """
    if ax is None:
        ax = plt.gca()
    if ground_truth is not None:
        ax.plot(np.array(ground_truth[beg:beg + intv]) * scaling_factor,
                linewidth=2.5, color="red", label="Ground Truth")

    for k in gain_range:
        k_value = "_".join((constant, 'kv' +
                            str(round(k, 1)))) if 'kp' in constant else "_".join(('kp' +
                                                                                  str(round(k, 1)), constant))

        color = plt.cm.winter(np.linalg.norm(k))
        if condition == k_value:
            color = 'red'

        if not calc_force:
            time = np.arange(
                0, len(
                    args[0][k_value][full_name]), 1) * time_step
            ax.plot(time[beg: beg +
                         intv], np.array(args[0][k_value][full_name][beg:beg +
                                                                     intv]) *
                    scaling_factor, color=color, label=k_value)
            ax.legend(bbox_to_anchor=(1.1, 1), loc='upper right')

        else:
            vector, norm = calculate_forces(full_name, k_value, *args)
            if show_vector:
                for i, axis in enumerate(['x', 'y', 'z']):
                    time = np.arange(0, len(vector), 1) * time_step
                    ax[i].plot(time[beg: beg +
                                    intv], np.array(vector[i, beg:beg +
                                                           intv]) *
                               scaling_factor, color=color, label=k_value)
                    ax[i].set_ylabel(axis)
                ax.legend(bbox_to_anchor=(1.1, 0.), loc='upper right')
            else:
                time = np.arange(0, len(norm), 1) * time_step
                ax.plot(time[beg: beg + intv], norm[beg:beg + intv]
                        * scaling_factor, color=color, label=k_value)
                ax.legend(bbox_to_anchor=(1.1, 1), loc='upper right')


def heatmap_plot(
        title,
        joint_data,
        colorbar_title,
        precision="g",
        linewidth="0.005",
        ax=None,
        cmap='viridis'):
    """ Plots a heatmap plot for global sensitivity analysis.

    Parameters
    ----------
    title: <str>
        Title of the heatmap.
    joint_data: <dict>
        Dictionary containing the joint information (angle etc).
    colorbar_title: <str>
        Title of the colorbar.
    precision: <str>
        Precision of the heatmap entries.
    linewidth: <str>
        Width of the lines in heatmap.
    ax:
        Axis to be plotted on, otherwise plt.gca().
    cmap: <str>
        Color map of the heatmap.
    """
    if ax is None:
        ax = plt.gca()

    ax = sns.heatmap(
        joint_data,
        annot=True,
        ax=ax,
        linewidth=linewidth,
        cmap=cmap,
        vmin=np.nanmin(joint_data),
        vmax=np.nanmax(joint_data),
        fmt=precision,
        cbar_kws={
            'label': colorbar_title})
    ax.set_title(title)
    ax.invert_yaxis()


def plot_pareto_gens(
    parent_dir,
    generations,
    inds_to_annotate,
    export_path=None
):
    """ Plots multiple generations with selected individuals.

    Parameters
    ----------
    parent_dir : <str>
        Directory where the FUN and VAR files are located.
    generations : <np.darray>
        Generations to be plotted.
    inds_to_annotate : <dict>
        A dictionary with keys 'gen<number>' and values int or string ('fastest' etc.) format
    export_path: <str>
        Path at which the plot will be saved.

    Example usage:
        plot_pareto_gens(
        parent_dir='/home/NeuroMechFly/scripts/neuromuscular_optimization/optimization_results/run_Drosophila_var_63_obj_2_pop_200_gen_100_211022_134952',
        generations=np.arange(15,100,14),
        inds_to_annotate = {
            'gen15': [0,3,5],
            'gen29': 12,
            'gen57': 14,
            'gen85': 48
        },
        export_path='./pareto.png'
        )

        OR

        plot_pareto_gens(
        parent_dir='/home/NeuroMechFly/scripts/neuromuscular_optimization/optimization_results/run_Drosophila_var_63_obj_2_pop_200_gen_100_211022_134952',
        generations=99,
        inds_to_annotate = {
            'gen99': ['fastest', 'win_win', 'most_stable']
        },
        export_path='./pareto.png'
        )

    """

    from NeuroMechFly.experiments.network_optimization.neuromuscular_control import DrosophilaSimulation as ds
    # import directly from collections for Python < 3.3
    from collections.abc import Iterable

    rc_params = {
        'axes.spines.right': False,
        'axes.spines.top': False,
    }
    plt.rcParams.update(rc_params)
    colors = (
        '#B4479A',
        '#3953A4',
        '#027545',
        '#808080',
        '#FE420F',
        '#650021',
        '#E6DAA6',
        '#008080',
        '#FFC0CB')

    if not isinstance(generations, Iterable):
        generations = [generations]

    fig, ax = plt.subplots(figsize=(7, 4))

    for i, gen in enumerate(generations):
        fun_path = os.path.join(parent_dir, f'FUN.{gen}')
        var_path = os.path.join(parent_dir, f'VAR.{gen}')
        fun, var = np.loadtxt(fun_path), np.loadtxt(var_path)

        ax.scatter(fun[:, 0], fun[:, 1], c=colors[i % len(colors)], alpha=0.3, s=30, label=f'Gen {gen+1}')
        if 'gen' + str(gen) in inds_to_annotate:
            if not isinstance(inds_to_annotate['gen' + str(gen)], Iterable):
                individuals = [inds_to_annotate['gen' + str(gen)]]
            else:
                individuals = inds_to_annotate['gen' + str(gen)]

            for j, ind_ in enumerate(individuals):
                cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
                ind_number = ds.select_solution(ind_, fun)
                if len(generations) > 1:
                    ax.scatter(fun[ind_number, 0], fun[ind_number, 1],
                               s=95, c=colors[i % len(colors)], edgecolor='black')
                else:
                    ax.scatter(fun[ind_number, 0], fun[ind_number, 1], label=f'Sol {ind_}', s=60, edgecolor='black', c=cycle[j])

    ax.set_xlabel('Distance')
    ax.set_ylabel('Stability')
    ax.legend()
    if export_path is not None:
        fig.savefig(export_path, bbox_inches='tight')
    plt.show()


def plot_population_statistics(
    result_directory,
    pop_no,
    generations,
    penalty_number,
    penalty_name,
    export_path=None
):
    """ Plots the population statistics (i.e. penalties across generations)

    Parameters
    ----------
    result_directory : str
        Directory where the PENALTIES.<gen> are.
    pop_no : int
        Number of individual in a population.
    generations : list
        Generations to be analyzed.
        e.g.: np.arange(15,100,14)
    penalty_number : int
        Column number of the penalty based on the log file.
    penalty_name : str
        Name of the penalty
    export_path: <str>
        Path at which the plot will be saved if not None
    """
    rc_params = {
        'axes.spines.right': False,
        'axes.spines.top': False,
    }
    plt.rcParams.update(rc_params)

    fig, ax = plt.subplots(figsize=(4, 6))
    penalty = np.zeros((pop_no, len(generations)))

    for i, generation in enumerate(generations):
        penalty[:, i] = np.loadtxt(
            os.path.join(
                result_directory, f'PENALTIES.{generation}'
            )
        )[:, penalty_number]

    cols = [f'{gen + 1}' for gen in generations]
    rows = [f'Ind {ind}' for ind in range(pop_no)]

    invert = -1 if penalty_number in (0, 1) else 1
    penalty_df = pd.DataFrame(invert * penalty, columns=cols, index=rows)
    sns.stripplot(
        data=penalty_df,
        size=5,
        color='red',
        edgecolor='black',
        alpha=0.3,
        ax=ax)
    sns.violinplot(
        data=penalty_df,
        scale='count',
        color='white',
        edgecolor='black',
        bw=0.35,
        ax=ax,
        cut=0,
        showfliers=False,
        showextrema=False)

    ax.set_xlabel('Generations')
    ax.set_title(penalty_name)

    if export_path is not None:
        fig.savefig(export_path, bbox_inches='tight')
    plt.show()


def plot_gait_diagram(data, ts=1e-4, ax=None, export_path=None):
    """ Plot the contacts from the given contact_flag data.

    Parameters
    ----------
    data: <pandas.DataFrame>
        Contact flag data.
    ts: <float>
        Time step of the simulation.
    ax:
        axis to be plotted on, otherwise the current is axis with plt.gca().
    export_path: <str>
        Path at which the plot will be saved.
    """
    # Total time
    total_time = len(data) * ts
    # Define the legs and its order for the plot
    legs = ("RH", "RM", "RF", "LH", "LM", "LF")
    # Setup the contact data
    contact_intervals = {}
    for leg in legs:
        # Combine contact information of all the tarsus segments
        values = np.squeeze(np.any(
            [value for key, value in data.items() if leg in key],
            axis=0,
        ).astype(int))
        intervals = np.where(
            np.abs(np.diff(values, prepend=[0], append=[0])) == 1
        )[0].reshape(-1, 2) * ts
        intervals[:, 1] = intervals[:, 1] - intervals[:, 0]
        contact_intervals[leg] = intervals
    # Define the figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3))
    width = 0.75
    for index, (key, value) in enumerate(contact_intervals.items()):
        ax.broken_barh(
            value, (index - width * 0.5, width), facecolor='k'
        )
    ax.set_xlabel("Time (s)")
    ax.set_yticks((0, 1, 2, 3, 4, 5))
    ax.set_yticklabels(legs)

    if export_path is not None:
        plt.savefig(export_path, bbox_inches='tight')


def load_opt_log(results_path):
    """ Loads the optimization muscle torques and joint position results.

    Parameters
    ----------
    results_path: str
        Directory containing the muscle, neural and physics folders.

    Returns
    ----------
    (muscle, joint_pos): Tuple
        Muscle and joint positions both in pandas.DataFrame format.
    """
    muscle_path = os.path.join(results_path, 'muscle/outputs.h5')
    joint_angle_path = os.path.join(results_path, 'physics/joint_positions.h5')
    muscle, joint_pos = pd.read_hdf(muscle_path), pd.read_hdf(joint_angle_path)
    return muscle, joint_pos


def plot_network_activity(
        results_path,
        time_step=1e-4,
        sim_duration=2.0,
        beg=1,
        end=1.5,
        torque_scale=1e9,
        link='Femur',
        export_path=None):
    """ Plots the CPG activity, muscle torques and joint angles.

    Parameters
    ----------
    results_path: str
        Directory containing the muscle, neural and physics folders.
    time_step : float, optional
        Time step of the simulation, by default 1e-4
    sim_duration : float, optional
        Duration of the simulation in seconds, by default 2.0
    beg : int, optional
        Beginning from which the data will be plotted, by default 1
    end : float, optional
        Beginning at which the data will end, by default 1.5
    torque_scale : [type], optional
        Conversion scale from SI units to uNmm, by default 1e9
    link : str, optional
        Link to be plotted, by default 'Femur', could be 'Coxa' or 'Tibia' as well.
    export_path : str, optional
        If not None then the plot will be saved to that path, by default None
    """
    from matplotlib.gridspec import GridSpec

    # Load data
    muscle, joint_pos = load_opt_log(results_path)

    rc_params = {'axes.spines.right': False, 'axes.spines.top': False}
    plt.rcParams.update(rc_params)
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    linestyles = ('solid', 'dashed', 'dotted')

    equivalence = {'ThC_yaw': 'Coxa_yaw',
                   'ThC_pitch': 'Coxa',
                   'ThC_roll': 'Coxa_roll',
                   'CTr_pitch': 'Femur',
                   'CTr_roll': 'Femur_roll',
                   'FTi_pitch': 'Tibia',
                   'TiTa_pitch': 'Tarsus1'}

    actuated_joints = {
        'F': ['ThC_pitch', 'CTr_pitch', 'FTi_pitch'],
        'M': ['ThC_roll', 'CTr_pitch', 'FTi_pitch'],
        'H': ['ThC_roll', 'CTr_pitch', 'FTi_pitch'],

    }
    sides = ['Front', 'Middle', 'Hind']

    duration = np.arange(0, sim_duration, time_step)
    beg = int(beg / time_step)
    end = int(end / time_step)

    fig = plt.figure(figsize=(8, 8))
    gs = plt.GridSpec(4, 3, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
    ax3 = fig.add_subplot(gs[2, :])
    (ax4, ax5, ax6) = fig.add_subplot(gs[3, 0]), fig.add_subplot(
        gs[3, 1]), fig.add_subplot(gs[3, 2])

    for i, side in enumerate(sides):
        part = side[0]
        if part in ('M', 'H') and link[0] == 'C':
            link = 'Coxa_roll'

        ax1.plot(duration[beg:end], muscle[f'joint_R{part}{link}_flexor_act'][beg:end], label=f'R{part}', color=cycle[i * 2])
        ax1.plot(duration[beg:end], muscle[f'joint_L{part}{link}_flexor_act'][beg:end], label=f'L{part}', color=cycle[i * 2 + 1])
        ax1.legend(bbox_to_anchor=(1.1, 1))
        ax1.set_ylabel(f'{link} Protractor (AU)')

        ax2.plot(duration[beg:end], muscle[f'joint_R{part}{link}_torque'][beg:end] * torque_scale, color=cycle[i * 2])
        ax2.plot(duration[beg:end], muscle[f'joint_L{part}{link}_torque'][beg:end] * torque_scale, color=cycle[i * 2 + 1])
        ax2.set_ylabel(f'{link} Torques ($\mu$Nmm)')

        ax3.plot(duration[beg:end], np.rad2deg(joint_pos[f'joint_R{part}{link}'][beg:end]), color=cycle[i * 2])
        ax3.plot(duration[beg:end], np.rad2deg(joint_pos[f'joint_L{part}{link}'][beg:end]), color=cycle[i * 2 + 1])
        ax3.set_ylabel(f'{link} Joint Angles(deg)')
        ax3.set_xlabel('Time (s)')

        for j, joint_angle in enumerate(actuated_joints[part]):
            ls = linestyles[j] if not joint_angle == 'ThC_pitch' else 'dashdot'
            if part == 'F':
                ax4.plot(duration[beg:end], np.rad2deg(joint_pos[f'joint_L{part}{equivalence[joint_angle]}'][beg:end]), label='LF ' + joint_angle, color=cycle[1], linestyle=ls)
            elif part == 'M':
                ax5.plot(duration[beg:end], np.rad2deg(joint_pos[f'joint_L{part}{equivalence[joint_angle]}'][beg:end]), label='LM ' + joint_angle, color=cycle[3], linestyle=ls)
            if part == 'H':
                ax6.plot(duration[beg:end], np.rad2deg(joint_pos[f'joint_L{part}{equivalence[joint_angle]}'][beg:end]), label='LH ' + joint_angle, color=cycle[5], linestyle=ls)
        ax4.legend(
            loc='upper center', bbox_to_anchor=(
                0.5, -0.1), fancybox=True)
        ax5.legend(
            loc='upper center', bbox_to_anchor=(
                0.5, -0.1), fancybox=True)
        ax6.legend(
            loc='upper center', bbox_to_anchor=(
                0.5, -0.1), fancybox=True)
        ax4.set_ylabel('Joint Angles(deg)')

    plt.tight_layout()
    if export_path is not None:
        plt.savefig(export_path, bbox_inches='tight')
    plt.show()


def read_ground_contacts(path_data):
    """ Reads ground contact's data obtained after running a simulation.

    Parameters
    ----------
    path_data: <str>
        Path to simulation results.

    Returns
    ----------
    grf: <dict>
        Ground reaction forces for all segments in all legs.
    """
    grf_data = os.path.join(path_data, 'physics', 'contact_normal_force.h5')
    data = pd.read_hdf(grf_data)
    grf = {}
    check = []
    for key, force in data.items():
        leg, force_axis = key.split('_')
        if leg not in check and "-" not in leg:
            check.append(leg)
            components = [k for k in data.keys() if leg in k and "-" not in k]
            data_x = data[components[0]].values
            data_y = data[components[1]].values
            data_z = data[components[2]].values
            res_force = np.linalg.norm([data_x, data_y, data_z], axis=0)
            if leg[:2] not in grf.keys():
                grf[leg[:2]] = []
            grf[leg[:2]].append(res_force)

    return grf


def read_collision_forces(path_data):
    """ Reads collision force's data obtained after running a simulation.

    Parameters
    ----------
    path_data: <str>
        Path to simulation results.

    Returns
    ----------
    collisions: <dict>
        Collision forces for all segments in all legs.
    """
    collisions_data = os.path.join(path_data, 'physics', 'collision_forces.h5')
    data = pd.read_hdf(collisions_data)

    collisions = {}
    check = []
    for key in data.keys():
        body_parts, force_axis = key.split('_')
        segment1, segment2 = body_parts.split('-')
        if body_parts not in check:
            check.append(body_parts)
            components = [k for k in data.keys() if body_parts in k]
            data_x = data[components[0]].values
            data_y = data[components[1]].values
            data_z = data[components[2]].values
            res_force = np.linalg.norm([data_x, data_y, data_z], axis=0)
            if segment1 not in collisions.keys():
                collisions[segment1] = {}
            if segment2 not in collisions.keys():
                collisions[segment2] = {}
            collisions[segment1][segment2] = res_force
            collisions[segment2][segment1] = res_force

    return collisions


def get_stance_periods(leg_force, start, stop):
    """ Get stance periods from GRF data.

    Parameters
    ----------
    leg_force: <np.array>
        GRF data associated with a leg.
    start: <float>
        Starting time for checking stance periods.
    stop: <float>
        Stoping time for checking stance periods.

    Returns
    ----------
    stance_plot: <list>
        Indices indicating beginning and ending of stance periods.
    """
    stance_ind = np.where(leg_force > 0)[0]
    if stance_ind.size != 0:
        stance_diff = np.diff(stance_ind)
        stance_lim = np.where(stance_diff > 1)[0]
        stance = [stance_ind[0] - 1]
        for ind in stance_lim:
            stance.append(stance_ind[ind] + 1)
            stance.append(stance_ind[ind + 1] - 1)
        stance.append(stance_ind[-1])
        start_gait_list = np.where(np.array(stance) >= start)[0]
        if len(start_gait_list) > 0:
            start_gait = start_gait_list[0]
        else:
            start_gait = start
        stop_gait_list = np.where(
            (np.array(stance) <= stop) & (
                np.array(stance) > start))[0]
        if len(stop_gait_list) > 0:
            stop_gait = stop_gait_list[-1] + 1
        else:
            stop_gait = start_gait
        if start_gait != stop_gait:
            stance_plot = stance[start_gait:stop_gait]
            if start_gait % 2 != 0:
                stance_plot.insert(0, start)
            if len(stance_plot) % 2 != 0:
                stance_plot.append(stop)
        else:
            stance_plot = [start, start]
    else:
        stance_plot = [start, start]

    return stance_plot


def plot_data(
    path_data,
    leg_key='RF',
    sim_data='walking',
    angles={},
    plot_angles_intraleg=False,
    plot_torques=True,
    plot_grf=True,
    plot_collisions=True,
    collisions_across=True,
    begin=0.0,
    end=0.0,
    time_step=5e-4,
    torqueScalingFactor=1e9,
    grfScalingFactor=1e6
):
    """ Plots data from the simulation.

    Parameters
    ----------
    path_data: <str>
        Path to simulation results.
    leg_key: <str>
        Key for specifying a leg to plot: angles (intraleg or interleg), torques, grf, or collisions. Options: 'LF', 'LM', 'LH', 'RF', 'RM', 'RH'.
    sim_data: <str>
        Behavior from data. Options: 'walking' or 'grooming'.
    plot_angles_intraleg: <bool>
        Plotting joint angles from all joints in leg 'leg_key'.
    plot_torques: <bool>
        Plotting torques generated by PyBullet controllers.
    plot_grf: <bool>
        Plotting ground reaction forces (if sim_data='walking').
    plot_collisions: <bool>
        Plotting self-collision forces (if sim_data='grooming').
    plot_collisions_across: <bool>
        Plotting grf/collisions as gray background across other plots.
    begin: <float>
        Starting time for initiating the plots.
    end: <float>
        Stoping time for finishing the plots. If 0.0, all data is plotted.
    time_step: <float>
        Data time step.
    torqueScalingFactor: <float>
        Scaling factor for torques (from Nm to uNmm).
    grfScalingFactor: <float>
        Scaling factor for ground reaction forces (from N to uN).
    """
    data2plot = {}

    equivalence = {'ThC_yaw': 'Coxa_yaw',
                   'ThC_pitch': 'Coxa',
                   'ThC_roll': 'Coxa_roll',
                   'CTr_pitch': 'Femur',
                   'CTr_roll': 'Femur_roll',
                   'FTi_pitch': 'Tibia',
                   'TiTa_pitch': 'Tarsus1'}

    leg_order = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']

    length_data = 0


    if plot_angles_intraleg:
        if bool(angles):
            angles_raw = angles[leg_key + '_leg']
        else:
            angles_data = read_joint_positions(
                path_data, equivalence, leg_order)
            angles_raw = angles_data[leg_key + '_leg']
        data2plot['angles'] = {}
        for k in equivalence.keys():
            data2plot['angles'][k] = []
        for label, values in angles_raw.items():
            if length_data == 0:
                length_data = len(values)
            data2plot['angles'][label] = values

    if plot_torques:
        torques_data = os.path.join(path_data, 'physics', 'joint_torques.h5')
        torques_all = pd.read_hdf(torques_data)
        torques_raw = {}
        for joint, torque in torques_all.items():
            if leg_key in joint and 'Haltere' not in joint:
                if 'Tarsus' not in joint or 'Tarsus1' in joint:
                    joint_data = joint.split('joint_')
                    label = joint_data[1][2:]
                    torques_raw[label] = torque.values
        data2plot['torques'] = {}
        for label, match_labels in equivalence.items():
            for key in torques_raw.keys():
                if key == match_labels:
                    if length_data == 0:
                        length_data = len(torques_raw[key])
                    data2plot['torques'][label] = torques_raw[key]

    if plot_grf:
        if sim_data == 'walking':
            data2plot['grf'] = read_ground_contacts(path_data)
            grf_leg = data2plot['grf'][leg_key]
            sum_force = np.sum(np.array(grf_leg), axis=0)
            leg_force = np.delete(sum_force, 0)

    if plot_collisions:
        if sim_data == 'grooming':
            data2plot['collisions'] = read_collision_forces(path_data)
            leg_collisions = []
            ant_collisions = []
            all_collisions = []
            for segment, coll in data2plot['collisions'].items():
                if segment[:2] == leg_key:
                    for k, val in coll.items():
                        all_collisions.append(val)
                        if 'Antenna' not in k:
                            leg_collisions.append(val)
                if 'Antenna' in segment and leg_key[0] == segment[0]:
                    for k, val in coll.items():
                        ant_collisions.append(val)

            sum_all = np.sum(np.array(all_collisions), axis=0)
            sum_leg = np.sum(np.array(leg_collisions), axis=0)
            sum_ant = np.sum(np.array(ant_collisions), axis=0)
            leg_force = np.delete(sum_all, 0)
            leg_vs_leg = np.delete(sum_leg, 0)
            leg_vs_ant = np.delete(sum_ant, 0)

    if end == 0:
        end = length_data * time_step

    steps = 1 / time_step
    start = int(begin * steps)
    stop = int(end * steps)

    if collisions_across:
        if not plot_grf and sim_data == 'walking':
            grf_data = read_ground_contacts(path_data)
            grf_leg = grf_data[leg_key]
            sum_force = np.sum(np.array(grf_leg), axis=0)
            leg_force = np.delete(sum_force, 0)
        if not plot_collisions and sim_data == 'grooming':
            collisions_dict = read_collision_forces(path_data)
            leg_collisions = []
            ant_collisions = []
            all_collisions = []
            for segment, coll in collisions_dict.items():
                if segment[:2] == leg_key:
                    for k, val in coll.items():
                        all_collisions.append(val)
                        if 'Antenna' not in k:
                            leg_collisions.append(val)
                if 'Antenna' in segment and leg_key[0] == segment[0]:
                    for k, val in coll.items():
                        ant_collisions.append(val)

            sum_all = np.sum(np.array(all_collisions), axis=0)
            sum_leg = np.sum(np.array(leg_collisions), axis=0)
            sum_ant = np.sum(np.array(ant_collisions), axis=0)
            leg_force = np.delete(sum_all, 0)
            leg_vs_leg = np.delete(sum_leg, 0)
            leg_vs_ant = np.delete(sum_ant, 0)

        stance_plot = get_stance_periods(leg_force, start, stop)

    fig, axs = plt.subplots(len(data2plot.keys()), sharex=True)
    fig.suptitle('NeuroMechFly Plots')

    torque_min = np.inf
    torque_max = 0
    grf_min = np.inf
    grf_max = 0

    for i, (plot, data) in enumerate(data2plot.items()):

        if plot == 'angles':
            for name, angle_rad in data.items():
                time = np.arange(0, len(angle_rad), 1) / steps
                angle = np.array(angle_rad) * 180 / np.pi
                if len(data2plot.keys()) == 1:
                    axs.plot(time[start:stop], angle[start:stop],
                             label=name.replace('_', ' '))
                else:
                    axs[i].plot(time[start:stop], angle[start:stop],
                                label=name.replace('_', ' '))
            if len(data2plot.keys()) == 1:
                axs.set_ylabel('Joint angles (deg)')
            else:
                axs[i].set_ylabel('Joint angles (deg)')

        if plot == 'torques':
            for joint, torque in data.items():
                torque_adj = np.delete(torque, 0)
                time = np.arange(0, len(torque_adj), 1) / steps
                if len(data2plot.keys()) == 1:
                    axs.plot(time[start:stop], torque_adj[start:stop] *
                             torqueScalingFactor, label=joint.replace('_', ' '))
                else:
                    axs[i].plot(time[start:stop], torque_adj[start:stop]
                                * torqueScalingFactor, label=joint.replace('_', ' '))

                t_min = np.min(torque_adj[start:stop] * torqueScalingFactor)
                t_max = np.max(torque_adj[start:stop] * torqueScalingFactor)

                if t_min < torque_min:
                    torque_min = t_min

                if t_max > torque_max:
                    torque_max = t_max

            if len(data2plot.keys()) == 1:
                axs.set_ylabel('Joint torque ' + r'$(\mu Nmm)$')
                axs.set_ylim(1.2 * torque_min, 1.1 * torque_max)
            else:
                axs[i].set_ylabel('Joint torque ' + r'$(\mu Nmm)$')
                axs[i].set_ylim(1.2 * torque_min, 1.1 * torque_max)

        if plot == 'grf':
            time = np.arange(0, len(leg_force), 1) / steps
            if len(data2plot.keys()) == 1:
                axs.plot(time[start:stop], leg_force[start:stop]
                         * grfScalingFactor, color='black')
                axs.set_ylabel('Ground reaction forces ' + r'$(\mu N)$')
            else:
                axs[i].plot(time[start:stop], leg_force[start:stop]
                            * grfScalingFactor, color='black')
                axs[i].set_ylabel('Ground reaction forces ' + r'$(\mu N)$')
            f_min = np.min(leg_force[start:stop] * grfScalingFactor)
            f_max = np.max(leg_force[start:stop] * grfScalingFactor)

            if f_min < grf_min:
                grf_min = f_min

            if f_max > grf_max:
                grf_max = f_max

            if len(data2plot.keys()) == 1:
                axs.set_ylim(-0.003, 1.1 * grf_max)
            else:
                axs[i].set_ylim(-0.003, 1.1 * grf_max)

        if plot == 'collisions':
            time = np.arange(0, len(leg_force), 1) / steps
            if len(data2plot.keys()) == 1:
                axs.plot(time[start:stop],
                         np.array(leg_vs_leg[start:stop]) * grfScalingFactor,
                         color='black',
                         label='Leg vs leg force')
                axs.plot(time[start:stop],
                         np.array(leg_vs_ant[start:stop]) * grfScalingFactor,
                         color='dimgray',
                         label='Leg vs antenna force')
                axs.set_ylabel('Collision forces ' + r'$(\mu N)$')
            else:
                axs[i].plot(time[start:stop],
                            np.array(leg_vs_leg[start:stop]) * grfScalingFactor,
                            color='black',
                            label='Leg vs leg force')
                axs[i].plot(time[start:stop],
                            np.array(leg_vs_ant[start:stop]) * grfScalingFactor,
                            color='dimgray',
                            label='Leg vs antenna force')
                axs[i].set_ylabel('Collision forces ' + r'$(\mu N)$')

        if len(data2plot.keys()) == 1:
            axs.grid(True)
        else:
            axs[i].grid(True)

        if (plot != 'grf' and i == 0) or ('angles' in plot and plot_angles_interleg):
            if len(data2plot.keys()) == 1:
                plot_handles, plot_labels = axs.get_legend_handles_labels()
            else:
                plot_handles, plot_labels = axs[i].get_legend_handles_labels()
            if collisions_across and sim_data == 'walking':
                gray_patch = mpatches.Patch(color='gray')
                all_handles = plot_handles + [gray_patch]
                all_labels = plot_labels + ['Stance']
            elif sim_data == 'grooming':
                gray_patch = mpatches.Patch(color='dimgray')
                darkgray_patch = mpatches.Patch(color='darkgray')
                if plot_collisions and plot != 'collisions' and collisions_across:
                    dark_line = Line2D([0], [0], color='black')
                    gray_line = Line2D([0], [0], color='dimgray')
                    all_handles = plot_handles + \
                        [dark_line] + [gray_line] + [gray_patch] + [darkgray_patch]
                    all_labels = plot_labels + ['Leg vs leg force'] + [
                        'Leg vs antenna force'] + ['Foreleg grooming'] + ['Antennal grooming']
                elif plot_collisions and plot != 'collisions' and not collisions_across:
                    dark_line = Line2D([0], [0], color='black')
                    gray_line = Line2D([0], [0], color='dimgray')
                    all_handles = plot_handles + \
                        [dark_line] + [gray_line]
                    all_labels = plot_labels + ['Leg vs leg force'] + [
                        'Leg vs antenna force']
                else:
                    all_handles = plot_handles + \
                        [gray_patch] + [darkgray_patch]
                    all_labels = plot_labels + \
                        ['Foreleg grooming'] + ['Antennal grooming']
            else:
                all_handles = plot_handles
                all_labels = plot_labels

            if len(data2plot.keys()) == 1:
                axs.legend(
                    all_handles,
                    all_labels,
                    loc='upper right',
                    bbox_to_anchor=(
                        1.135,
                        1))
            else:
                axs[i].legend(
                    all_handles,
                    all_labels,
                    loc='upper right',
                    bbox_to_anchor=(
                        1.135,
                        1))

        if collisions_across:
            for ind in range(0, len(stance_plot), 2):
                time = np.arange(0, len(leg_force), 1) / steps
                if sim_data == 'walking':
                    c = 'gray'
                if sim_data == 'grooming':
                    if np.sum(leg_vs_leg[stance_plot[ind]:stance_plot[ind + 1]]) > 0:
                        c = 'dimgray'
                    elif np.sum(leg_vs_ant[stance_plot[ind]:stance_plot[ind + 1]]) > 0:
                        c = 'darkgray'
                    else:
                        c = 'darkgray'
                if len(data2plot.keys()) == 1:
                    axs.fill_between(time[stance_plot[ind]:stance_plot[ind + 1]], 0,
                                     1, facecolor=c, alpha=0.5, transform=axs.get_xaxis_transform())
                else:
                    axs[i].fill_between(time[stance_plot[ind]:stance_plot[ind + 1]], 0,
                                        1, facecolor=c, alpha=0.5, transform=axs[i].get_xaxis_transform())

    if len(data2plot.keys()) == 1:
        axs.set_xlabel('Time (s)')
    else:
        axs[len(axs) - 1].set_xlabel('Time (s)')
    plt.show()


def plot_collision_diagram(
        path_data,
        sim_data,
        gt=False,
        begin=0,
        end=0,
        time_step=0.001):
    """ Plots collision/gait diagrams.

    Parameters
    ----------
    path_data: <str>
        Path to simulation results.
    sim_data: <str>
        Behavior from data. Options: 'walking' or 'grooming'.
    begin: <float>
        Starting time for initiating the plots.
    end: <float>
        Stoping time for finishing the plots. If 0.0, all data is plotted.
    time_step: <float>
        Data time step.
    """
    data = {}
    length_data = 0

    if sim_data == 'walking':
        title_plot = 'Gait diagram'
        collisions = {
            'LF': [],
            'LM': [],
            'LH': [],
            'RF': [],
            'RM': [],
            'RH': []}
        if not gt:
            data = read_ground_contacts(path_data)
        else:
            file_path = os.path.join(path_data, "ground_truth_contact.pkl")
            data = np.load(file_path, allow_pickle=True)

        for leg in collisions.keys():
            sum_force = np.sum(np.array(data[leg]), axis=0)
            segment_force = np.delete(sum_force, 0)
            collisions[leg].append(segment_force)
            if length_data == 0:
                length_data = len(segment_force)

    elif sim_data == 'grooming':
        data = read_collision_forces(path_data)
        title_plot = 'Collisions diagram'
        collisions = {
            'LAntenna': [],
            'LFTibia': [],
            'LFTarsus1': [],
            'LFTarsus2': [],
            'LFTarsus3': [],
            'LFTarsus4': [],
            'LFTarsus5': [],
            'RFTarsus5': [],
            'RFTarsus4': [],
            'RFTarsus3': [],
            'RFTarsus2': [],
            'RFTarsus1': [],
            'RFTibia': [],
            'RAntenna': []}

        for segment1 in collisions.keys():
            seg_forces = []
            for segment2, force in data[segment1].items():
                seg_forces.append(force)
            sum_force = np.sum(np.array(seg_forces), axis=0)
            segment_force = np.delete(sum_force, 0)
            collisions[segment1].append(segment_force)
            if length_data == 0:
                length_data = len(segment_force)

    if end == 0:
        end = length_data * time_step

    steps = 1 / time_step
    start = int(begin * steps)
    stop = int(end * steps)

    fig, axs = plt.subplots(len(collisions.keys()),
                            sharex=True, gridspec_kw={'hspace': 0})
    fig.suptitle(title_plot)

    for i, (segment, force) in enumerate(collisions.items()):
        time = np.arange(0, len(force[0]), 1) / steps
        stance_plot = get_stance_periods(force[0], start, stop)
        for ind in range(0, len(stance_plot), 2):
            axs[i].fill_between(time[stance_plot[ind]:stance_plot[ind + 1]], 0, 1,
                                facecolor='black', alpha=1, transform=axs[i].get_xaxis_transform())

        axs[i].fill_between(time[start:stance_plot[0]],
                            0,
                            1,
                            facecolor='white',
                            alpha=1,
                            transform=axs[i].get_xaxis_transform())

        axs[i].fill_between(time[stance_plot[-1]:stop],
                            0,
                            1,
                            facecolor='white',
                            alpha=1,
                            transform=axs[i].get_xaxis_transform())

        axs[i].set_yticks((0.5,))
        axs[i].set_yticklabels((segment,))

    axs[len(axs) - 1].set_xlabel('Time (s)')
    if sim_data == 'walking':
        black_patch = mpatches.Patch(color='black', label='Stance')
    elif sim_data == 'grooming':
        black_patch = mpatches.Patch(color='black', label='Collision')
    axs[0].legend(
        handles=[black_patch],
        loc='upper right',
        bbox_to_anchor=(
            1.1,
            1))
    plt.show()


def plot_fly_path(
        path_data,
        generations=None,
        solutions=None,
        sequence=False,
        heading=True,
        ball_radius=5.0,
        begin=0,
        end=0,
        time_step=0.001,
        ax=None
):
    """ Plots collision/gait diagrams.

    Parameters
    ----------
    path_data: <str>
        Path to simulation results.
    generations: <list>
        Numbers of the generations to plot (for optimization experiments).
    solutions: <list>
        Names of the solutions to plot (for optimization experiments).
    sequence: <bool>
        Plotting path every time step.
    heading: <bool>
        Plotting heading of the fly (if sequence=True).
    ball_radius: <float>
        Radius of the spherical treadmill in millimeters.
    begin: <float>
        Starting time for initiating the plots.
    end: <float>
        Stoping time for finishing the plots. If 0.0, all data is plotted.
    time_step: <float>
        Data time step.
    """
    ball_data_list = []

    val_max = 0
    val_min = np.inf

    if generations:
        if not isinstance(generations, list):
            g = [generations]
        else:
            g = generations

        for gen in g:
            if solutions:
                if not isinstance(solutions, list):
                    s = [solutions]
                else:
                    s = solutions
            else:
                gen_folder = os.path.join(path_data, f'gen_{gen}')
                s = [d.split('_')[-1] for d in os.listdir(gen_folder)]
            for sol in s:
                sim_res_folder = os.path.join(path_data, f'gen_{gen}', f'sol_{sol}', 'physics', 'ball_rotations.h5')
                ball_data_list.append(sim_res_folder)
    else:
        sim_res_folder = os.path.join(
            path_data, 'physics', 'ball_rotations.h5')
        ball_data_list.append(sim_res_folder)

    if ax is None:
        fig = plt.figure()
        ax = plt.axes()

    m = MarkerStyle(marker=r'$\rightarrow$')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    colors = plt.cm.jet(np.linspace(0.3, 1, len(ball_data_list)))

    for ind, ball_data in enumerate(ball_data_list):

        data = pd.read_hdf(ball_data)

        if end == 0:
            end = len(data) * time_step

        steps = 1 / time_step
        start = int(begin * steps)
        stop = int(end * steps)

        data_array = np.array(data.values)

        x = []
        y = []

        for count, i in enumerate(range(start, stop - 1)):
            th = data_array[i][2]
            forward = (data_array[i][0] - data_array[0][0]) * ball_radius
            lateral = (data_array[i][1] - data_array[0][1]) * ball_radius
            x.append(forward)
            y.append(lateral)

            if sequence:
                ax.clear()
                curr_time = (i + 2) / steps
                print(f'\rTime: {curr_time:.3f}', end='')
                sc = ax.scatter(
                    x,
                    y,
                    c=np.linspace(
                        begin,
                        begin + len(x) / steps,
                        len(x)),
                    cmap='jet',
                    vmin=begin,
                    vmax=end)

                if heading:
                    m._transform.rotate_deg(th * 180 / np.pi)
                    ax.scatter(x[-1], y[-1], marker=m, s=200, color='black')
                    m._transform.rotate_deg(-th * 180 / np.pi)

                if count == 0:
                    sc.set_clim([begin, end])
                    cb = plt.colorbar(sc)
                    cb.set_label('Time (s)')

                ax.set_xlabel('x (mm)')
                ax.set_ylabel('y (mm)')
                plt.draw()
                plt.pause(0.001)

        max_x = np.max(np.array(x))
        min_x = np.min(np.array(x))

        if max_x > val_max:
            val_max = max_x

        if min_x < val_min:
            val_min = min_x

        lim = val_max + 0.05 * val_max
        low = val_min - 0.05 * val_min
        # ax.set_xlim(low, lim)

        if not sequence:
            if generations:
                gen_label = g[int(ind / len(s))] + 1
                sol_label = s[int(ind % len(s))]
                ax.plot(x,
                        y,
                        linewidth=2,
                        label=f'Gen {gen_label}-{sol_label}',
                        c=colors[ind])
            else:
                ax.plot(x, y, linewidth=2)


def get_data(
        data_path,
        begin,
        end,
        time_step,
        data_from=[],
        offset=0,
        window_time=0.2):
    fictrac_columns = ["Frame_counter",
                       "delta_rot_cam_x",
                       "delta_rot_cam_y",
                       "delta_rot_cam_z",
                       "delta_rot_error",
                       "delta_rot_lab_x",
                       "delta_rot_lab_y",
                       "delta_rot_lab_z",
                       "abs_rot_cam_x",
                       "abs_rot_cam_y",
                       "abs_rot_cam_z",
                       "abs_rot_lab_x",
                       "abs_rot_lab_y",
                       "abs_rot_lab_z",
                       "integrated_lab_x",
                       "integrated_lab_y",
                       "integrated_lab_heading",
                       "animal_movement_direction_lab",
                       "animal_movement_speed",
                       "integrated_side_movement",
                       "integrated_forward_movement",
                       "timestamp",
                       "seq_counter",
                       "delta_time",
                       "alt_time"]

    if ".dat" in data_path:
        data = pd.read_csv(data_path, header=None, names=fictrac_columns)

    if ".h5" in data_path:
        data = pd.read_hdf(data_path)

    if end == 0:
        end = len(data) * time_step

    steps = 1 / time_step
    start = int((begin + offset) * steps)
    stop = int((end + offset) * steps)

    if not data_from:
        data_from = list(data.columns)

    norm_data = {}
    for key in data_from:
        if window_time > 0:
            filtered_data = scipy.ndimage.median_filter(
                data[key], size=int(window_time / time_step))
        else:
            filtered_data = np.array(data[key].values)
        baseline = np.mean(filtered_data[start:start + int(0.1 / time_step)])
        norm_data[key] = filtered_data[start:stop] - baseline
        if "lab_heading" in key:
            diff_heading = np.abs(np.diff(norm_data[key]))
            cross_points = np.where(diff_heading > np.pi)[0]
            if len(cross_points) % 2 != 0:
                cross_points = np.append(cross_points, stop)
            heading_fictrac = norm_data[key].copy()
            for p in range(1, len(cross_points), 2):
                init = cross_points[p - 1] + 1
                fin = cross_points[p] + 1
                heading_fictrac[init:fin] = heading_fictrac[init:fin] - 2 * np.pi
            norm_data[key] = heading_fictrac

    return norm_data


def plot_fly_path_comparison(
        fictrac_path,
        sim_path,
        plot_vel=True,
        plot_traj=False,
        ball_radius=5,
        begin=0,
        end=0,
        offset_fictrac=0,
        offset_sim=0,
        time_step_fictrac=0.01,
        time_step_sim=0.001,
        filter_window_time=0.2
):
    """ Comparing fly path/ball rotations between ground truth (obtained from FicTrac) and simulation

    Parameters
    ----------
    path_data: <str>
        Path to simulation results.
    ball_radius: <float>
        Radius of the spherical treadmill in millimeters.
    begin: <float>
        Starting time for initiating the plots.
    end: <float>
        Stoping time for finishing the plots. If 0.0, all data is plotted.
    time_step: <float>
        Data time step.
    """
    data_from_fictrac = ["integrated_forward_movement",
                         "integrated_side_movement",
                         "integrated_lab_heading",
                         "delta_rot_lab_x",
                         "delta_rot_lab_y",
                         "delta_rot_lab_z"]

    fictrac_data = get_data(
        fictrac_path,
        begin,
        end,
        time_step_fictrac,
        data_from_fictrac,
        offset_fictrac,
        filter_window_time)
    fw_fictrac = fictrac_data["integrated_forward_movement"] * ball_radius
    side_fictrac = fictrac_data["integrated_side_movement"] * ball_radius
    heading_fictrac = fictrac_data["integrated_lab_heading"]

    vel_fw_fictrac = -fictrac_data["delta_rot_lab_x"] / time_step_fictrac
    vel_side_fictrac = -fictrac_data["delta_rot_lab_y"] / time_step_fictrac
    vel_heading_fictrac = -fictrac_data["delta_rot_lab_z"] / time_step_fictrac

    data_from_sim = ["x", "y", "z"]
    sim_data_path = os.path.join(sim_path, 'physics', 'ball_rotations.h5')
    ball_data = get_data(
        sim_data_path,
        begin,
        end,
        time_step_sim,
        data_from_sim,
        offset_sim,
        filter_window_time)
    fw_sim = ball_data["x"] * ball_radius
    side_sim = ball_data["y"] * ball_radius
    heading_sim = ball_data["z"]

    sim_vel_data_path = os.path.join(sim_path, 'physics', 'ball_velocities.h5')
    vel_data = get_data(
        sim_vel_data_path,
        begin,
        end,
        time_step_sim,
        data_from_sim,
        offset_sim,
        filter_window_time)
    vel_fw_sim = -vel_data["y"]
    vel_side_sim = vel_data["x"]
    vel_heading_sim = -vel_data["z"]

    window = 11
    order = 3

    '''
    vel_fw_fictrac = scipy.signal.savgol_filter(fw_fictrac, window, order, deriv=1, delta=time_step_fictrac, mode='nearest')
    vel_side_fictrac = scipy.signal.savgol_filter(side_fictrac, window, order, deriv=1, delta=time_step_fictrac, mode='nearest')
    vel_heading_fictrac = scipy.signal.savgol_filter(heading_fictrac, window, order, deriv=1, delta=time_step_fictrac, mode='nearest')

    vel_fw_sim = scipy.signal.savgol_filter(fw_sim, 111, order, deriv=1, delta=time_step_sim, mode='nearest')
    vel_side_sim = scipy.signal.savgol_filter(side_sim, 111, order, deriv=1, delta=time_step_sim, mode='nearest')
    vel_heading_sim = scipy.signal.savgol_filter(heading_sim, 111, order, deriv=1, delta=time_step_sim, mode='nearest')
    '''

    time_fictrac = np.arange(begin, end, time_step_fictrac)
    time_sim = np.arange(begin, end, time_step_sim)
    filter_window_time = 0.2

    if plot_traj:
        x_head_fictrac, y_head_fictrac = get_flat_trajectory(
            fw_fictrac, side_fictrac, heading_fictrac)
        x_head_sim, y_head_sim = get_flat_trajectory(
            -fw_sim, -side_sim, -heading_sim)

        plt.figure()
        plt.plot(x_head_fictrac, y_head_fictrac, label="Fictrac path")
        plt.plot(x_head_sim, y_head_sim, label="NeuroMechFly path")
        plt.xlabel('Distance (mm)', fontsize=14)
        plt.ylabel('Distance (mm)', fontsize=14)
        plt.legend(fontsize=11)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)

        # plt.figure(2)
        #plt.plot(fw_fictrac, side_fictrac, label='Fictrac (x-y)')
        #plt.plot(fw_sim, side_sim, label='NeuroMechFly (x-y)')
        #plt.xlabel('Distance (mm)', fontsize=14)
        #plt.ylabel('Distance (mm)', fontsize=14)
        # plt.legend()

        plt.figure()
        plt.plot(time_fictrac, side_fictrac, label='Fictrac')
        plt.plot(time_sim, side_sim, label='NeuroMechFly')
        plt.xlabel('Time (s)', fontsize=14)
        plt.ylabel('Side distance (mm)', fontsize=14)
        plt.legend(fontsize=11)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)

        plt.figure()
        plt.plot(time_fictrac, fw_fictrac, label='Fictrac')
        plt.plot(time_sim, fw_sim, label='NeuroMechFly')
        plt.xlabel('Time (s)', fontsize=14)
        plt.ylabel('Forward distance (mm)', fontsize=14)
        plt.legend(fontsize=11)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)

        plt.figure()
        plt.plot(time_fictrac, heading_fictrac, label='Fictrac')
        plt.plot(time_sim, heading_sim, label='NeuroMechFly')
        plt.xlabel('Time (s)', fontsize=14)
        plt.ylabel('Heading rotation (rad)', fontsize=14)
        plt.legend(fontsize=11)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)

    if plot_vel:
        plt.figure()
        interp_fictrac, corr_coef = calculate_correlation_between(
            vel_fw_fictrac, vel_fw_sim, time_fictrac, time_sim)
        plt.plot(time_sim, interp_fictrac, label='Fictrac')
        #plt.plot(time_fictrac, smooth_vel_fw_fictrac, label='vel fw smooth fictrac')
        plt.plot(time_sim, vel_fw_sim, label='NeuroMechFly')
        plt.xlabel('Time (s)', fontsize=14)
        plt.ylabel('Forward Velocity (rad/s)', fontsize=14)
        plt.legend(fontsize=11)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        print(f"Correlation coeficient (Fw vel): {corr_coef}")

        plt.figure()
        interp_fictrac, corr_coef = calculate_correlation_between(
            vel_side_fictrac, vel_side_sim, time_fictrac, time_sim)
        plt.plot(time_sim, interp_fictrac, label='Fictrac')
        #plt.plot(time_fictrac, scipy.ndimage.median_filter(vel_side_fictrac,size=20), label='vel fw smooth fictrac')
        plt.plot(time_sim, vel_side_sim, label='NeuroMechFly')
        plt.xlabel('Time (s)', fontsize=14)
        plt.ylabel('Side Velocity (rad/s)', fontsize=14)
        plt.legend(fontsize=11)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        print(f"Correlation coeficient (Side vel): {corr_coef}")

        plt.figure()
        interp_fictrac, corr_coef = calculate_correlation_between(
            vel_heading_fictrac, vel_heading_sim, time_fictrac, time_sim)
        plt.plot(time_sim, interp_fictrac, label='Fictrac')
        #plt.plot(time_fictrac, scipy.ndimage.median_filter(vel_heading_fictrac,size=20), label='vel fw smooth fictrac')
        plt.plot(time_sim, vel_heading_sim, label='NeuroMechFly')
        plt.xlabel('Time (s)', fontsize=14)
        plt.ylabel('Heading Velocity (rad/s)', fontsize=14)
        plt.legend(fontsize=11)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        print(f"Correlation coeficient (Heading vel): {corr_coef}")

    plt.show()


def calculate_correlation_between(fictrac, sim, time_fictrac, time_sim):
    interpolated_fictrac = pchip_interpolate(time_fictrac, fictrac, time_sim)

    corr_coef, p_value = scipy.stats.spearmanr(interpolated_fictrac, sim)

    return interpolated_fictrac, corr_coef


def plot_fly_path_comparison_old(
        fictrac_path,
        ball_path,
        ball_radius=5,
        begin=0,
        end=0,
        offset_fictrac=0,
        time_step_fictrac=0.01,
        time_step_sim=0.001
):
    """ Comparing fly path/ball rotations between ground truth (obtained from FicTrac) and simulation

    Parameters
    ----------
    path_data: <str>
        Path to simulation results.
    ball_radius: <float>
        Radius of the spherical treadmill in millimeters.
    begin: <float>
        Starting time for initiating the plots.
    end: <float>
        Stoping time for finishing the plots. If 0.0, all data is plotted.
    time_step: <float>
        Data time step.
    """

    fictrac_columns = ["Frame_counter",
                       "delta_rot_cam_x",
                       "delta_rot_cam_y",
                       "delta_rot_cam_z",
                       "delta_rot_error",
                       "delta_rot_lab_x",
                       "delta_rot_lab_y",
                       "delta_rot_lab_z",
                       "abs_rot_cam_x",
                       "abs_rot_cam_y",
                       "abs_rot_cam_z",
                       "abs_rot_lab_x",
                       "abs_rot_lab_y",
                       "abs_rot_lab_z",
                       "integrated_lab_x",
                       "integrated_lab_y",
                       "integrated_lab_heading",
                       "animal_movement_direction_lab",
                       "animal_movement_speed",
                       "integrated_side_movement",
                       "integrated_forward_movement",
                       "timestamp",
                       "seq_counter",
                       "delta_time",
                       "alt_time"]

    fictrac_data_path = fictrac_path
    fictrac_data = pd.read_csv(
        fictrac_data_path,
        header=None,
        names=fictrac_columns)

    ball_data_path = os.path.join(ball_path, 'physics', 'ball_rotations.h5')
    ball_data = np.array(pd.read_hdf(ball_data_path).values).transpose()

    vel_data_path = os.path.join(ball_path, 'physics', 'ball_velocities.h5')
    vel_data = np.array(pd.read_hdf(vel_data_path).values).transpose()

    if end == 0:
        end = len(fictrac_data) * time_step_fictrac

    steps_fictrac = 1 / time_step_fictrac
    start_fictrac = int((begin + offset_fictrac) * steps_fictrac)
    stop_fictrac = int((end + offset_fictrac) * steps_fictrac)

    # y_fictrac = -(fictrac_data["integrated_lab_x"][start_fictrac:stop_fictrac]-fictrac_data["integrated_lab_x"][start_fictrac]) #* ball_radius
    # x_fictrac =
    # (fictrac_data["integrated_lab_y"][start_fictrac:stop_fictrac] -
    # fictrac_data["integrated_lab_y"][start_fictrac]) #* ball_radius

    side_fictrac = (fictrac_data["integrated_side_movement"][start_fictrac:stop_fictrac] -
                    fictrac_data["integrated_side_movement"][start_fictrac + 5]) * ball_radius
    fw_fictrac = (fictrac_data["integrated_forward_movement"][start_fictrac:stop_fictrac] -
                  fictrac_data["integrated_forward_movement"][start_fictrac + 5]) * ball_radius
    heading_raw = (fictrac_data["integrated_lab_heading"][start_fictrac:stop_fictrac] -
                   fictrac_data["integrated_lab_heading"][start_fictrac + 5])

    delta_fw_fictrac = -(fictrac_data["delta_rot_lab_x"][start_fictrac:stop_fictrac] -
                         fictrac_data["delta_rot_lab_x"][start_fictrac + 5]) / time_step_fictrac
    delta_side_fictrac = -(fictrac_data["delta_rot_lab_y"][start_fictrac:stop_fictrac] -
                           fictrac_data["delta_rot_lab_y"][start_fictrac + 5]) / time_step_fictrac
    delta_heading_fictrac = -(fictrac_data["delta_rot_lab_z"][start_fictrac:stop_fictrac] -
                              fictrac_data["delta_rot_lab_z"][start_fictrac + 5]) / time_step_fictrac

    diff_heading = np.abs(np.diff(heading_raw))
    cross_points = np.where(diff_heading > np.pi)[0]
    if len(cross_points) % 2 != 0:
        cross_points = np.append(cross_points, stop_fictrac)
    heading_fictrac = heading_raw.copy()
    for p in range(1, len(cross_points), 2):
        init = cross_points[p - 1] + 1
        fin = cross_points[p] + 1
        heading_fictrac[init:fin] = heading_fictrac[init:fin] - 2 * np.pi

    steps_sim = 1 / time_step_sim
    start_sim = int(begin * steps_sim)
    stop_sim = int(end * steps_sim)

    fw_sim = (ball_data[0][start_sim:stop_sim] -
              ball_data[0][start_sim + 5]) * ball_radius
    side_sim = (ball_data[1][start_sim:stop_sim] -
                ball_data[1][start_sim + 5]) * ball_radius
    heading_sim = (ball_data[2][start_sim:stop_sim] -
                   ball_data[2][start_sim + 5])

    delta_fw_sim = np.diff(fw_sim)
    delta_side_sim = np.diff(side_sim)
    delta_heading_sim = np.diff(heading_sim)

    delta_fw_sim = np.insert(delta_fw_sim, 0, 0) / time_step_sim
    delta_side_sim = np.insert(delta_side_sim, 0, 0) / time_step_sim
    delta_heading_sim = np.insert(delta_heading_sim, 0, 0) / time_step_sim

    #vel_x = (vel_data[0][start_sim:stop_sim]-vel_data[0][start_sim+5])
    #vel_y = (vel_data[1][start_sim:stop_sim]-vel_data[1][start_sim+5])
    #vel_z = (vel_data[2][start_sim:stop_sim]-vel_data[2][start_sim+5])

    window = 11
    order = 3

    vel_fw_fictrac = scipy.signal.savgol_filter(
        fw_fictrac,
        window,
        order,
        deriv=1,
        delta=time_step_fictrac,
        mode='nearest')
    vel_side_fictrac = scipy.signal.savgol_filter(
        side_fictrac,
        window,
        order,
        deriv=1,
        delta=time_step_fictrac,
        mode='nearest')
    vel_heading_fictrac = scipy.signal.savgol_filter(
        heading_fictrac,
        window,
        order,
        deriv=1,
        delta=time_step_fictrac,
        mode='nearest')

    vel_fw_sim = scipy.signal.savgol_filter(
        fw_sim, 111, order, deriv=1, delta=time_step_sim, mode='nearest')
    vel_side_sim = scipy.signal.savgol_filter(
        side_sim, 111, order, deriv=1, delta=time_step_sim, mode='nearest')
    vel_heading_sim = scipy.signal.savgol_filter(
        heading_sim, 111, order, deriv=1, delta=time_step_sim, mode='nearest')

    x_head_fictrac, y_head_fictrac = get_flat_trajectory(
        fw_fictrac.values, side_fictrac.values, heading_fictrac.values)
    x_head_sim, y_head_sim = get_flat_trajectory(
        -fw_sim, -side_sim, -heading_sim)

    plt.figure(0)
    plt.plot(x_head_fictrac, y_head_fictrac, label="fictrac path")
    plt.plot(x_head_sim, y_head_sim, label="NeuroMechFly path")
    plt.xlabel('Distance (mm)')
    plt.ylabel('Distance (mm)')
    plt.legend()

    '''
    plt.figure(1)
    ax = plt.axes()
    m = MarkerStyle(marker=r'$\rightarrow$')
    gap = 0.5
    max_x = np.max(x_fictrac) + gap
    min_x = np.min(x_fictrac) - gap
    max_y = np.max(y_fictrac) + gap
    min_y = np.min(y_fictrac) - gap
    #print(heading_fictrac)
    #plt.plot(x_fictrac, y_fictrac)
    for i in range(len(x_fictrac)):
        ax.clear()
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        curr_time = (i+2)/steps_fictrac
        print(f'\rTime: {curr_time:.3f}', end='')
        sc = ax.scatter(
            x_fictrac[0:i],
            y_fictrac[0:i],
            c=np.linspace(
                begin,
                begin+len(x_fictrac[0:i])/steps_fictrac,
                len(x_fictrac[0:i])),
            cmap='winter',
            vmin=begin,
            vmax=end)
        m._transform.rotate_deg(heading_fictrac.values[i] * 180 / np.pi)
        ax.scatter(x_fictrac.values[i], y_fictrac.values[i], marker=m, s=200, color='black')
        m._transform.rotate_deg(-heading_fictrac.values[i] * 180 / np.pi)
        if i == 0:
            sc.set_clim([begin, end])
            cb = plt.colorbar(sc)
            cb.set_label('Time (s)')

        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        plt.draw()
        plt.pause(0.001)
    '''

    time_fictrac = np.arange(begin, end, time_step_fictrac)
    time_sim = np.arange(begin, end, time_step_sim)

    '''
    plt.figure(10)
    plt.plot(time_fictrac, fw_fictrac, label='Fictrac forward')
    plt.plot(time_fictrac, side_fictrac, label='Fictrac side')
    plt.plot(time_fictrac, heading_fictrac, label='Fictrac heading')
    plt.xlabel('Time (s)')
    plt.ylabel('Distance (mm)')
    plt.legend()

    plt.figure(11)
    plt.plot(time_fictrac, delta_fw_fictrac, label='Fictrac forward')
    plt.plot(time_fictrac, delta_side_fictrac, label='Fictrac side')
    plt.plot(time_fictrac, delta_heading_fictrac, label='Fictrac heading')
    plt.xlabel('Time (s)')
    plt.ylabel('Distance (mm)')
    plt.legend()

    plt.show()
    quit()
    '''

    plt.figure(2)
    plt.plot(fw_fictrac, side_fictrac, label='Fictrac (x-y)')
    plt.plot(fw_sim, side_sim, label='NeuroMechFly (x-y)')
    # plt.ylim(-0.5,1)
    plt.xlabel('Distance (mm)')
    plt.ylabel('Distance (mm)')
    plt.legend()

    plt.figure(3)
    plt.plot(time_fictrac, side_fictrac, label='Fictrac side')
    plt.plot(time_sim, side_sim, label='NeuroMechFly side')
    plt.xlabel('Time (s)')
    plt.ylabel('Distance (mm)')
    plt.legend()

    plt.figure(4)
    plt.plot(time_fictrac, fw_fictrac, label='Fictrac forward')
    plt.plot(time_sim, fw_sim, label='NeuroMechFly forward')
    plt.xlabel('Time (s)')
    plt.ylabel('Distance (mm)')
    plt.legend()

    plt.figure(5)
    plt.plot(time_fictrac, heading_fictrac, label='Fictrac heading')
    plt.plot(time_sim, heading_sim, label='NeuroMechFly heading')
    plt.xlabel('Time (s)')
    plt.ylabel('Distance (rad)')
    plt.legend()

    plt.figure(6)
    #plt.plot(time_fictrac, delta_fw_fictrac, label='delta fw fictrac')
    plt.plot(time_fictrac, vel_fw_fictrac, label='vel fw fictrac')
    plt.plot(time_sim, vel_fw_sim, label='vel fw sim')
    #plt.plot(time_sim, delta_fw_sim, label='delta fw sim')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (mm/s)')
    plt.legend()

    plt.figure(7)
    #plt.plot(time_fictrac, delta_side_fictrac, label='delta side fictrac')
    plt.plot(time_fictrac, vel_side_fictrac, label='vel side fictrac')
    plt.plot(time_sim, vel_side_sim, label='vel side sim')
    #plt.plot(time_sim, delta_side_sim, label='delta side sim')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (mm/s)')
    plt.legend()

    plt.figure(8)
    #plt.plot(time_fictrac, delta_heading_fictrac, label='delta heading fictrac')
    plt.plot(time_fictrac, vel_heading_fictrac, label='vel heading fictrac')
    plt.plot(time_sim, vel_heading_sim, label='vel heading sim')
    #plt.plot(time_sim, delta_heading_sim, label='delta heading sim')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (rad/s)')
    plt.legend()

    plt.show()


def get_flat_trajectory(fw, side, heading):

    x_trajectory = [0]
    y_trajectory = [0]
    diff_x = np.diff(fw)
    diff_y = np.diff(side)
    for ind in range(len(diff_x)):
        new_x = diff_x[ind] * np.cos(heading[ind + 1]) + \
            diff_y[ind] * np.sin(heading[ind + 1]) + x_trajectory[-1]
        new_y = diff_x[ind] * np.sin(heading[ind + 1]) - \
            diff_y[ind] * np.cos(heading[ind + 1]) + y_trajectory[-1]
        x_trajectory.append(new_x)
        y_trajectory.append(new_y)

    return x_trajectory, y_trajectory


def compare_collision_diagram(
        path_data,
        gt_data,
        sim_data,
        begin=0,
        end=0,
        time_step_sim=0.001,
        time_step_gt=0.01):
    """ Plots collision/gait diagrams.

    Parameters
    ----------
    path_data: <str>
        Path to simulation results.
    sim_data: <str>
        Behavior from data. Options: 'walking' or 'grooming'.
    begin: <float>
        Starting time for initiating the plots.
    end: <float>
        Stoping time for finishing the plots. If 0.0, all data is plotted.
    time_step: <float>
        Data time step.
    """
    data = {}
    length_data = 0

    if sim_data == 'walking':
        title_plot = 'Gait diagram'
        collisions = {
            'LF': [],
            'LM': [],
            'LH': [],
            'RF': [],
            'RM': [],
            'RH': []}

        collisions_gt = {
            'LF': [],
            'LM': [],
            'LH': [],
            'RF': [],
            'RM': [],
            'RH': []}

        data_sim = read_ground_contacts(path_data)

        gt_file_path = os.path.join(gt_data, "ground_truth_contact.pkl")
        data_gt = np.load(gt_file_path, allow_pickle=True)

        for leg in collisions.keys():
            sum_force = np.sum(np.array(data_sim[leg]), axis=0)
            segment_force = np.delete(sum_force, 0)
            collisions[leg].append(segment_force)
            if length_data == 0:
                length_data = len(segment_force)

        for leg in collisions_gt.keys():
            sum_force = np.sum(np.array(data_gt[leg]), axis=0)
            segment_force = np.delete(sum_force, 0)
            collisions_gt[leg].append(segment_force)
            if length_data == 0:
                length_data = len(segment_force)

    elif sim_data == 'grooming':
        data = read_collision_forces(path_data)
        title_plot = 'Collisions diagram'
        collisions = {
            'LAntenna': [],
            'LFTibia': [],
            'LFTarsus1': [],
            'LFTarsus2': [],
            'LFTarsus3': [],
            'LFTarsus4': [],
            'LFTarsus5': [],
            'RFTarsus5': [],
            'RFTarsus4': [],
            'RFTarsus3': [],
            'RFTarsus2': [],
            'RFTarsus1': [],
            'RFTibia': [],
            'RAntenna': []}

        for segment1 in collisions.keys():
            seg_forces = []
            for segment2, force in data[segment1].items():
                seg_forces.append(force)
            sum_force = np.sum(np.array(seg_forces), axis=0)
            segment_force = np.delete(sum_force, 0)
            collisions[segment1].append(segment_force)
            if length_data == 0:
                length_data = len(segment_force)

    if end == 0:
        end = length_data * time_step_sim

    steps_sim = 1 / time_step_sim
    start_sim = int(begin * steps_sim)
    stop_sim = int(end * steps_sim)

    steps_gt = 1 / time_step_gt
    start_gt = int(begin * steps_gt)
    stop_gt = int(end * steps_gt)

    fig, axs = plt.subplots(len(collisions.keys()),
                            sharex=True, gridspec_kw={'hspace': 0})
    fig.suptitle(title_plot)
    stance_frames = {}
    stance_frames_gt = {}
    for i, (segment, force) in enumerate(collisions.items()):
        time = np.arange(0, len(force[0]), 1) / steps_sim
        stance_plot = get_stance_periods(force[0], start_sim, stop_sim)
        stance_frames[segment] = []
        for ind in range(0, len(stance_plot), 2):
            start_stance = stance_plot[ind]
            stop_stance = stance_plot[ind + 1]
            num_steps = int(stop_stance - start_stance)
            axs[i].fill_between(time[start_stance:stop_stance], 0, 1,
                                facecolor='deepskyblue', alpha=0.5,
                                transform=axs[i].get_xaxis_transform())
            stance_frames[segment].extend(np.linspace(start_stance,
                                                      stop_stance,
                                                      num_steps,
                                                      endpoint=False))

        axs[i].fill_between(time[start_sim:stance_plot[0]],
                            0,
                            1,
                            facecolor='white',
                            alpha=0.5,
                            transform=axs[i].get_xaxis_transform())

        axs[i].fill_between(time[stance_plot[-1]:stop_sim],
                            0,
                            1,
                            facecolor='white',
                            alpha=0.5,
                            transform=axs[i].get_xaxis_transform())

        axs[i].set_yticks((0.5,))
        axs[i].set_yticklabels((segment,))

    for i, (segment, force) in enumerate(collisions_gt.items()):
        scale_factor = time_step_gt / time_step_sim
        stop_time = np.round(len(force[0]) * scale_factor)
        time = np.arange(0, stop_time, 1) / steps_sim
        time_gt = np.arange(0, len(force[0]), 1) / steps_gt
        stance_plot = get_stance_periods(force[0], start_gt, stop_gt)
        stance_frames_gt[segment] = []
        for ind in range(0, len(stance_plot), 2):

            start_stance = int(np.floor(stance_plot[ind] * scale_factor))
            stop_stance = int(np.ceil(stance_plot[ind + 1] * scale_factor))
            num_steps = int(stop_stance - start_stance)

            axs[i].fill_between(time[start_stance:stop_stance], 0, 1,
                                facecolor='y', alpha=0.5,
                                transform=axs[i].get_xaxis_transform())
            stance_frames_gt[segment].extend(np.linspace(start_stance,
                                                         stop_stance,
                                                         num_steps,
                                                         endpoint=False))

        axs[i].fill_between(time_gt[start_gt:stance_plot[0]],
                            0,
                            1,
                            facecolor='white',
                            alpha=0.5,
                            transform=axs[i].get_xaxis_transform())

        axs[i].fill_between(time_gt[stance_plot[-1]:stop_gt],
                            0,
                            1,
                            facecolor='white',
                            alpha=0.5,
                            transform=axs[i].get_xaxis_transform())

        axs[i].set_yticks((0.5,))
        axs[i].set_yticklabels((segment,))

    results = pd.DataFrame()
    tot_frames = stop_sim - start_sim
    for leg, frames in stance_frames.items():
        tp = np.count_nonzero(
            np.isin(
                np.array(frames),
                np.array(
                    stance_frames_gt[leg])))
        fp = len(frames) - tp
        tp_count_gt = np.count_nonzero(
            np.isin(
                np.array(
                    stance_frames_gt[leg]),
                np.array(frames)))
        fn = len(stance_frames_gt[leg]) - tp_count_gt
        tn = tot_frames - tp - fp - fn

        df_vals = pd.DataFrame([[tp / tot_frames,
                                 tn / tot_frames,
                                 fp / tot_frames,
                                 fn / tot_frames,
                                 (tp + tn) / tot_frames]],
                               columns=['True positive',
                                        'True negative',
                                        'False positive',
                                        'False negative',
                                        'Accuracy'])
        df_vals['Leg'] = leg
        results = results.append(df_vals, ignore_index=True)
        #print(leg, [[key, v/tot_frames] for key, v in results[leg].items()])

    axs[len(axs) - 1].set_xlabel('Time (s)')
    if sim_data == 'walking':
        gt_patch = mpatches.Patch(color='y', alpha=0.5, label='GT-Stance')
        sim_patch = mpatches.Patch(
            color='deepskyblue',
            alpha=0.5,
            label='NMF-Stance')
        patches = [gt_patch, sim_patch]
    elif sim_data == 'grooming':
        black_patch = mpatches.Patch(color='black', label='Collision')
        patches = [black_patch]
    axs[0].legend(
        handles=patches,
        loc='upper right',
        bbox_to_anchor=(
            1.1,
            1))

    print(results)
    print(np.mean(results['Accuracy']))
    fig, ax2 = plt.subplots()
    ax2.bar(results['Leg'], results['True positive'], label='True positive')
    ax2.bar(
        results['Leg'],
        results['True negative'],
        bottom=results['True positive'],
        label='True negative')
    ax2.bar(
        results['Leg'],
        results['False negative'],
        bottom=results['True positive'] +
        results['True negative'],
        label='False negative')
    ax2.bar(
        results['Leg'],
        results['False positive'],
        bottom=results['True positive'] +
        results['True negative'] +
        results['False negative'],
        label='False positive')
    ax2.set_xlabel('Leg')
    ax2.set_ylabel('Percentage')
    ax2.legend()

    plt.show()
