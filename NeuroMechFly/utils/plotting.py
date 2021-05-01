""" Script to plot the simulation results. """
import os
import math
import pickle
import pkgutil
import itertools
import cv2 as cv
import numpy as np
import pandas as pd
import seaborn as sns
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.legend_handler import HandlerTuple
from pathlib import Path
from scipy import stats
from scipy import ndimage
from scipy.spatial.transform import Rotation as R
from sklearn import svm
from sklearn.metrics import mean_squared_error
from statsmodels.stats import weightstats as stests
from .sensitivity_analysis import calculate_forces

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

    Args:
        mu (np.array): mean, shape [N_samples, N_lines] or [N_samples]
        error (np.array): error to be plotted, e.g. standard error of the mean, shape [N_samples, N_lines] or [N_samples]
        conf (int): confidence interval, if none, stderror is plotted instead of std
        plot_label (str, optional): the label for each line either a string if only one line or list of strings if multiple lines
        x (np.array, optional): shape [N_samples]. If not specified will be np.arange(mu.shape[0])
        alpha (float, optional): transparency of the shaded area. default 0.3
        color ([type], optional): pre-specify colour. if None, use Python default colour cycle
        ax ([type], optional): axis to be plotted on, otherwise the current is axis with plt.gca()
    """
    if ax is None:
        ax = plt.gca()
    if x is None:
        x = np.arange(0, mu.shape[0], 1) * time_step
    p = ax.plot(x[beg:end], mu[beg:end], lw=1, color=color, label=plot_label)
    if len(mu.shape) is 1:
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
    constant='Kv0.9',
    condition='Kp0.4_Kv0.9',
    beg=2000,
    intv=250,
    time_step=0.001,
    ground_truth=None
):
    """Plot the joint info of one specific leg versus independent variable.

    Args:
        *args (np.array): force to be plotted, i.e. grf, lateral friction, thorax
        multiple (bool, optional): plots vectors instead of norm.
        data (dictionary, optional): dictionary to be plotted, i.e. joint torques
        full_name (str, optional): key name, 'joint_LMTibia'.
        gain_range (np.array, optional): range of gains to be plotted, i.e. np.arange(0.1,1.4,0.2).
        scaling_factor (int, optional): scale to change the units.
        ax ([type], optional): axis to be plotted on, otherwise the current is axis with plt.gca()
        beg (int, optional): beginning of the data to be plotted. the entire data is long
        intv (int, optional): int of the data to be plotted
        ground_truth (np.array, optional): ground truth for position or velocity
    """
    if ax is None:
        ax = plt.gca()
    if ground_truth is not None:
        ax.plot(np.array(ground_truth[beg:beg + intv]) * scaling_factor,
                linewidth=2.5, color="red", label="Ground Truth")

    for k in gain_range:
        k_value = "_".join((constant, 'Kv' +
                            str(round(k, 1)))) if 'Kp' in constant else "_".join(('Kp' +
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
    
    Args:
        title (str): Title of the heatmap
        joint_data (dict): Dictionary containing the joint information (angle etc)
        colorbar_title (str): Title of the colorbar
        precision (str): Precision of the heatmap entries
        linewidth (str): Width of the lines in heatmap
        ax (obj): axis to be plotted on, otherwise plt.gca()
        cmap (str): color map of the heatmap
    """
    if ax is None:
        ax = plt.gca()

    ax = sns.heatmap(
        joint_data,
        annot=True,
        ax=ax,
        linewidth=linewidth,
        cmap=cmap,
        fmt=precision,
        cbar_kws={
            'label': colorbar_title})
    ax.set_title(title)
    ax.invert_yaxis()


def read_ground_contacts(
        path_data,
        leg_key):
    """Read ground reaction forces data, calculates magnitude for each segment

    Parameters:
        path_data (str): Path to data for plotting
        leg_key (str): Key for specifying leg to plot, e.g, LF, LM, LH, RF, RM, RH

    Return:
        grf (np.array): ground reaction forces for all segments in a specific leg
    """
    grf_data = os.path.join(path_data, 'physics', 'ground_contacts.h5')
    data = pd.read_hdf(grf_data)
    grf_x = []
    grf_y = []
    grf_z = []
    for leg, force in data.items():
        if leg[:2] == leg_key:
            if 'x' in leg:
                grf_x.append(force)
            if 'y' in leg:
                grf_y.append(force)
            if 'z' in leg:
                grf_z.append(force)
    grf = np.linalg.norm([grf_x, grf_y, grf_z], axis=0)

    return grf


def read_collision_forces(path_data):
    """Read ground reaction forces data, calculates magnitude for each segment

    Parameters:
        path_data (str): Path to data for plotting

    Return:
        collisions (dict): dictionary with all collision forces for each segment
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


def plot_angles_torques_grf(
        path_data,
        leg_key,
        sim_data='walking',
        angles={},
        plot_angles=True,
        plot_torques=True,
        plot_grf=True,
        plot_collisions=True,
        collisions_across=True,
        begin=0.0,
        end=0.0,
        time_step=0.001,
        torqueScalingFactor=1e9,
        grfScalingFactor=1e6):
    """Plot angles, torques and ground reaction forces for a single leg

    Parameters:
        path_data (str): Path to data for plotting
        leg_key (str): Key for specifying leg to plot, e.g, LF, LM, LH, RF, RM, RH
        sim_data (str, default walking): behavior from data, e.g., walking or grooming
        angles (dict, optional): angles to plot
        plot_angles (bool, default True): Select if plotting angles
        plot_torques (bool, default True): Select if plotting torques
        plot_grf (bool, default True): Select if plotting ground reaction forces
        plot_collisions (bool, default True): Select if plotting self-collision forces
        plot_collisions_across (bool, default True): Select if plotting collisions across angles and torques plots
        begin (float, default 0.0): Time point for starting the plot
        end (float, default 0.0): Time point for finishing the plot, if 0.0, all time points are plotted
        time_step (float, default 0.001): Data time step
        torqueScalingFactor (float, default 1.0): Scaling factor for torques
        grfScalingFactor (float, default 1.0): Scaling factor for ground reaction forces
    """

    data2plot = {}

    equivalence = {'ThC yaw': ['yaw', 'Coxa_yaw'],
                   'ThC pitch': ['pitch', 'Coxa'],
                   'ThC roll': ['roll', 'Coxa_roll'],
                   'CTr pitch': ['th_fe', 'Femur'],
                   'CTr roll': ['roll_tr', 'Femur_roll'],
                   'FTi pitch': ['th_ti', 'Tibia'],
                   'TiTa pitch': ['th_ta', 'Tarsus1']}

    length_data = 0

    if plot_angles:
        angles_raw = angles[leg_key + '_leg']
        data2plot['angles'] = {}
        for label, match_labels in equivalence.items():
            for key in angles_raw.keys():
                if key in match_labels:
                    if length_data == 0:
                        length_data = len(angles_raw[key])
                    data2plot['angles'][label] = angles_raw[key]

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
                if key in match_labels:
                    if length_data == 0:
                        length_data = len(torques_raw[key])
                    data2plot['torques'][label] = torques_raw[key]

    if plot_grf:
        if sim_data == 'walking':
            data2plot['grf'] = read_ground_contacts(path_data, leg_key)
            sum_force = np.sum(np.array(data2plot['grf']), axis=0)
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
            grf = read_ground_contacts(path_data, leg_key)
            sum_force = np.sum(np.array(grf), axis=0)
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
            stop_gait_list = np.where(np.array(stance) <= stop)[0]
            if len(stop_gait_list) > 0:
                stop_gait = stop_gait_list[-1] + 1
            else:
                stop_gait = start_gait
            stance_plot = stance[start_gait:stop_gait]
            if start_gait % 2 != 0:
                stance_plot.insert(0, start)
            if len(stance_plot) % 2 != 0:
                stance_plot.append(stop)
        else:
            stance_plot = [0, 0]

    fig, axs = plt.subplots(len(data2plot.keys()), sharex=True)
    fig.suptitle('Plots ' + leg_key + ' leg')

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
                    axs.plot(time[start:stop], angle[start:stop], label=name)
                else:
                    axs[i].plot(time[start:stop],
                                angle[start:stop], label=name)
            if len(data2plot.keys()) == 1:
                axs.set_ylabel('Joint angle (deg)')
            else:
                axs[i].set_ylabel('Joint angle (deg)')

        if plot == 'torques':
            for joint, torque in data.items():
                torque_adj = np.delete(torque, 0)
                time = np.arange(0, len(torque_adj), 1) / steps
                if len(data2plot.keys()) == 1:
                    axs.plot(time[start:stop], torque_adj[start:stop]
                             * torqueScalingFactor, label=joint)
                else:
                    axs[i].plot(time[start:stop], torque_adj[start:stop]
                                * torqueScalingFactor, label=joint)

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

        if plot != 'grf' and i == 0:
            if len(data2plot.keys()) == 1:
                plot_handles, plot_labels = axs.get_legend_handles_labels()
            else:
                plot_handles, plot_labels = axs[i].get_legend_handles_labels()
            if collisions_across and sim_data == 'walking':
                gray_patch = mpatches.Patch(color='gray')
                all_handles = plot_handles + [gray_patch]
                all_labels = plot_labels + ['Stance']
            elif collisions_across and sim_data == 'grooming':
                gray_patch = mpatches.Patch(color='dimgray')
                darkgray_patch = mpatches.Patch(color='darkgray')
                if plot_collisions and plot != 'collisions':
                    dark_line = Line2D([0], [0], color='black')
                    gray_line = Line2D([0], [0], color='dimgray')
                    all_handles = plot_handles + \
                        [dark_line] + [gray_line] + [gray_patch] + [darkgray_patch]
                    all_labels = plot_labels + ['Leg vs leg force'] + [
                        'Leg vs antenna force'] + ['Foreleg grooming'] + ['Antennal grooming']
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
                if sim_data == 'walking':
                    c = 'gray'
                if sim_data == 'grooming':
                    if np.sum(leg_vs_leg[stance_plot[ind]
                              :stance_plot[ind + 1]]) > 0:
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


def plot_collisions_diagram(
        sim_data,
        begin=0,
        end=0,
        opt_res=False,
        generation='',
        exp='',
        tot_time=9.0,
        time_step=0.001):
    data = {}
    pkg_path = Path(pkgutil.get_loader("NeuroMechFly").get_filename())
    sim_res_folder = os.path.join(pkg_path.parents[1], 'scripts/KM/results')

    if sim_data == 'walking':
        if not opt_res:
            collisions_data = sim_res_folder + '/grfSC_data_ball_walking.pkl'
        else:
            sim_res_folder = os.path.join(
                pkg_path.parents[1], 'scripts/Optimization/Output_data/grf', exp)
            collisions_data = sim_res_folder + '/grf_optimization_gen_' + generation + '.pkl'
    elif sim_data == 'grooming':
        collisions_data = sim_res_folder + '/selfCollisions_data_ball_grooming.pkl'

    if end == 0:
        end = tot_time

    steps = 1 / time_step
    start = int(begin * steps)
    stop = int(end * steps)

    with open(collisions_data, 'rb') as fp:
        data = pickle.load(fp)

    if sim_data == 'walking':
        title_plot = 'Gait diagram: gen ' + \
            str(int(generation) + 1) if opt_res else 'Gait diagram'
        collisions = {
            'LF': [],
            'LM': [],
            'LH': [],
            'RF': [],
            'RM': [],
            'RH': []}
        for leg, force in data.items():
            collisions[leg[:2]].append(force.transpose()[0])

    elif sim_data == 'grooming':
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
            'RAntenna': []}  # , 'LEye':[], 'REye':[]}
        # for segment, coll in data.items():
        #    for key, forces in coll.items():
        #        if segment in collisions.keys():
        #            collisions[segment].append([np.linalg.norm(force) for force in forces])
        for segment, coll in data.items():
            if segment in collisions.keys():
                for key, vals in coll.items():
                    collisions[segment].append(vals.transpose()[0])

    fig, axs = plt.subplots(len(collisions.keys()),
                            sharex=True, gridspec_kw={'hspace': 0})
    fig.suptitle(title_plot)

    for i, (segment, force) in enumerate(collisions.items()):
        sum_force = np.sum(np.array(force), axis=0)
        segment_force = np.delete(sum_force, 0)
        time = np.arange(0, len(segment_force), 1) / steps
        stance_ind = np.where(segment_force > 0)[0]
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
            stop_gait_list = np.where(np.array(stance) <= stop)[0]
            if len(stop_gait_list) > 0:
                stop_gait = stop_gait_list[-1] + 1
            else:
                stop_gait = start_gait
            stance_plot = stance[start_gait:stop_gait]
            if start_gait % 2 != 0:
                stance_plot.insert(0, start)
            if len(stance_plot) % 2 != 0:
                stance_plot.append(stop)

            for ind in range(0, len(stance_plot), 2):
                axs[i].fill_between(time[stance_plot[ind]:stance_plot[ind + 1]], 0, 1,
                                    facecolor='black', alpha=1, transform=axs[i].get_xaxis_transform())
        else:
            axs[i].fill_between(time[start:stop],
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


def plot_opt_res(
        begin=0,
        end=0,
        key='Coxa',
        plot_act=True,
        plot_torques=True,
        plot_angles=True,
        exp='',
        generation=''):
    pkg_path = Path(pkgutil.get_loader("NeuroMechFly").get_filename())
    opt_res_folder = os.path.join(
        pkg_path.parents[1],
        'scripts/Optimization/Output_data')

    muscles_data = opt_res_folder + '/muscles/' + exp + \
        '/outputs_optimization_gen_' + generation + '.h5'
    angles_data = opt_res_folder + '/angles/' + exp + \
        '/jointpos_optimization_gen_' + generation + '.h5'

    if end == 0:
        end = 5.0

    start = int(begin * 1000)
    stop = int(end * 1000)

    data2plot = {}
    mn_flex = {}
    mn_ext = {}
    torques = {}
    angles = {}
    if plot_act:
        data = pd.read_hdf(muscles_data)
        for joint, val in data.items():
            if key in joint and 'act' in joint and 'active' not in joint and (
                    'Coxa' in joint or 'Femur' in joint or 'Tibia' in joint):
                if key == 'Coxa' and ('M' in joint or 'H' in joint):
                    if 'roll' in joint:
                        name = joint.split('_')[1] + \
                            ' roll ' + joint.split('_')[3]
                        if 'flexor' in joint:
                            mn_flex[name] = val
                        elif 'extensor' in joint:
                            mn_ext[name] = val
                else:
                    name = joint.split('_')[1] + ' ' + joint.split('_')[2]

                    if 'flexor' in joint:
                        mn_flex[name] = val
                    elif 'extensor' in joint:
                        mn_ext[name] = val

        data2plot['mn_flex'] = mn_flex
        data2plot['mn_ext'] = mn_ext

    if plot_torques:
        data = pd.read_hdf(muscles_data)
        for joint, val in data.items():
            if key in joint and 'torque' in joint and (
                    'Coxa' in joint or 'Femur' in joint or 'Tibia' in joint):
                if key == 'Coxa' and ('M' in joint or 'H' in joint):
                    if 'roll' in joint:
                        name = joint.split('_')[1] + ' roll'
                        torques[name] = val
                else:
                    name = joint.split('_')[1]
                    torques[name] = val
        data2plot['torques'] = torques
    if plot_angles:
        data = pd.read_hdf(angles_data)
        for joint, val in data.items():
            # and ('Coxa' in joint or 'Femur' in joint or 'Tibia' in joint):
            if key in joint:
                joint_name = joint.split('_')
                if len(joint_name) > 2:
                    if 'roll' in joint and ('M' in joint or 'H' in joint):
                        name = joint_name[1] + ' ' + joint_name[2]
                        angles[name] = val
                else:
                    if 'F' in joint:
                        name = joint.split('_')[1]
                        angles[name] = val
        data2plot['angles'] = angles

    fig, axs = plt.subplots(len(data2plot.keys()), sharex=True)

    for i, (plot, data) in enumerate(data2plot.items()):
        if plot == 'mn_flex' or plot == 'mn_ext':
            for joint, act in data.items():
                time = np.arange(0, len(act), 1) / 1000
                axs[i].plot(time[start:stop], act[start:stop], label=joint)
            axs[i].set_ylabel(
                'MN activation - ' +
                plot.split('_')[1] +
                ' (a.u.)')
            axs[i].set_ylim(0.3, 2.1)

        if plot == 'torques':
            for joint, torq in data.items():
                time = np.arange(0, len(torq), 1) / 1000
                axs[i].plot(time[start:stop], torq[start:stop]
                            * 100, label=joint)
            axs[i].set_ylabel('Joint torque ' + r'$(\mu Nm)$')
            axs[i].set_ylim(-2, 2)

        if plot == 'angles':
            for joint, ang in data.items():
                time = np.arange(0, len(ang), 1) / 1000
                axs[i].plot(time[start:stop], ang[start:stop]
                            * 180 / np.pi, label=joint)
            axs[i].set_ylabel('Joint angle (deg)')

        axs[i].grid(True)
        axs[i].legend()

    axs[len(axs) - 1].set_xlabel('Time (s)')
    plt.show()


def plot_fly_path(
        begin=0,
        end=0,
        sequence=False,
        save_imgs=False,
        exp='',
        heading=True,
        opt_res=False,
        generations=[]):

    ball_data_list = []
    colors = ['winter', 'copper', 'Purples', 'Greens', 'Oranges']

    fig = plt.figure()
    ax = plt.axes()
    m = MarkerStyle(marker=r'$\rightarrow$')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')

    val_max = 0

    pkg_path = Path(pkgutil.get_loader("NeuroMechFly").get_filename())

    if not opt_res:
        sim_res_folder = os.path.join(
            pkg_path.parents[1], 'scripts/KM/results')
        ball_data_list.append(
            sim_res_folder +
            '/ballRot_data_ball_walking.pkl')
    else:
        sim_res_folder = os.path.join(
            pkg_path.parents[1],
            'scripts/Optimization/Output_data/ballRotations',
            exp)
        for gen in generations:
            ball_data_list.append(
                sim_res_folder +
                '/ballRot_optimization_gen_' +
                gen +
                '.pkl')

    for ind, ball_data in enumerate(ball_data_list):

        with open(ball_data, 'rb') as fp:
            data = pickle.load(fp)

        if end == 0:
            end = len(data)

        data_array = np.array(data)
        x_diff = np.diff(-data_array.transpose()[0])
        y_diff = np.diff(data_array.transpose()[1])
        #th = -np.diff(data_array.transpose()[2])

        x = [0]
        y = [0]
        # , coords in enumerate(x_diff[begin:end]):
        for i in range(begin, end - 1):
            if heading:
                th = data[i][2]
                x_new = x_diff[i] * np.cos(th) - y_diff[i] * np.sin(th)
                y_new = y_diff[i] * np.cos(th) + x_diff[i] * np.sin(th)
                x.append(x[-1] + x_new)
                y.append(y[-1] + y_new)
            else:
                x.append(data[i][0])
                y.append(data[i][1])

            if sequence:
                print('\rFrame: ' + str(i), end='')
                sc = ax.scatter(
                    x,
                    y,
                    c=np.linspace(
                        begin / 100,
                        len(x) / 100,
                        len(x)),
                    cmap='winter',
                    vmin=begin / 100,
                    vmax=end / 100)

                m._transform.rotate_deg(th * 180 / np.pi)
                ax.scatter(x[-1], y[-1], marker=m, s=200, color='black')
                m._transform.rotate_deg(-th * 180 / np.pi)

                if i == 0:
                    sc.set_clim([begin / 100, end / 100])
                    cb = plt.colorbar(sc)
                    cb.set_label('Time (s)')

                ax.set_xlabel('x (mm)')
                ax.set_ylabel('y (mm)')
                if save_imgs:
                    file_name = exp.split('/')[-1]
                    new_folder = 'results/' + \
                        file_name.replace('.pkl', '/') + 'fly_path'
                    if not os.path.exists(new_folder):
                        os.makedirs(new_folder)
                    name = new_folder + '/img_' + '{:06}'.format(i) + '.jpg'
                    fig.set_size_inches(6, 4)
                    plt.savefig(name, dpi=300)
                else:
                    plt.draw()
                    plt.pause(0.001)
                ax.clear()

        if opt_res:
            max_x = np.max(np.array(x))

            if max_x > val_max:
                val_max = max_x

            lim = val_max + 0.05 * val_max
            ax.set_xlim(-2, lim)
            ax.set_ylim(-lim / 2, lim / 2)

        if not sequence:
            #sc = ax.scatter(x,y,c=np.arange(begin/100,end/100,0.01),cmap=colors[ind])
            # sc.set_clim([begin/100,end/100])
            #cb = plt.colorbar(sc)
            # if ind==0:
            #    cb.set_label('Time (s)')
            ax.plot(x, y, linewidth=2, label='Gen ' +
                    str(int(generations[ind]) + 1))
    ax.legend()
    plt.show()


def draw_collisions_on_imgs(
        data_2d,
        experiment,
        sim_data='grooming',
        side='R',
        begin=0,
        end=0,
        save_imgs=False,
        pause=0,
        grfScalingFactor=1,
        scale=3,
        tot_time=9.0,
        time_step_data=0.001,
        time_step_img=0.01):
    data = {}

    pkg_path = Path(pkgutil.get_loader("NeuroMechFly").get_filename())
    sim_res_folder = os.path.join(pkg_path.parents[1], 'scripts/KM/results')

    grf_data = sim_res_folder + '/selfCollisions_data_ball_' + sim_data + '.pkl'

    colors = {
        'a': (
            0, 0, 139), '1': (
            0, 0, 255), '2': (
                71, 99, 255), '3': (
                    92, 92, 205), '4': (
                        128, 128, 240), '5': (
                            122, 160, 255)}

    if end == 0:
        end = tot_time

    steps_data = 1 / time_step_data
    steps_img = 1 / time_step_img
    diff_steps = time_step_img / time_step_data
    start = int(begin * steps_data)
    stop = int(end * steps_data)
    start_img = int(begin * steps_img) + 1
    stop_img = int(end * steps_img) + 2

    with open(grf_data, 'rb') as fp:
        data = pickle.load(fp)
    all_collisions = {
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
        'RAntenna': []}  # , 'LEye':[], 'REye':[]}
    forces = {}
    angles = {}
    for key, val in all_collisions.items():
        if side in key and 'Antenna' not in key:
            forces[key] = val
            angles[key] = val

    for segment, coll in data.items():
        if segment in forces.keys():
            for key, vals in coll.items():
                # if key in all_collisions.keys():
                forces[segment].append(vals.transpose()[0])
                angles[segment].append(vals.transpose()[1])

    raw_imgs = []
    ind_folder = experiment.find('df3d')
    imgs_folder = experiment[:ind_folder - 1]
    for frame in range(start_img, stop_img):
        if side == 'R':
            cam_num = 1
        elif side == 'L':
            cam_num = 5
        img_name = imgs_folder + '/camera_' + \
            str(cam_num) + '_img_' + '{:06}'.format(frame) + '.jpg'
        # print(img_name)
        img = cv.imread(img_name)
        raw_imgs.append(img)

    for i, (leg, force) in enumerate(forces.items()):
        if side in leg:
            sum_force = np.sum(np.array(force), axis=0)
            leg_force = np.delete(sum_force, 0)
            time = np.arange(0, len(leg_force), 1) / 100
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
                stop_gait_list = np.where(np.array(stance) <= stop)[0]
                if len(stop_gait_list) > 0:
                    stop_gait = stop_gait_list[-1] + 1
                else:
                    stop_gait = start_gait
                stance_plot = stance[start_gait:stop_gait]
                if start_gait % 2 != 0:
                    stance_plot.insert(0, start)
                if len(stance_plot) % 2 != 0:
                    stance_plot.append(stop)

                bodyPart = leg[2:]
                if 'Tarsus' in bodyPart:
                    bodyPart = 'Claw'
                    prev = 'Tarsus'
                    if '1' in leg[2:]:
                        div = 1 / 5
                    elif '2' in leg[2:]:
                        div = 2 / 6
                    elif '3' in leg[2:]:
                        div = 3 / 6
                    elif '4' in leg[2:]:
                        div = 4 / 6
                    elif '5' in leg[2:]:
                        div = 5 / 6
                if 'Tibia' in bodyPart:
                    bodyPart = 'Tarsus'
                    prev = 'Tarsus'
                    div = 1
                for ind in range(0, len(stance_plot), 2):
                    for frame in range(
                            stance_plot[ind], stance_plot[ind + 1], int(diff_steps)):
                        cam_frame = int(frame / diff_steps) + 1
                        img_frame = int((frame - start) / diff_steps)

                        x_base = data_2d[cam_num][side +
                                                  'F_leg'][bodyPart][cam_frame][0]
                        y_base = data_2d[cam_num][side +
                                                  'F_leg'][bodyPart][cam_frame][1]

                        x_prev = data_2d[cam_num][side +
                                                  'F_leg'][prev][cam_frame][0]
                        y_prev = data_2d[cam_num][side +
                                                  'F_leg'][prev][cam_frame][1]

                        x_px = int(x_prev + div * (x_base - x_prev))
                        y_px = int(y_prev + div * (y_base - y_prev))

                        start_pnt = [x_px, y_px]
                        force_vals = np.array(forces[leg]).transpose()[
                            frame + 1]

                        poi = np.where(force_vals > 0)[0]
                        if poi.size != 0:
                            mean_angle = np.mean(
                                np.array(
                                    angles[leg]).transpose()[
                                    frame + 1][poi])
                        else:
                            mean_angle = 0
                        # if leg == 'LF':
                        #    print(frame,mean_angle*180/np.pi)

                        force_x = leg_force[frame] * \
                            np.cos(mean_angle) * grfScalingFactor
                        force_z = leg_force[frame] * \
                            np.sin(mean_angle) * grfScalingFactor
                        h, w, c = raw_imgs[img_frame].shape
                        end_pnt = [
                            x_px + (force_x * h / (2 * scale)), y_px - (force_z * h / (2 * scale))]
                        color = colors[leg[-1]]
                        raw_imgs[img_frame] = draw_lines(
                            raw_imgs[img_frame],
                            start_pnt,
                            end_pnt,
                            color=color,
                            thickness=3,
                            arrowHead=True)

    for i, img in enumerate(raw_imgs):
        print('\rFrame: ' + str(i + start_img), end='')
        if save_imgs:
            file_name = experiment.split('/')[-1]
            new_folder = 'results/' + \
                file_name.replace('.pkl', '/') + 'grf_on_raw'
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)
            name = new_folder + '/img_' + '{:06}'.format(i) + '.jpg'
            cv.imwrite(name, img)
        else:
            #print(np.array(grf['LF']).transpose()[i+start+1], np.array(angle['LF']).transpose()[i+start+1]*180/np.pi)
            cv.imshow('img', img)
            cv.waitKey(pause)
    cv.destroyAllWindows()


def draw_grf_on_imgs(
        data_2d,
        experiment,
        sim_data='walking',
        side='R',
        begin=0,
        end=0,
        save_imgs=False,
        pause=0,
        grfScalingFactor=100,
        scale=3,
        tot_time=9.0,
        time_step_data=0.001,
        time_step_img=0.01):
    data = {}

    pkg_path = Path(pkgutil.get_loader("NeuroMechFly").get_filename())
    sim_res_folder = os.path.join(pkg_path.parents[1], 'scripts/KM/results')

    grf_data = sim_res_folder + '/grf_data_ball_' + sim_data + '.pkl'

    colors = {
        'RF_leg': (
            0, 0, 204), 'RM_leg': (
            51, 51, 255), 'RH_leg': (
                102, 102, 255), 'LF_leg': (
                    153, 76, 0), 'LM_leg': (
                        255, 128, 0), 'LH_leg': (
                            255, 178, 102)}

    if end == 0:
        end = tot_time

    steps_data = 1 / time_step_data
    steps_img = 1 / time_step_img
    diff_steps = time_step_img / time_step_data
    start = int(begin * steps_data)
    stop = int(end * steps_data)
    start_img = int(begin * steps_img) + 1
    stop_img = int(end * steps_img) + 2

    with open(grf_data, 'rb') as fp:
        data = pickle.load(fp)
    grf = {'LF': [], 'LM': [], 'LH': [], 'RF': [], 'RM': [], 'RH': []}
    angle = {'LF': [], 'LM': [], 'LH': [], 'RF': [], 'RM': [], 'RH': []}
    for leg, force in data.items():
        grf[leg[:2]].append(force.transpose()[0])
        angle[leg[:2]].append(force.transpose()[1])

    raw_imgs = []
    ind_folder = experiment.find('df3d')
    imgs_folder = experiment[:ind_folder - 1]
    for frame in range(start_img, stop_img):
        if side == 'R':
            cam_num = 1
        elif side == 'L':
            cam_num = 5
        img_name = imgs_folder + '/camera_' + \
            str(cam_num) + '_img_' + '{:06}'.format(frame) + '.jpg'
        # print(img_name)
        img = cv.imread(img_name)
        raw_imgs.append(img)

    for i, (leg, force) in enumerate(grf.items()):
        if side in leg:
            sum_force = np.sum(np.array(force), axis=0)
            leg_force = np.delete(sum_force, 0)
            #time = np.arange(0,len(leg_force),1)/100
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
                stop_gait_list = np.where(np.array(stance) <= stop)[0]
                if len(stop_gait_list) > 0:
                    stop_gait = stop_gait_list[-1] + 1
                else:
                    stop_gait = start_gait
                stance_plot = stance[start_gait:stop_gait]
                if start_gait % 2 != 0:
                    stance_plot.insert(0, start)
                if len(stance_plot) % 2 != 0:
                    stance_plot.append(stop)

                for ind in range(0, len(stance_plot), 2):
                    for frame in range(
                            stance_plot[ind], stance_plot[ind + 1], int(diff_steps)):
                        cam_frame = int(frame / diff_steps) + 1
                        img_frame = int((frame - start) / diff_steps)

                        x_px = int(data_2d[cam_num]
                                   [leg + '_leg']['Claw'][cam_frame][0])
                        y_px = int(data_2d[cam_num]
                                   [leg + '_leg']['Claw'][cam_frame][1])
                        start_pnt = [x_px, y_px]
                        # np.array(grf['LF']).transpose()[i+start+1]
                        force_vals = np.array(grf[leg]).transpose()[frame + 1]
                        poi = np.where(force_vals > 0)[0]
                        if poi.size != 0:
                            mean_angle = np.mean(
                                np.array(
                                    angle[leg]).transpose()[
                                    frame + 1][poi])
                        else:
                            mean_angle = 0
                        # if leg == 'LF':
                        #    print(frame,mean_angle*180/np.pi)

                        force_x = leg_force[frame] * \
                            np.cos(mean_angle) * grfScalingFactor
                        force_z = leg_force[frame] * \
                            np.sin(mean_angle) * grfScalingFactor
                        h, w, c = raw_imgs[img_frame].shape
                        end_pnt = [
                            x_px + (force_x * h / (2 * scale)), y_px - (force_z * h / (2 * scale))]
                        color = colors[leg + '_leg']
                        raw_imgs[img_frame] = draw_lines(
                            raw_imgs[img_frame],
                            start_pnt,
                            end_pnt,
                            color=color,
                            thickness=3,
                            arrowHead=True)

    for i, img in enumerate(raw_imgs):
        print('\rFrame: ' + str(i + start_img), end='')
        if save_imgs:
            file_name = experiment.split('/')[-1]
            new_folder = 'results/' + \
                file_name.replace('.pkl', '/') + 'grf_on_raw'
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)
            name = new_folder + '/img_' + '{:06}'.format(i) + '.jpg'
            cv.imwrite(name, img)
        else:
            #print(np.array(grf['LF']).transpose()[i+start+1], np.array(angle['LF']).transpose()[i+start+1]*180/np.pi)
            cv.imshow('img', img)
            cv.waitKey(pause)
    cv.destroyAllWindows()


def draw_legs_from_2d_data(
        cams_info,
        data_2d,
        exp_dir,
        side='L',
        begin=0,
        end=0,
        saveimgs=False,
        pause=0,
        show_joints=True,
        dir_name='legsJoints'):

    for key, info in cams_info.items():
        r = R.from_dcm(info['R'])
        th = r.as_euler('zyx', degrees=True)[1]
        if 90 - th < 15 and side == 'R':
            data = data_2d[key - 1]
            cam_id = key - 1
        elif 90 - th > 165 and side == 'L':
            data = data_2d[key - 1]
            cam_id = key - 1
        # elif abs(th)+1 < 10:
        #    front_view['F_points2d'] = data_2d[key-1]
        #    front_view['cam_id'] = key-1

    colors = {
        'RF_leg': (
            0, 0, 204), 'RM_leg': (
            51, 51, 255), 'RH_leg': (
                102, 102, 255), 'LF_leg': (
                    153, 76, 0), 'LM_leg': (
                        255, 128, 0), 'LH_leg': (
                            255, 178, 102)}

    if end == 0:
        end = len(data['LF_leg']['Coxa'])

    for frame in range(begin, end):
        df3d_dir = exp_dir.find('df3d')
        folder = exp_dir[:df3d_dir]
        img_name = folder + 'camera_' + \
            str(cam_id) + '_img_' + '{:06}'.format(frame) + '.jpg'
        img = cv.imread(img_name)
        for leg, body_parts in data.items():
            if leg[0] == 'L' or leg[0] == 'R':
                color = colors[leg]
                for segment, points in body_parts.items():
                    if segment != 'Coxa':
                        if 'Femur' in segment:
                            start_point = data[leg]['Coxa'][frame]
                            end_point = points[frame]
                        if 'Tibia' in segment:
                            start_point = data[leg]['Femur'][frame]
                            end_point = points[frame]
                        if 'Tarsus' in segment:
                            start_point = data[leg]['Tibia'][frame]
                            end_point = points[frame]
                        if 'Claw' in segment:
                            start_point = data[leg]['Tarsus'][frame]
                            end_point = points[frame]

                        if show_joints:
                            img = draw_joints(img, start_point, color=color)
                            if 'Claw' in segment:
                                img = draw_joints(img, end_point, color=color)
                        img = draw_lines(
                            img, start_point, end_point, color=color)

        if saveimgs:
            file_name = exp_dir.split('/')[-1]
            new_folder = 'results/' + \
                file_name.replace('.pkl', '/') + 'tracking_2d_' + dir_name + '/'
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)
            name = new_folder + 'camera_' + \
                str(cam_id) + '_img_' + '{:06}'.format(frame) + '.jpg'
            cv.imwrite(name, img)
        else:
            cv.imshow('img', img)
            cv.waitKey(pause)
    cv.destroyAllWindows()


def draw_joints(img, start, radius=8, color=(255, 0, 0), thickness=-1):
    coords = np.array(start).astype(int)
    joint = (coords[0], coords[1])

    cv.circle(img, joint, radius, color, thickness)

    return img


def draw_lines(
        img,
        start,
        end,
        color=(
            255,
            0,
            0),
    thickness=5,
        arrowHead=False):
    coords_prev = np.array(start).astype(int)
    coords_next = np.array(end).astype(int)

    start_point = (coords_prev[0], coords_prev[1])
    end_point = (coords_next[0], coords_next[1])

    if arrowHead:
        if np.linalg.norm(coords_prev - coords_next) > 100:
            tL = 0.25
        else:
            tL = 0.5
        cv.arrowedLine(
            img,
            start_point,
            end_point,
            color,
            thickness,
            tipLength=tL)
    else:
        cv.line(img, start_point, end_point, color, thickness)

    return img


def plot_error(
        errors_dict,
        begin=0,
        end=0,
        name='filename.png',
        dpi=300,
        split=False,
        save=False,
        BL=2.88):
    legs = list(errors_dict.keys())
    angles = list(errors_dict[legs[0]].keys())
    df_errors = pd.DataFrame()

    colors = [(204 /
               255, 0, 0), (1, 51 /
                            255, 51 /
                            255), (1, 102 /
                                   255, 102 /
                                   255), (0, 76 /
                                          255, 153 /
                                          255), (0, 0.5, 1), (102 /
                                                              255, 178 /
                                                              255, 1)]

    if end == 0:
        end = len(errors_dict[legs[0]][angles[0]]['min_error'])

    for leg in legs:
        for angle in angles:
            vals = []
            for err in errors_dict[leg][angle]['min_error'][begin:end]:
                mae = err[0] / (len(err) - 1)
                norm_error = mae / BL * 100
                vals.append(norm_error)

            df_vals = pd.DataFrame(vals, columns=['norm_error'])
            df_vals['leg'] = leg
            df_vals['angle'] = angle

            df_errors = df_errors.append(df_vals, ignore_index=True)
    mean_vals = []

    data = [e.loc[ids, 'norm_error'].values for ids in e.groupby(
        'angle').groups.values()]
    stats.kruskal(*data)
    ph = sp.posthoc_conover(
        e,
        val_col='norm_error',
        group_col='angle',
        p_adjust='holm')
    ph.where(ph > 0.01)
    # stats.kruskal()

    '''
    for angle1 in angles:
        x1 = df_errors['norm_error'].loc[df_errors['angle']==angle1]
        #mean_vals.append(np.mean(x1))
        #print(angle1 + ' mean/std = ' + str(np.mean(x1)) + ' /+- ' + str(np.std(x1)))
        for angle2 in angles[angles.index(angle1)+1:]:
        #if angle != 'base':
            x2 = df_errors['norm_error'].loc[df_errors['angle']==angle2]

            ztest , pval = stests.ztest(x1, x2=x2, value=0, alternative='two-sided')

            print(angle1 + ' vs ' + angle2 + ': ', ztest, pval,)
            if pval > 0.001:
                print(angle1 + " is not statistically different from " + angle2)
        print()

    ax = sns.violinplot(x='angle', y='norm_error', data=df_errors, color="0.8", cut=0, inner='quartiles')
    for violin, alpha in zip(ax.collections[::1], [0.7]*len(angles)):
        violin.set_alpha(alpha)
    ax = sns.stripplot(x='angle', y='norm_error', hue='leg', data=df_errors, dodge=split, jitter=True, zorder=0, size=3)
    #ax = sns.swarmplot(x='angle', y='norm_error', hue='leg', data=df_errors, zorder=0, size=3)
    handles, labels = ax.get_legend_handles_labels()
    if split:
        ax.legend(handles[len(angles)*len(legs):], labels[len(angles)*len(legs):],loc='upper right', bbox_to_anchor=(1.12, 1))
    else:
        ax.legend(handles[len(angles):], labels[len(angles):],loc='upper right', bbox_to_anchor=(1.12, 1))
    #ax.set_yscale('log')
    #ax.set_ylim(0,6)
    #plt.plot(np.arange(len(mean_vals)), mean_vals, 'kP', markersize=10)
    plt.title('Comparison adding an extra DOF')

    figure = plt.gcf()  # get current figure
    w_img = int(len(angles)*1.5)
    h_img = int(w_img*3/4)
    figure.set_size_inches(w_img, h_img) # set figure's size manually to your full screen (32x18)
    if save:
        plt.savefig(name, dpi=dpi,bbox_inches='tight')
    plt.show()
    '''

    return df_errors
