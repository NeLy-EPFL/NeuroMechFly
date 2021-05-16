""" Script to plot the simulation results. """
import os
import cv2 as cv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from .sensitivity_analysis import calculate_forces

"""
import math
import pickle
import pkgutil
import itertools
from pathlib import Path
import matplotlib.ticker as mtick
import matplotlib.transforms as mtransforms
from matplotlib.legend_handler import HandlerTuple
from scipy import stats
from scipy import ndimage
from scipy.spatial.transform import Rotation as R
from sklearn import svm
from sklearn.metrics import mean_squared_error
from statsmodels.stats import weightstats as stests
"""


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
    constant='kv0.9',
    condition='kp0.4_kv0.9',
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

def plot_pareto_front(path_data, g, s=''):
    from ..experiments.network_optimization.neuromuscular_control import DrosophilaSimulation as ds

    if not isinstance(g, list):
        generations = [g]
    else:
        generations = g

    if not isinstance(s, list):
        solutions = [s]
    else:
        solutions = s

    edge = plt.cm.cool(np.linspace(0,1,len(solutions)))
    ax = plt.gca()
    for gen in generations:
        fun_path = os.path.join(path_data,f"FUN.{gen}")    
        fun = np.loadtxt(fun_path)

        color = next(ax._get_lines.prop_cycler)['color']
        plt.scatter(fun[:,0],
                    fun[:,1],
                    c=color,
                    s=60,
                    label=f"gen: {gen+1}")

        for i, sol in enumerate(solutions):
            if sol != '':
                ind = ds.select_solution(sol, fun)
            else:
                ind=-1

            if ind > -1:
                plt.scatter(fun[ind,0],
                            fun[ind,1],
                            c=color,
                            s=60,
                            edgecolors=edge[i],
                            linewidth=3.5,
                            label=f'sol: {ind} ({sol})')
    
    plt.xlabel('Distance')
    plt.ylabel('Stability')
    title = 'Pareto front'
    plt.title(title)
    plt.legend()
    plt.show()

def read_muscles_act(path_data, equivalence, leg_order):
    muscles_path = os.path.join(path_data, 'muscle', 'outputs.h5')
    data_raw = pd.read_hdf(muscles_path)
    data={}
    for leg in leg_order:
        name = f"{leg}_leg"
        data[name]={}
    for leg in data.keys():
        for new_name, old_name in equivalence.items():
            key = f"joint_{leg[:2]}{old_name}"
            for k in data_raw.keys():
                if key in k:
                    name = k.replace(key, new_name)
                    if not ('pitch' in name and 'roll' in name):
                        data[leg][name] = data_raw[k].values

    return data

def read_joint_positions(path_data, equivalence, leg_order):
    angles_path = os.path.join(path_data, 'physics', 'joint_positions.h5')
    angles_raw = pd.read_hdf(angles_path)
    angles={}
    for leg in leg_order:
        name = f"{leg}_leg"
        angles[name]={}
        
    for leg in angles.keys():
        for new_name, old_name in equivalence.items():
            key = f"joint_{leg[:2]}{old_name}"
            angles[leg][new_name] = angles_raw[key].values

    return angles
        

def read_ground_contacts(path_data):
    """Read ground reaction forces data, calculates magnitude for each segment

    Parameters:
        path_data (str): Path to data for plotting

    Return:
        grf (np.array): ground reaction forces for all segments in a specific leg
    """
    grf_data = os.path.join(path_data, 'physics', 'ground_contacts.h5')
    data = pd.read_hdf(grf_data)
    grf = {}
    check = []
    for key, force in data.items():
        leg, force_axis = key.split('_')
        if leg not in check:
            check.append(leg)
            components = [k for k in data.keys() if leg in k]
            data_x = data[components[0]].values
            data_y = data[components[1]].values
            data_z = data[components[2]].values
            res_force = np.linalg.norm([data_x, data_y, data_z], axis=0)
            if leg[:2] not in grf.keys():
                grf[leg[:2]] = []
            grf[leg[:2]].append(res_force)
    
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

def get_stance_periods(leg_force,start,stop):
    """Read ground reaction forces data, calculates magnitude for each segment

    Parameters:
        leg_force (np.array): Forces associated with a leg
        start (float): Starting time for checking stance periods
        stop (float): Stoping time for checking stance periods
    Return:
        stance_plot (list): list with indices indicating beginning and ending of stance periods
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
        stop_gait_list = np.where((np.array(stance) <= stop)&(np.array(stance) > start))[0]
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
        joint_key='ThC',
        sim_data='walking',
        angles={},
        plot_muscles_act=False,
        plot_torques_muscles=False,
        plot_angles_interleg=False,
        plot_angles_intraleg=False,
        plot_torques=True,
        plot_grf=True,
        plot_collisions=True,
        collisions_across=True,
        begin=0.0,
        end=0.0,
        time_step=0.001,
        torqueScalingFactor=1e9,
        grfScalingFactor=1e6
    ):

    """Plot data from the simulation

    Parameters:
        path_data (str): Path to data for plotting
        leg_key (str): Key for specifying a leg to plot (angles, torques, grf, collisions), e.g, LF, LM, LH, RF, RM, RH
        joint_key (str): Key for specifying a joint to plot (angles_interleg, torques_muscles, muscles_act), e.g, Coxa, Femur, Tibia, Tarsus
        sim_data (str, default walking): behavior from data, e.g., walking or grooming
        angles (dict, optional): angles to plot calculated externally, e.g., using df3dPostProcessing
        plot_muscles_act (bool, default True): Plotting muscle's activation
        plot_torques_muscle (bool, default False): Plotting torques generated by muscles
        plot_angles_interleg (bool, default False): Plotting angles from simulation
        plot_angles_intraleg (bool, default True): Plotting angles from external source and defined in variable angles
        plot_torques (bool, default True): Plotting torques generated by simulation controllers
        plot_grf (bool, default True): Plotting ground reaction forces (if sim_data='walking')
        plot_collisions (bool, default True): Plotting self-collision forces (if sim_data='grooming')
        plot_collisions_across (bool, default True): Plotting grf/collisions across other plots
        begin (float, default 0.0): Time point for starting the plot
        end (float, default 0.0): Time point for finishing the plot, if 0.0, all time points are plotted
        time_step (float, default 0.001): Data time step
        torqueScalingFactor (float, default 1.0): Scaling factor for torques
        grfScalingFactor (float, default 1.0): Scaling factor for ground reaction forces
    """
    
    data2plot = {}
    mn_flex = {}
    mn_ext = {}
    torques_muscles = {}
    angles_sim = {}

    equivalence = {'ThC_yaw': 'Coxa_yaw',
                   'ThC_pitch': 'Coxa',
                   'ThC_roll': 'Coxa_roll',
                   'CTr_pitch': 'Femur',
                   'CTr_roll': 'Femur_roll',
                   'FTi_pitch': 'Tibia',
                   'TiTa_pitch': 'Tarsus1'}

    leg_order = ['LF','LM','LH','RF','RM','RH']

    length_data = 0

    if plot_muscles_act:
        muscles_data = read_muscles_act(path_data, equivalence, leg_order)

        for leg, joint_data in muscles_data.items():
            for joint, data in joint_data.items():
                if joint_key in joint:
                    name = f"{leg[:2]} {joint.split('_')[0]} {joint.split('_')[1]}"
                    if 'flexor' in joint:
                        mn_flex[name] = data
                    if 'extensor' in joint:
                        mn_ext[name] = data
                    if length_data == 0:
                        length_data = len(data)

        data2plot['mn_prot'] = mn_flex
        data2plot['mn_ret'] = mn_ext
        
    if plot_torques_muscles:
        muscles_data = read_muscles_act(path_data, equivalence, leg_order)

        for leg, joint_data in muscles_data.items():
            for joint, data in joint_data.items():
                if joint_key in joint:
                    name = f"{leg[:2]} {joint.split('_')[0]} {joint.split('_')[1]}"
                    if 'torque' in joint:
                        torques_muscles[name] = data
                    if length_data == 0:
                        length_data = len(data)
                        
        data2plot['torques_muscles'] = torques_muscles
    
    if plot_angles_interleg:
        angles_data = read_joint_positions(path_data, equivalence, leg_order)

        for leg, joint_data in angles_data.items():
            for joint, data in joint_data.items():
                if joint_key == joint:
                    name = f"{leg[:2]} {joint.replace('_',' ')}"
                    angles_sim[name] = data                
                elif joint_key in joint and 'pitch' in joint:
                    if not ('ThC' in joint_key and ('M' in leg or 'H' in leg)):
                        name = f"{leg[:2]} {joint.replace('_',' ')}"
                        angles_sim[name] = data                    
                elif joint_key in joint and 'roll' in joint:
                    if 'ThC' in joint_key and ('M' in leg or 'H' in leg):
                        name = f"{leg[:2]} {joint.replace('_',' ')}"
                        angles_sim[name] = data
                
                if length_data == 0:
                    length_data = len(data)                
        data2plot['angles_sim'] = angles_sim

    if plot_angles_intraleg:
        if bool(angles):
            angles_raw = angles[leg_key + '_leg']
        else:
            angles_data = read_joint_positions(path_data, equivalence, leg_order)
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
        if plot == 'mn_prot' or plot == 'mn_ret':
            for joint, act in data.items():
                time = np.arange(0, len(act), 1) / steps
                if len(data2plot.keys()) == 1:
                    axs.plot(time[start:stop], act[start:stop], label=joint)
                else:
                    axs[i].plot(time[start:stop], act[start:stop], label=joint)
            if len(data2plot.keys()) == 1:
                axs.set_ylabel('Muscle activation\n' + plot.split('_')[1] + ' (a.u.)')
            else:
                axs[i].set_ylabel('Muscle activation\n' + plot.split('_')[1] + ' (a.u.)')

        if plot == 'torques_muscles':
            for joint, torq in data.items():
                time = np.arange(0, len(torq), 1) / steps
                if len(data2plot.keys()) == 1:
                    axs.plot(time[start:stop], torq[start:stop], label=joint)
                else:
                    axs[i].plot(time[start:stop], torq[start:stop], label=joint)
            if len(data2plot.keys()) == 1:
                axs.set_ylabel('Joint torques\nfrom muscle ' + r'$(\mu Nmm)$')
                axs.set_ylim(-0.5, 0.5)
            else:
                axs[i].set_ylabel('Joint torques\nfrom muscle ' + r'$(\mu Nmm)$')
                axs[i].set_ylim(-0.5, 0.5)
                
        if plot == 'angles_sim':
            for joint, ang in data.items():
                time = np.arange(0, len(ang), 1) / steps
                angle = np.array(ang) * 180 / np.pi
                if len(data2plot.keys()) == 1:
                    axs.plot(time[start:stop], angle[start:stop], label=joint)
                else:
                    axs[i].plot(time[start:stop], angle[start:stop], label=joint)
            if len(data2plot.keys()) == 1:
                axs.set_ylabel('Joint angles (deg)')
            else:
                axs[i].set_ylabel('Joint angles (deg)')

        if plot == 'angles':
            for name, angle_rad in data.items():
                time = np.arange(0, len(angle_rad), 1) / steps
                angle = np.array(angle_rad) * 180 / np.pi
                if len(data2plot.keys()) == 1:
                    axs.plot(time[start:stop], angle[start:stop], label=name.replace('_',' '))
                else:
                    axs[i].plot(time[start:stop],
                                angle[start:stop], label=name.replace('_',' '))
            if len(data2plot.keys()) == 1:
                axs.set_ylabel('Joint angles (deg)')
            else:
                axs[i].set_ylabel('Joint angles (deg)')

        if plot == 'torques':
            for joint, torque in data.items():
                torque_adj = np.delete(torque, 0)
                time = np.arange(0, len(torque_adj), 1) / steps
                if len(data2plot.keys()) == 1:
                    axs.plot(time[start:stop], torque_adj[start:stop]
                             * torqueScalingFactor, label=joint.replace('_',' '))
                else:
                    axs[i].plot(time[start:stop], torque_adj[start:stop]
                                * torqueScalingFactor, label=joint.replace('_',' '))

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

        if (plot != 'grf' and i == 0) or ('angles_sim' not in plot and 'angles' in plot and plot_angles_interleg):
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
        begin=0,
        end=0,
        time_step=0.001):

    """Plot collision/gait diagrams

    Parameters:
        path_data (str): Path to data for plotting
        sim_data (str): behavior from data, e.g., walking or grooming
        
        opt_res (bool, default False): Select if the collision/gait diagrams are from and optimization result
        exp (str): experiment name (if opt_res is True)
        generation (str): Generation number (if opt_res is True)
        begin (float, default 0.0): Time point for starting the plot
        end (float, default 0.0): Time point for finishing the plot, if 0.0, all time points are plotted
        time_step (float, default 0.001): Data time step
    """
    
    data = {}
    length_data = 0
    
    if sim_data == 'walking':
        data = read_ground_contacts(path_data)
        title_plot = 'Gait diagram'
        collisions = {
            'LF': [],
            'LM': [],
            'LH': [],
            'RF': [],
            'RM': [],
            'RH': []}
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
            seg_forces=[]
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
        stance_plot = get_stance_periods(force[0],start,stop)
        for ind in range(0, len(stance_plot), 2):
                axs[i].fill_between(time[stance_plot[ind]:stance_plot[ind + 1]], 0, 1,
                                    facecolor='black', alpha=1, transform=axs[i].get_xaxis_transform())

        axs[i].fill_between(time[start:stance_plot[0]], 0, 1, facecolor='white', alpha=1, transform=axs[i].get_xaxis_transform())

        axs[i].fill_between(time[stance_plot[-1]:stop], 0, 1, facecolor='white', alpha=1, transform=axs[i].get_xaxis_transform())
        
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
        generations=[],
        solutions=[],
        sequence=False,
        heading=True,
        begin=0,
        end=0,
        time_step=0.001
        ):

    """Plot reconstruction of the fly path from ball rotations

    Parameters:
        path_data (str): Path to data for plotting
        generations (list): Numbers of the generations to plot (for optimization experiments)
        generations (list): Name of the solution to plot (for optimization experiments)
        sequence (bool, default False): Plotting path every time step
        heading (bool, default True): Plotting heading of the fly (if sequence=True)
        begin (float, default 0.0): Time point for starting the plot
        end (float, default 0.0): Time point for finishing the plot, if 0.0, all time points are plotted
        time_step (float, default 0.001): Data time step
    """

    ball_data_list = []

    val_max = 0
    val_min = np.inf

    if generations:
        for gen in generations:
            if not solutions:
                gen_folder = os.path.join(path_data, f'gen_{gen}')
                solutions = [s.split('_')[-1] for s in os.listdir(gen_folder)]
            for sol in solutions:
                sim_res_folder = os.path.join(path_data, f'gen_{gen}', f'sol_{sol}','physics','ball_rotations.h5')
                ball_data_list.append(sim_res_folder)
    else:
        sim_res_folder = os.path.join(path_data, 'physics','ball_rotations.h5')
        ball_data_list.append(sim_res_folder)

    fig = plt.figure()
    ax = plt.axes()
    m = MarkerStyle(marker=r'$\rightarrow$')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    colors = plt.cm.Greens(np.linspace(0.3,1,len(ball_data_list)))

    for ind, ball_data in enumerate(ball_data_list):

        data = pd.read_hdf(ball_data)

        if end == 0:
            end = len(data) * time_step

        steps = 1 / time_step
        start = int(begin * steps)
        stop = int(end * steps)

        data_array = np.array(data.values)
        x_diff = np.diff(-data_array.transpose()[0])
        y_diff = np.diff(data_array.transpose()[1])

        x = [0]
        y = [0]
        
        for count, i in enumerate(range(start, stop-1)):
            if heading:
                th = data_array[i][2]
                x_new = x_diff[i] * np.cos(th) - y_diff[i] * np.sin(th)
                y_new = y_diff[i] * np.cos(th) + x_diff[i] * np.sin(th)
                x.append(x[-1] + x_new)
                y.append(y[-1] + y_new)
            else:
                x.append(data_array[i][0])
                y.append(data_array[i][1])

            if sequence:
                ax.clear()
                curr_time = (i+2)/steps
                print(f'\rTime: {curr_time:.3f}', end='')
                sc = ax.scatter(
                    x,
                    y,
                    c=np.linspace(
                        begin,
                        begin+len(x)/steps,
                        len(x)),
                    cmap='winter',
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
                
        
        if generations:
            max_x = np.max(np.array(x))
            min_x = np.min(np.array(x))

            if max_x > val_max:
                val_max = max_x

            if min_x < val_min:
                val_min = min_x

            lim = val_max + 0.05 * val_max
            low = val_min - 0.05 * val_min
            ax.set_xlim(low, lim)
            ax.set_ylim(-lim / 2, lim / 2)

        if not sequence:
            if generations:
                gen_label = generations[int(ind/len(solutions))] + 1
                sol_label = solutions[int(ind%len(solutions))]
                ax.plot(x,
                        y,
                        linewidth=2,
                        label=f'Gen {gen_label}-{sol_label}',
                        c=colors[ind])
                ax.legend()
            else:
                ax.plot(x, y, linewidth=2)
    
    plt.show()

