import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
import pickle
import os
from .sensitivity_analysis import calculate_forces
plt.style.use('seaborn-colorblind')

legs = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']
joints = [
    'Coxa',
    'Coxa_yaw',
    'Coxa_roll',
    'Femur',
    'Femur_roll',
    'Tibia',
    'Tarsus1']


file_names = ['ground_contacts',
              'ground_friction_dir1',
              'ground_friction_dir2',
              'joint_positions',
              'joint_torques',
              'joint_velocities',
              'thorax_force']

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
        precision="d",
        linewidth="0.005",
        ax=None,
        cmap='viridis'):
    """ Plots a heatmap plot for global sensitivity analysis. """
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
