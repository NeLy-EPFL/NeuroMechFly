import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from NeuroMechFly.utils.sensitivity_analysis import (
    calculate_MSE_joints, calculate_statistics_joints)
from NeuroMechFly.utils.plotting import (
    plot_mu_sem, plot_kp_joint, heatmap_plot)

joints = [
    'Coxa',
    'Coxa_yaw',
    'Coxa_roll',
    'Femur',
    'Femur_roll',
    'Tibia',
    'Tarsus1']
legs = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']
beg = 3600
end = 4600
time_step = 0.001
gain = np.arange(0.1, 1.1, 0.1)

joints_torque = ['joint_LFCoxa', 'joint_LMCoxa', 'joint_LHCoxa']
unit_torque = '$\\mu$Nmm'
legs_grf = ['LF', 'LM', 'LH']
unit_force = '$\\mu$N'


def plot_figure_s1(save_fig: bool):
    """ Regenerates the Supplementary Figure 1. """
    joint_angles_mse_df, _ = calculate_MSE_joints(
        joint_angles, baseline_position, beg=100)
    joint_vel_mse_df, _ = calculate_MSE_joints(
        joint_velocities, baseline_velocity, beg=100)

    pivot_pos, pivot_vel = 0, 0
    for leg in legs:
        temp_pos = joint_angles_mse_df.loc[joint_angles_mse_df['Leg'] == leg, :]
        pivot_pos += temp_pos.pivot(index='Kp', columns='Kv', values='MSE')
        temp_vel = joint_vel_mse_df.loc[joint_vel_mse_df['Leg'] == leg, :]
        pivot_vel += temp_vel.pivot(index='Kp', columns='Kv', values='MSE')

    fig, ax = plt.subplots(figsize=(8, 8))
    heatmap_plot(
        title="Joint Angles with Varying Kp and Kv",
        joint_data=pivot_pos,
        precision=".3f",
        colorbar_title="MSE(rad$^2$)",
        ax=ax,
        cmap="rocket")
    if save_fig:
        plt.savefig('./figureS1_A.png')
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 8))
    heatmap_plot(
        title="Joint Velocities with Varying Kp and Kv",
        joint_data=pivot_vel,
        precision=".1f",
        colorbar_title="MSE((rad/s)$^2$)",
        ax=ax,
        cmap="rocket")
    if save_fig:
        plt.savefig('./figureS1_B.png')
    plt.show()


def plot_figure_s2_a(save_fig: bool):
    """ Regenerates the Supplementary Figure 2 A. """
    fig, ax = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    for i, joint in enumerate(joints_torque):
        plot_kp_joint(
            joint_torques,
            show_vector=False,
            calc_force=False,
            full_name=joint,
            ax=ax[i],
            gain_range=np.arange(0.1, 1.1, 0.1),
            constant='Kv0.9',
            condition='Kp0.4_Kv0.9',
            beg=beg,
            intv=end - beg,
            time_step=time_step
        )

        ax[i].set(
            xlabel='Time (s)',
            ylabel='Joint Torques ({})'.format(unit_torque),
            title='{} {} Pitch Joint Torque wrt Changing Kp'.format(joint[6:8], joint[-4:])
        )

    plt.tight_layout()
    if save_fig:
        plt.savefig('./figure_s2_a.png')
    plt.show()


def plot_figure_s2a_extra(save_fig: bool):
    """ Plots the Supplementary Figure 2 A's Variation. """
    torques_stat = calculate_statistics_joints(
        joint_torques,
        scaling_factor=1,
        constant='Kv0.9',
        force_calculation=False,
        joints=[
            'Coxa',
            'Coxa_yaw',
            'Coxa_roll',
            'Femur',
            'Femur_roll',
            'Tibia',
            'Tarsus1'])

    fig, ax = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    for i, joint in enumerate(joints_torque):
        # Plot standard error and mean of one single joint torque
        plot_mu_sem(
            mu=torques_stat[joint]['mu'],
            error=torques_stat[joint]['stderr'],
            ax=ax[i],
            beg=3600,
            end=4600,
        )

        ax[i].set(
            xlabel='Time (s)',
            ylabel='Joint Torques ({})'.format(unit_torque),
            title='{} {} Pitch Joint Torque Mean and Std Error'.format(joint[6:8], joint[-4:])
        )
    if save_fig:
        plt.savefig('./figure_s2a_extra.png')
    plt.show()


def plot_figure_s2b(save_fig: bool):
    """ Regenerates the Supplementary Figure 2 B. """

    fig, ax = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    for i, leg in enumerate(legs_grf):
        plot_kp_joint(
            ground_contact_forces,
            show_vector=False,
            calc_force=True,
            full_name=leg,
            gain_range=np.arange(0.1, 1.1, 0.1),
            ax=ax[i],
            constant='Kv0.9',
            condition='Kp0.4_Kv0.9',
            beg=beg,
            intv=end - beg
        )

        ax[i].set(
            xlabel='Time (s)',
            ylabel='Ground Reaction Forces ({})'.format(unit_force),
            title='{} Leg Ground Reaction Forces wrt Changing Kp'.format(leg)
        )
    if save_fig:
        plt.savefig('./figure_s2b.png')
    plt.show()


def plot_figure_s2b_extra(save_fig: bool):
    """ Plots the Supplementary Figure 2 B's Variation. """
    grf_stat = calculate_statistics_joints(
        ground_contact_forces,
        scaling_factor=1,
        constant='Kv0.9',
        force_calculation=True,
    )

    fig, ax = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    for i, leg in enumerate(legs_grf):
        # Plot standard error and mean of one single leg GRF
        plot_mu_sem(
            mu=grf_stat[leg]['mu'],
            error=grf_stat[leg]['stderr'],
            ax=ax[i],
            beg=beg,
            end=end,
        )

        ax[i].set(
            xlabel='Time (s)',
            ylabel='Ground Reaction Forces ({})'.format(unit_force),
            title='{} Leg Ground Reaction Forces wrt Changing Kp'.format(leg)
        )
    if save_fig:
        plt.savefig('./figure_s2b_extra.png')
    plt.show()


def load_data(pybullet_path, baseline_path_pos, baseline_path_vel):
    """ Loads the sensitivity analysis data. """
    global ground_contact_forces
    global joint_torques
    global joint_angles
    global joint_velocities
    global baseline_position
    global baseline_velocity

    #: Load sensitivity analysis results
    pybullet_data = pd.read_pickle(pybullet_path)

    #: Load baseline files
    baseline_position = pd.read_pickle(baseline_path_pos)
    baseline_velocity = pd.read_pickle(baseline_path_vel)

    #: Get the data from the physics dictionary
    ground_contact_forces = pybullet_data['ground_contacts']
    joint_torques = pybullet_data['joint_torques']
    joint_angles = pybullet_data['joint_positions']
    joint_velocities = pybullet_data['joint_velocities']


if __name__ == '__main__':
    """ Main. """
    # NOTE: User should change the paths according to where they download the
    # data
    path_pybullet = '../data/sensitivity_analysis/pybullet_data'
    baseline_path_pos = '../data/sensitivity_analysis/7sc_nameconv_walking_joint_angle.pkl'
    baseline_path_vel = '../data/sensitivity_analysis/7sc_nameconv_walking_joint_velocity.pkl'
    #: True if figures to be saved
    save_figures = False
    #: Read data
    load_data(path_pybullet, baseline_path_pos, baseline_path_vel)
    #: Plot figures
    plot_figure_s1(save_figures)
    plot_figure_s2_a(save_figures)
    plot_figure_s2a_extra(save_figures)
    plot_figure_s2b(save_figures)
    plot_figure_s2b_extra(save_figures)