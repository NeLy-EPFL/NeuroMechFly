""" Script that contains the functions to perform sensitivity analysis. """
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

##### GLOBAL VARIABLES #####
legs = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']

joints = [
    'Coxa',
    'Coxa_yaw',
    'Coxa_roll',
    'Femur',
    'Femur_roll',
    'Tibia',
    'Tarsus1']

##### CALCULATE STATISTICS #####
def calculate_forces(leg, k_value, *args):
    """ Computes the ground reaction force on one single leg.

    Args
    ----
        leg (string): name of the leg, i.e., 'LF' 'RM'
        k_value (string): value of the gain, i.e. 'kp1.0_kv0.9'
        args: dictionary containing the measured forces
    
    Returns
    -------
        force_vector (np.array): (3, length) array containing GRF forces in x, y, z
        force_norm (np.array): (length,) array containing the norm of the GRF forces
    """
    force = {'x': 0, 'y': 0, 'z': 0}
    for key in args[0][k_value].keys():
        for ax in ['x', 'y', 'z']:
            if leg in key and ax in key:
                force[ax] += sum(f[k_value][key]
                                 for f in args if f is not None)

    force_vector = np.vstack([force[ax] for ax in force.keys()])
    force_norm = np.linalg.norm(force_vector, axis=0)
    return force_vector, force_norm


def calculate_stack_array(
    *args,
    force_calc,
    leg=None,
    constant='kv0.9',
    scaling_factor=1
):
    """ Concatenates and scales physical quantities.
   
    Args
    ----
        args: arrays to be concatenated
        force_cal (bool): if true, then calculates the norm of the vector
        leg (string): name of the leg, i.e., 'LF' 'RM'
        constant (string): value of the constant gain, i.e. 'kv0.9'
        scaling_factor: scales the force and torque measurements, used for unit changes 
            i.e. 1e6 for forces and 1e9 for torques
   
    Returns
    -------
        stack_array(np.array): array of values that have the same constant gain (kp or kv)
    """    
    first_iteration = True
    for k_value in args[0].keys():
        if constant in k_value:
            if force_calc:
                _, data_stack = calculate_forces(leg, k_value, *args)
            else:
                data_stack = np.array(args[0][k_value][leg])

            if first_iteration:
                stack_array = data_stack
                first_iteration = False
            else:
                stack_array = np.vstack(
                    (stack_array, data_stack)
                )

    return stack_array * scaling_factor


def calculate_statistics_joints(
    *args,
    scaling_factor=1,
    constant='kv0.9',
    force_calculation=False,
    joints=[
        'Coxa',
        'Coxa_yaw',
        'Coxa_roll',
        'Femur',
        'Femur_roll',
        'Tibia',
        'Tarsus1']):
    """ Calculates statistical properties of joint physical quantities.

    Args
    ----
        scaling_factor (int, optional): scales the force and torque measurements, used for unit changes
        constant (str, optional): used for fixing one of two independent variables. E.g. 'kv0.9'
        force_calculation (bool, optional): true-> calculates force, false->calculates torque, angles, velocity
        joints (list, optional): if GRF then ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']

    Returns
    -------
        stat_joints (dict): dict contains mean, standard deviaton and standard error of the given data
    """
    stat_joints = {}
    for leg, joint in itertools.product(legs, joints):
        name = leg if force_calculation else 'joint_' + leg + joint
        stack_array = calculate_stack_array(
            *args,
            force_calc=force_calculation,
            leg=name,
            constant=constant,
            scaling_factor=scaling_factor)
        stat_joints[name] = calculate_stats(stack_array)

    return stat_joints


def calculate_stats(data):
    """ Calculates, std, mean, and stderror of a given data. 

    Args
    ----
        data (np.array): array(n_gains, length) containining the physical quantities of different gain values

    Returns
    -------
        stat_dict (dict): dict contains mean, standard deviaton and standard error of the given data"""
    stat_dict = {}
    stat_dict['mu'] = np.mean(data, axis=0)
    stat_dict['stderr'] = np.std(data, ddof=1, axis=0) / np.sqrt(data.shape[0])
    stat_dict['std'] = np.std(data, axis=0)

    return stat_dict


def calculate_mse_joints(
    joint_data,
    ground_truth,
    beg=100
):
    """ Calculates MSE between the ground truth and given data.

    Args
    ----
        joint_data (dict): dictionary containing the joint information (angle or velocity)
        ground_truth (dict): dictionary containing the ground truth angle or velocity data
        beg (int, optional): beginning of the process. Defaults to 100.

    Returns
    -------
        error_df (pd.DataFrame): mean squared error between the baseline and the simulation values 
        in a pandas data frame format
        error_dict (dictionary): mean squared error between the baseline and the simulation values 
        in a dictionary format
    """
    leg_mse = []
    error_dict = {leg: {} for leg in legs}
    for gain_name in joint_data.keys():
        for leg in legs:
            mse = 0
            for joint in joints:
                key_name = 'joint_' + leg + joint
                joint_baseline = ground_truth[key_name][beg:]
                joint_comparison = joint_data[gain_name][key_name][beg:]
                #assert len(joint_baseline) == len(joint_comparison), "Two arrays should be of the same length"
                
                mse += mean_squared_error(joint_baseline, joint_comparison)

            error_dict[leg][gain_name] = mse / len(joints)
            leg_mse.append([leg,
                            float(gain_name[2:5]),
                            float(gain_name[-3:]),
                            mse / len(joints)])

    error_df = pd.DataFrame(leg_mse, columns=['Leg', 'Kp', 'Kv', 'MSE'])
    return error_df, error_dict