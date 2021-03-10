import pickle
import itertools
import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d,splrep,splev,pchip_interpolate
from .sensitivity_analysis import take_derivative

############################################### GLOBAL VARIABLES ###############################################
legs = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']

joints = ['Coxa', 'Coxa_yaw', 'Coxa_roll', 'Femur', 'Femur_roll', 'Tibia', 'Tarsus1']

dict_conversion = {'Coxa':'pitch', 
'Coxa_yaw':'yaw', 
'Coxa_roll':'roll', 
'Femur':'th_fe', 
'Femur_roll':'roll_tr', 
'Tibia':'th_ti', 
'Tarsus1':'th_ta'}

file_names = ['ground_contacts', 
'ground_friction_dir1', 
'ground_friction_dir2', 
'joint_positions', 
'joint_torques', 
'joint_velocities',
'thorax_force']


############################################### LOAD DATA ###############################################
def load_pickle(full_path):
    """ Loads the pickle file. """
    try:
        with open(full_path,'rb') as f:
            d = pickle.load(f)
        return d
    except FileNotFoundError:
        print(f"{full_path} is not a valid path")


def save_pickle(variable, file_name, path='./'):
    """ Saves the variable as a pickle file. """
    try:
        with open(path+file_name,'wb') as f:
            pickle.dump(variable,f)
            print('Saved successfully!')
    except IOError:
        print(f"{path} does not exist!")

def load_data(path):
    """ Loads the pybullet data from h5 files into a pandas df. """
    pybullet_data = {key: dict() for key in file_names}
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} is not a valid path")
     
    for file_name in os.listdir(path):
        if file_name.startswith('.'):
            continue
        for pybullet_file_name in file_names:
            print(f"Loading {file_name}...")
            dir_name = path + '/'+ file_name + '/physics/' + pybullet_file_name + '.h5'
            pybullet_data[pybullet_file_name][file_name] = pd.read_hdf(dir_name)
    return pybullet_data

def get_physics_data(load_path, save_path, READ_DATA):
    if READ_DATA:
        pybullet_data = load_data(load_path)
        with open(save_path,'wb') as f:
            pickle.dump(pybullet_data, f)
    else:
        pybullet_data=load_pickle(load_path) 

    return pybullet_data

############################################### PREPROCESS ###############################################
def convert_dictionary(nested_dict):
    """ Converts df3dPostProcess dicts into PyBullet compatible structure. """
    new_dict = dict() 
    for leg, joint in itertools.product(legs,joints):
        new_name = 'joint_'+leg+joint
        key1, key2 = leg+'_leg', dict_conversion[joint]
        new_dict[new_name] = nested_dict[key1][key2]
    return new_dict

def add_offset(angle_data):
    """ Adds offsets to df3dPostProcess results to make them compatible with the fly model."""
    new_angle = {leg: {joint: list() for joint in angle_data[leg].keys()} for leg in angle_data.keys()}
    new_velocity = {leg: {joint: list() for joint in angle_data[leg].keys()} for leg in angle_data.keys()}
    for leg in angle_data.keys():
        for joint in angle_data[leg].keys():
            if 'H' in leg and joint == 'roll':
                new_angle[leg][joint] = np.pi + \
                    np.array(angle_data[leg][joint][1000:8001])
            elif 'M' in leg and joint == 'roll':
                new_angle[leg][joint] = -np.pi*0.5 + \
                    np.array(angle_data[leg][joint][1000:8001])
            elif joint == 'th_fe' or joint == 'th_ta':
                new_angle[leg][joint] = -np.pi + \
                    np.array(angle_data[leg][joint][1000:8001])
            elif joint == 'th_ti':
                new_angle[leg][joint] = np.pi + \
                    np.array(angle_data[leg][joint][1000:8001])
            else:
                new_angle[leg][joint] = \
                    np.array(angle_data[leg][joint][1000:8001])

            new_velocity[leg][joint] = \
                take_derivative(0.001,new_angle[leg][joint])
    return new_angle, new_velocity

############################################### CALCULATE INTERPOLATION AND SMOOTHING ###############################################
def calculate_interpolation(x_points, y_points, x_new_points, **kwargs):
    """  Interpolates the data.

    Args:
        x_points (array_like): known x coordinates.
        y_points (array_like): known y coordinates of shape (x_points.shape, R). 
        x_new_points (array_like): points or point at which to intepolate.
        **kwargs
        int_type (str, optional): interpolation type. Defaults to 'Spline'.
        derivative (int, optional): derivative order. Defaults to 0.

    Raises:
        NameError

    Returns:
        [ndarray]: interpolated signal.
    """    
    int_type = kwargs.get('int_type', 'Spline')
    derivative = kwargs.get('derivative', 0)

    if int_type == 'Linear':
        f_linear= interp1d(x_points,y_points,kind='linear')
        y = f_linear(x_new_points)
    elif int_type == 'Cubic':
        f_cubic = splrep(x_points,y_points,s=0.0, k=3)
        y = splev(x_new_points,f_cubic,der=derivative)
    elif int_type == 'Spline':
        f_spline = splrep(x_points,y_points,s=0.05, k=3)
        y = splev(x_new_points,f_spline,der=derivative)
    elif int_type == 'Pchip':
        y = pchip_interpolate(x_points, y_points, x_new_points)
    else:
        raise NameError('Please enter Linear, Cubic, Spline,\
             or Pchip as the interpolation type!')
    return y

def calculate_interpolation_dict(x_points, data, x_new_points, **kwargs):
    """ Calculate the interpolation of the entire dictionary and returns interpolation dicts and their derivatives. """
    signal_int, signal_int_der = dict(), dict()
    int_type = kwargs.get('int_type', 'Spline')

    for joint in data.keys():
        signal_int[joint] = calculate_interpolation(x_points,\
             data[joint],x_new_points, **kwargs)
        if int_type == 'Spline' or int_type == 'Cubic':
            signal_int_der[joint] = calculate_interpolation(x_points,\
                 data[joint],x_new_points, **kwargs)
        else: 
            time_step = x_new_points[1]-x_new_points[0]
            signal_int_der[joint] = np.diff(signal_int[joint])*1/time_step

    return signal_int, signal_int_der
