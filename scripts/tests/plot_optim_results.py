
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from NeuroMechFly.utils.plotting import plot_penalties, plot_pareto_front

joints = ['Coxa','Femur', 'Tibia']
legs = ['LF','LM','LH','RF','RM','RH']


def plot_joint_info(result_path: str, quantity: str):
    """
    path = '/NeuroMechFly/scripts/neuromuscular_optimization/simulation_last_run/gen_final/sol_slowest/'
    quantity = torques, velocites, positions etc.
    """
    quantity = quantity.lower()
    if quantity in ('velocities', 'positions'):
        path = os.path.join(result_path, f'physics/joint_{quantity}.h5')
        suffix = ''
    elif quantity == 'torques':
        path = os.path.join(result_path, 'muscle/outputs.h5')
        suffix = '_torque'
    else:
        raise ValueError('Quantity cannot be read!')

    print(path)
    data = pd.read_hdf(path)

    for leg in legs:
        fig, ax = plt.subplots(figsize = (10, 5))
        for joint in joints:
            if leg[-1]=='M' or leg[-1]=='H':
                joint = joint.replace('Coxa', 'Coxa_roll')
            name = f'joint_{leg}{joint}{suffix}'
            ax.plot(data[name], label = name)
        ax.legend()
        ax.set_title(f'joint {quantity}')
        plt.show()


if __name__ == '__main__':

    opt_path = os.path.join(os.getcwd(), '../neuromuscular_optimization')

    plot_penalty = True
    plot_joints = False

    if plot_penalty:
        # path = os.path.join(opt_path, '/run_Drosophila_var_63_obj_2_pop_20_gen_70_211003_180829')
        path = os.path.join(opt_path, 'optimization_results/run_Drosophila_var_63_obj_2_pop_20_gen_70_211003_225816')
        print(path)
        gens = list(np.arange(8,70,2))
        ind_num = 1
        variable_names = ['Distance', 'Stability', 'Lava', 'Velocity', 'Joint Lim']

        plot_penalties(path, gens, ind_num, variable_names)
        plt.show()

        plot_pareto_front(path, gens)
        plt.show()

    if plot_joints:
        path = os.path.join(opt_path, 'simulation_run_Drosophila_var_63_obj_2_pop_20_gen_100_210927_231846/gen_43/sol_slowest')
        plot_joint_info(path, 'positions')