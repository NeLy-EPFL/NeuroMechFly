"""Generate a new drosophila template based on the joint data"""

import os
import numpy as np
from NeuroMechFly.sdf.sdf import ModelSDF, Link, Joint, Inertial, Collision
from NeuroMechFly.sdf.units import SimulationUnitScaling
from NeuroMechFly.sdf import utils
import yaml
import glob
import pickle

MODEL_NAME = "neuromechfly_noLimits.sdf"
NEW_MODEL_NAME = "neuromechfly_limitsFromData.sdf"

def main():
    """Main"""

    units = SimulationUnitScaling(
        meters=1e0,
        seconds=1e0,
        kilograms=1e0
    )

    #: Read the sdf model from template
    model = ModelSDF.read(MODEL_NAME)[0]
    link_index = utils.link_name_to_index(model)
    joint_index = utils.joint_name_to_index(model)

    #: Load joint angles during walking
    angles_path = glob.glob('../../data/walking/df3d/joint_angles*.pkl') 

    if angles_path:
        with open(angles_path[0], 'rb') as f:
            angles = pickle.load(f)
    else:
        raise Exception("Angles file not found")

    # : Change joint limits based on the data
    sides = ('L', 'R')
    positions = ('F', 'M', 'H')
    _joints = ('Coxa', 'Femur', 'Tibia')
    actuated_joints = [
        'joint_{}{}{}'.format(side, pos, joint)
        for side in sides
        for pos in positions
        for joint in _joints
    ]
    for j, joint in enumerate(actuated_joints):
        pos = joint.split('_')[1][1]
        if (('M' == pos) or ('H' == pos)) and ('Coxa' in joint):
            actuated_joints[j] = joint.replace('Coxa', 'Coxa_roll')

    limits={}
    equivalence = {"Coxa":"pitch","Coxa_roll":"roll","Femur":"th_fe","Tibia":"th_ti"}
    for leg, dof in angles.items():
        for angle, val in dof.items():
            mean_val = np.mean(val)
            std_val = np.std(val)
            max_lim = mean_val + 1*std_val
            min_lim = mean_val - 1*std_val
            for new_name, original_name in equivalence.items():
                if angle == original_name:
                    name = leg[:2] + new_name
                    if name not in limits.keys():
                        limits[name] = [min_lim,max_lim]
                    #if min_lim > limits[name][0]:
                    #    limits[name][0] = min_lim
                    #if max_lim < limits[name][1]:
                    #    limits[name][1] = max_lim

    for joint in actuated_joints:
        joint_obj = model.joints[joint_index[joint]]
        label = joint.replace('joint_','')
        joint_obj.axis.limits[0] = limits[label][0]
        joint_obj.axis.limits[1] = limits[label][1]
    
    #: Write the joint limits in a new file 
    model.write(filename="{}".format(NEW_MODEL_NAME))
        
if __name__ == '__main__':
    main()
