"""Generate Drosophila Template"""

import os
import numpy as np
from NeuroMechFly.sdf.sdf import ModelSDF, Link, Joint, Inertial, Collision
from NeuroMechFly.sdf.units import SimulationUnitScaling
from NeuroMechFly.sdf import utils
import yaml
from df3dPostProcessing import df3dPostProcess

MODEL_NAME = "drosophila_100x_limits_from_data2"

def main():
    """Main"""

    units = SimulationUnitScaling(
        meters=1e0,
        seconds=1e0,
        kilograms=1e0
    )

    #: Read the sdf model from template
    model = ModelSDF.read(
        'drosophila_100x_noLimits.sdf')[0]
    model.change_units(units)

    link_index = utils.link_name_to_index(model)
    joint_index = utils.joint_name_to_index(model)

    ####### CALCULATE ANGLES ##########

    experiment = '/home/lobato/Desktop/DF3D_data/180921_aDN_PR_Fly8_005_SG1_behData_images/images/df3d/pose_result__home_nely_Desktop_animationSimfly_video2_180921_aDN_PR_Fly8_005_SG1_behData_images_images.pkl' ### walking

    df3d = df3dPostProcess(experiment)
    align = df3d.align_3d_data()
    angles = df3d.calculate_leg_angles()

    ######## CHANGE JOINT LIMITS #########
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
    #print(actuated_joints)

    limits={}
    equivalence = {"Coxa":"pitch","Coxa_roll":"roll","Femur":"th_fe","Tibia":"th_ti"}
    for leg, dof in angles.items():
        for angle, val in dof.items():
            mean_val = np.mean(val)
            std_val = np.std(val)
            max_lim = mean_val + 2*std_val
            min_lim = mean_val - 2*std_val
            print(leg, angle, np.array([min_lim, max_lim, mean_val, std_val])*180/np.pi)
            for new_name, original_name in equivalence.items():
                if angle == original_name:
                    name = leg[:2] + new_name
                    if name not in limits.keys():
                        limits[name] = [min_lim,max_lim]
                    #if min_lim > limits[name][0]:
                    #    limits[name][0] = min_lim
                    #if max_lim < limits[name][1]:
                    #    limits[name][1] = max_lim

    for leg, lim in limits.items():
        print(leg, np.array(lim)*180/np.pi)

    for joint in actuated_joints:
        joint_obj = model.joints[joint_index[joint]]
        label = joint.replace('joint_','')
        joint_obj.axis.limits[0] = limits[label][0]
        joint_obj.axis.limits[1] = limits[label][1]
    
    '''
    for joint in model.joints:
        joint_obj = model.joints[joint_index[joint.name]]
        joint_obj.axis.limits[0] = -np.pi
        joint_obj.axis.limits[1]= np.pi
        if 'LM' in joint.name or 'LH' in joint.name:# or 'LF' in joint.name or 'RF' in joint.name:
            ljoint = model.joints[joint_index[joint.name]]
            ljoint.axis.xyz=[-axis if axis == 1 else axis for axis in ljoint.axis.xyz]
    '''
    ########## WRITE ##########    
    model.units = units
    model.write(filename="{}.sdf".format(MODEL_NAME))
        
if __name__ == '__main__':
    main()
