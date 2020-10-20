"""Generate Drosophila Template"""

import os
import numpy as np
from farms_sdf.sdf import ModelSDF, Link, Joint, Inertial, Collision
from farms_sdf.units import SimulationUnitScaling
from farms_sdf import utils
import yaml
import farms_pylog as pylog

MODEL_NAME = "drosophila_100x_noLimits"

def add_planar_constraint(model, units, joint_type='prismatic'):
    """ Add planar constraints to the model. """
    #: Add planar joints
    world_link = Link(
        name='world',
        pose=np.zeros((6,)),
        visual=None,
        collision=None,
        inertial=Inertial.empty(units),
        units=units
    )
    support_link_1 = Link(
        name='prismatic_support_1',
        pose=np.zeros((6,)),
        visual=None,
        collision=None,
        inertial=Inertial.empty(units), 
        units=units
    )
    support_link_2 = Link(
        name='prismatic_support_2',
        pose=np.zeros((6,)),
        visual=None,
        collision=None,
        inertial=Inertial.empty(units), 
        units=units
    )
    prismatic_1 = Joint(
        name='prismatic_support_1',
        joint_type=joint_type,
        parent=world_link,
        child=support_link_1,
        xyz=[0.0, 1.0, 0.0],
        limits=[1, -1, 0, 100]
    )
    prismatic_2 = Joint(
        name='prismatic_support_2',
        joint_type=joint_type,
        parent=support_link_1,
        child=support_link_2,
        xyz=[0.0, 0.0, 1.0],
        limits=[1, -1, 0, 100]
    )
    root = utils.find_root(model)
    link_name_id = utils.link_name_to_index(model)
    revolute_1 = Joint(
        name='revolute_support_1',
        joint_type='continuous' if joint_type!='fixed' else joint_type,
        parent=support_link_2,
        child=model.links[link_name_id[root]],
        xyz=[1.0, 0.0, 0.0],
    )
    
    model.links.append(world_link)
    model.links.append(support_link_1)
    model.links.append(support_link_2)
    model.joints.append(prismatic_1)
    model.joints.append(prismatic_2)
    model.joints.append(revolute_1)

def main():
    """Main"""

    units = SimulationUnitScaling(
        meters=1e0,
        seconds=1e0,
        kilograms=1e0
    )

    #: Read the sdf model from template
    model = ModelSDF.read(
        'drosophila_100x.sdf')[0]
    model.change_units(units)

    link_index = utils.link_name_to_index(model)
    joint_index = utils.joint_name_to_index(model)

    #: Change collision shapes of feet to ball
    #: FEET
    '''
    sides = ('L', 'R')
    positions = ('F', 'M', 'H')
    feet_links = tuple([
	'{}{}Tarsus5'.format(side, pos)
        for side in sides
        for pos in positions
    ])
    for feet in feet_links:
        model.links[link_index[feet]].collisions= [Collision.sphere(
            feet, 0.005, units)]
    '''

    ########## FIX ##########
    '''
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
            actuated_joints[j] = joint.replace('Coxa', 'CoxaFake2')
    for joint in model.joints:
        if joint.name not in actuated_joints:
            model.joints[joint_index[joint.name]].type = 'fixed'
        else:
            if 'LM' in joint.name or 'LH' in joint.name:
                model.joints[joint_index[joint.name]].axis.xyz=[-axis if axis == 1 else axis for axis in model.joints[joint_index[joint.name]].axis.xyz]
    '''

    ######## CHANGE AXES AND LIMITS DIRECTION FOR LEFT SIDE #########
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
            
    for joint in model.joints:
        joint_obj = model.joints[joint_index[joint.name]]
        joint_obj.axis.limits[0] = -np.pi
        joint_obj.axis.limits[1]= np.pi
        if 'LM' in joint.name or 'LH' in joint.name:# or 'LF' in joint.name or 'RF' in joint.name:
            ljoint = model.joints[joint_index[joint.name]]
            ljoint.axis.xyz=[-axis if axis == 1 else axis for axis in ljoint.axis.xyz]
        


    ########## PLANAR ##########    
    add_planar_constraint(model, units)
    link_index = utils.link_name_to_index(model)
    joint_index = utils.joint_name_to_index(model)

    ########## WRITE ##########    
    model.units = units
    model.write(filename="{}.sdf".format(MODEL_NAME))
        
if __name__ == '__main__':
    main()
