"""Generate Drosophila Template"""

import os
import numpy as np
from NeuroMechFly.sdf.sdf import ModelSDF, Link, Joint, Inertial, Collision
from NeuroMechFly.sdf.units import SimulationUnitScaling
from NeuroMechFly.sdf import utils
import yaml

MODEL_NAME = "neuromechfly_noLimits"

def add_planar_constraint(model, units, joint_type='prismatic'):
    """ Add planar constraints to the model.

    Parameters
    ----------
    model : <obj>
        SDF model
    units : <SimulationUnitScaling>
        Units of SDF model.
    joint_type : str, optional
        Joint type connecting the links, by default 'prismatic'
    """ 
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
        xyz=[1.0, 0.0, 0.0],
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
        xyz=[0.0, 1.0, 0.0],
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
        meters=1e3,
        seconds=1e0,
        kilograms=1e3
    )

    #: Read the sdf model from template
    model = ModelSDF.read(
        "{}.sdf".format(MODEL_NAME))[0]

    link_index = utils.link_name_to_index(model)
    joint_index = utils.joint_name_to_index(model)

    ########## CHANGE PARENTS AND SELECTED ROTATION AXES###########
    leg_joints = ('Cox','Fem','Tib','Tar')

    axes_to_change = []
    
    for joint in model.joints:
        name_split = joint.name.split('_')
        if name_split[1][2:5] in leg_joints:
            p = name_split[1][1]
            j = name_split[1][2:]
            if p == 'F':
                if j=='Femur' and len(name_split) == 2:
                    joint.parent = name_split[1][:2]+'Coxa'
                elif j=='Femur' and len(name_split) > 2:
                    joint.parent = name_split[1][:2]+'Femur'
                elif j=='Tibia':
                    joint.parent = name_split[1][:2]+'Femur_roll'
            else:
                if j=='Coxa' and len(name_split) == 2:
                    joint.parent = 'Thorax'
                elif j=='Coxa' and name_split[2]=='roll':
                    joint.parent = name_split[1][:2]+'Coxa_yaw'
                elif j=='Coxa' and name_split[2]=='yaw':
                    joint.parent = name_split[1][:2]+'Coxa'
                elif j=='Femur' and len(name_split) == 2:
                    joint.parent = name_split[1][:2]+'Coxa_roll'
                elif j=='Femur' and len(name_split) > 2:
                    joint.parent = name_split[1][:2]+'Femur'
                elif j=='Tibia':
                    joint.parent = name_split[1][:2]+'Femur_roll'
                    
        if joint.name in axes_to_change:
            joint.axis.xyz=[-axis if axis == 1 else axis for axis in joint.axis.xyz]
    

    ########## PLANAR ##########    
    add_planar_constraint(model, units)
    link_index = utils.link_name_to_index(model)
    joint_index = utils.joint_name_to_index(model)

    ########## WRITE ##########    
    model.write(filename="{}.sdf".format(MODEL_NAME))
        
if __name__ == '__main__':
    main()
