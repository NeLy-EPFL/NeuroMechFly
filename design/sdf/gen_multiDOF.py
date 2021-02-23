"""Generate Drosophila Template"""

import os
import numpy as np
from NeuroMechFly.sdf.sdf import ModelSDF, Link, Joint
from NeuroMechFly.sdf.units import SimulationUnitScaling
import yaml

MESH_PATH = os.path.join("../meshes/stl", "")
#MODEL_NAME = "neuromechfly_noLimits"

with open('../../config/gen_sdf_1x.yaml') as f:
	data = yaml.safe_load(f)

def main():
    """Main"""
    name_file = "neuromechfly_noLimits"
    #joint_limits = [-np.pi, np.pi, 1e10, 2*np.pi*100]
    scale = 1
    units = SimulationUnitScaling(meters=1000, kilograms=1000)
    links = {}
    for name, elem in data.items():
        if '_' in name:
            links[name] = Link.empty(name=name,
                               pose=np.concatenate([
                                   elem["link_location"],
                                   np.zeros(3)
                               ]),
                               units=units         
                          )
        else:
            links[name] = Link.from_mesh(name=name,
                               mesh=os.path.join(MESH_PATH, name+".stl", ),
                               pose=np.concatenate([
                                   elem["link_location"],
                                   np.zeros(3)
                               ]),
                               shape_pose=np.zeros(6),
                               scale=scale,
                               units=units,
                               inertial_from_bounding=False,
                               # color=[0.1, 0.7, 0.1, 1]
                          )


###################### CHANGE DENSITY #################################################
    abdomen = ['A1A2','A3','A4','A5','A6']
    thorax = ['Thorax','LHaltere','RHaltere']
    head = ['Head','RAntenna','LAntenna','LEye','REye','Rostrum','Haustellum']

    abdomen_mass = 45e-8 #0.45mg
    thorax_mass = 31e-8 #0.31mg
    head_mass = 12.5e-8 #0.125mg
    wings_mass = 0.5e-8 #0.05mg
    legs_mass = 11e-8 #0.11mg

    abdomen_vol = 0
    thorax_vol = 0
    head_vol = 0
    wings_vol = 0
    legs_vol = 0
    
    for name, link in links.items():
        if name in abdomen:
            abdomen_vol += link.inertial.volume
        elif name in thorax:
            thorax_vol += link.inertial.volume
        elif name in head:
            head_vol += link.inertial.volume
        elif 'Wing' in name:
            wings_vol += link.inertial.volume
        else:
            legs_vol += link.inertial.volume

    for name, elem in data.items():
        if '_' not in name:
            if name in abdomen:
                density = abdomen_mass/abdomen_vol
            elif name in thorax:
                density = thorax_mass/thorax_vol
            elif name in head:
                density = head_mass/head_vol
            elif 'Wing' in name:
                density = wings_mass/wings_vol
            else:
                density = legs_mass/legs_vol
                
            links[name] = Link.from_mesh(name=name,
                               mesh=os.path.join(MESH_PATH, name+".stl", ),
                               pose=np.concatenate([
                                   elem["link_location"],
                                   np.zeros(3)
                               ]),
                               shape_pose=np.zeros(6),
                               scale=scale,
                               units=units,
                               inertial_from_bounding=False,
                               density = density
                          )

    #t_mass=0
    #for name, link in links.items():
    #    t_mass+=link.inertial.mass
    #print(t_mass)

    ##############################################################################
    
    joints = {"joint_{}".format(name): 
        Joint(
            name="joint_{}".format(name),
            joint_type="revolute",
            parent=links[elem['parent_link']],
            child=links[name],
            xyz=elem['rot_axis'],
            limits=[elem['lower_limit'], elem['upper_limit'], elem['max_effort'], elem['max_vel']]
        ) for (name, elem) in data.items() if (
            elem['parent_link'].lower() != "none")
    }
    sdf = ModelSDF(
        name=name_file,
        pose=np.zeros(6),
        links=links.values(),
        joints=joints.values(),
        units=units
    )
    sdf.write(filename="{}.sdf".format(name_file))

if __name__ == '__main__':
    main()
