"""Generate Drosophila Template"""

import os
import numpy as np
from farms_sdf.sdf import ModelSDF, Link, Joint
from farms_sdf.units import SimulationUnitScaling
import yaml

MESH_PATH = os.path.join("../meshes/stl", "")
MODEL_NAME = "drosophila_100x"

with open('../../config/gen_sdf_100x.yaml') as f:
	data = yaml.safe_load(f)

def main():
    """Main"""
    name_file = "drosophila_100x"
    #joint_limits = [-np.pi, np.pi, 1e10, 2*np.pi*100]
    scale = 1
    units = SimulationUnitScaling()
    links = {}
    for name, elem in data.items():
        if '_' in name:
            links[name] = Link.fake_joint(name=name,
                               pose=np.concatenate([
                                   elem["link_location"],
                                   np.zeros(3)
                               ]),
                               units=units,
                               radius = 0.001,         
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
    
    joints = {"joint_{}".format(name): 
        Joint(
            name="joint_{}".format(name),
            joint_type="revolute",
            parent=links[elem['parent_link']],
            child=links[name],
            xyz=elem['rot_axis'],
            limits=[elem['lower_limit'], elem['upper_limit'], elem['max_effort'], elem['max_vel']]
            #limits=[elem['lower_limit'], elem['upper_limit'], elem['max_effort'], elem['max_vel']]
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
