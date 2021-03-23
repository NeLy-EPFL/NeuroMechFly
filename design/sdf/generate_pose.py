""" Fly locomotion generation. """

import os
import pathlib
import sys

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from farms_models.utils import get_sdf_path
import yaml

import bpy
import mathutils
import mouse_scene
from bpy_extras.object_utils import world_to_camera_view
from farms_blender.core.display import display_farms_types
from farms_blender.core.objects import objs_of_farms_types
from farms_blender.core.pose import set_model_pose
from farms_blender.core.sdf import get_base_link, load_sdf

script_path = os.path.abspath(bpy.data.filepath)
sys.path.append(script_path)

def main():
    """ main """
    #: Load model
    model_name, *_, model_objs = load_sdf(
        "./neuromechfly_noLimits.sdf",
        resources_scale=1e-2
    )

    #: Display
    display = {
        'view': True,
        'render': True,
        'link': False,
        'visual': True,
        'collision': False,
        'inertial': False,
        'com': False,
        'muscle': False,
        'joint': False,
        'joint_axis': False,
    }
    display_farms_types(objs=model_objs, **display)
    #: Set pose
    with open('../../config/pose_tripod.yaml', 'r') as stream:
        model_pose =  yaml.load(stream, yaml.FullLoader)

    set_model_pose(model_pose['joints'], units='degrees')

if __name__ == '__main__':
    main()
