""" Script to generate reduce fly sdf model from full fly model. """

import os
from pathlib import Path

from farms_blender.core.objects import objs_of_farms_types, remove_objects
from farms_blender.core.pose import set_model_pose
from farms_blender.core.sdf import export_sdf, load_sdf
from farms_data.io.yaml import read_yaml
from farms_sdf.sdf import ModelSDF, Link
from farms_sdf import utils

# Global config paths
SCRIPT_PATH = Path(__file__).parent.absolute()
DATA_PATH = SCRIPT_PATH.joinpath("..", "..", "data")
CONFIG_PATH = DATA_PATH.joinpath("config")
POSE_CONFIG_PATH = CONFIG_PATH.joinpath("pose")
SDF_MODEL_PATH =  DATA_PATH.joinpath("design", "sdf")

# Load the original full mouse model
model = load_sdf(SDF_MODEL_PATH.joinpath("neuromechfly_limitsFromData.sdf"))

# Set the pose of the spine and tail
default_pose = read_yaml(POSE_CONFIG_PATH.joinpath("fixed_joints.yaml"))
set_model_pose(default_pose["joints"], units="degrees")

# Fix joints to reduce model complexity
for joint, _ in objs_of_farms_types(joint=True):
    if any((name == joint["farms_name"] for name in default_pose["joints"])):
        joint["farms_joint_type"] = "fixed"

# Change joints
for joint, _ in objs_of_farms_types(joint=True):
    if joint["farms_joint_type"] == "revolute":
        joint["farms_joint_type"] = "continuous"

# Remove collision shapes from the body
remove_collision_names = (
    "Wing",
    "A1A2", "A3", "A4", "A5", "A6",
    "Thorax", "Haltere",
    "Head", "Eye", "Antenna", "Rostrum", "Haustellum",
    "Coxa", "Femur", "Tibia"
)

check_names = lambda obj, names : any((name in obj.name for name in names))

remove_collision_objs = [
    obj
    for obj, _ in objs_of_farms_types(collision=True)
    if check_names(obj, remove_collision_names)
]

remove_objects(remove_collision_objs)

# Fix contralateral side joint axis
for joint, _ in objs_of_farms_types(joint=True):
    joint_side = joint.name[6]
    if (joint_side == "R") and ("roll" in joint.name) or ("yaw" in joint.name):
        joint["farms_axis_xyz"][0] *= -1
        joint["farms_axis_xyz"][1] *= -1
        joint["farms_axis_xyz"][2] *= -1

# Export the model
export_sdf("neuromechfly_noLimits", SDF_MODEL_PATH.joinpath("neuromechfly_locomotion_optimization.sdf"))
