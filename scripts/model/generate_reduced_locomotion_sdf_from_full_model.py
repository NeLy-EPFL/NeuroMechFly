""" Script to generate reduce fly sdf model from full fly model. """

import os
from pathlib import Path

import bpy
from farms_blender.core.objects import objs_of_farms_types, remove_objects
from farms_blender.core.pose import set_model_pose
from farms_blender.core.sdf import export_sdf, load_sdf
from farms_data.io.yaml import read_yaml
from farms_sdf import utils
from farms_sdf.sdf import Link, ModelSDF

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

# Fix contralateral side joint axis and limits
for joint, _ in objs_of_farms_types(joint=True):
    joint_side = joint.name[6]
    if (joint_side == "R") and ("roll" in joint.name) or ("yaw" in joint.name):
        # Get left joint
        limits = (
            bpy.data.objects[
                joint["farms_name"].replace("joint_R", "joint_L")
            ]["farms_joint_limits"]
        )
        # Invert limits for right joint
        joint["farms_joint_limits"][0] = limits[0]
        joint["farms_joint_limits"][1] = limits[1]
        # Invert axis direction for right joint
        joint["farms_axis_xyz"][0] *= -1
        joint["farms_axis_xyz"][1] *= -1
        joint["farms_axis_xyz"][2] *= -1

# Update joint limits
# 'joint_LFFemur': {'min': -2.6662613939920927, 'max': -1.2265979890155232},
# 'joint_LFTibia': {'min': 0.755579943967942, 'max': 2.537248153705797},
# 'joint_LMCoxa_roll': {'min': 1.7633980792760797, 'max': 2.3280868205896708},
# 'joint_LMFemur': {'min': -2.363638739290062, 'max': -1.6424144373215874},
# 'joint_LMTibia': {'min': 1.5013089603392067, 'max': 2.612560417725856},
# 'joint_LHCoxa_roll': {'min': 2.342643179253425, 'max': 2.7964939180771236},
# 'joint_LHFemur': {'min': -2.4611009381052775, 'max': -1.2565385270550669},
# 'joint_LHTibia': {'min': 0.9004679093523356, 'max': 2.610439744875883},
# 'joint_LFCoxa': {'min': -0.4122592952311641, 'max': 0.5658698497740794}
angle_limits = {
    'FFemur': {'min': -2.7070260597076814, 'max': -1.1927677266616241},
    'FTibia': {'min': 0.7795120577106078, 'max': 2.5142727572427592},
    'MCoxa_roll': {'min': 1.7633980792760797, 'max': 2.3280868205896708},
    'MFemur': {'min': -2.4202848346149346, 'max': -1.7940671597190856},
    'MTibia': {'min': 1.5742404712255267, 'max': 2.4686521684717793},
    'HCoxa_roll': {'min': 2.342643179253425, 'max': 2.7964939180771236},
    'HFemur': {'min': -2.4822204407825286, 'max': -1.1429577335102055},
    'HTibia': {'min': 0.6137228182602668, 'max': 2.6959491072995783},
    'FCoxa': {'min': -0.3230026538529655, 'max': 0.5548562980669347},
}

for joint_name, limit in angle_limits.items():
    for side in ('L', 'R'):
        joint_obj = bpy.data.objects[f"joint_{side}{joint_name}"]
        joint_obj["farms_joint_limits"][0] = limit["min"]
        joint_obj["farms_joint_limits"][1] = limit["max"]

# Export the model
export_sdf("neuromechfly_noLimits", SDF_MODEL_PATH.joinpath("neuromechfly_locomotion_optimization.sdf"))
