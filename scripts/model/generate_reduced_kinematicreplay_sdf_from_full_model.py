""" Script to generate reduce fly sdf model from full fly model. """

import os
import yaml
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
model = load_sdf(os.path.join(SDF_MODEL_PATH,"neuromechfly_noLimits.sdf"))

# Set the pose of the spine and tail
with open(os.path.join(POSE_CONFIG_PATH, "fixed_joints_kinematic_replay.yaml"), 'r') as stream:
    model_pose =  yaml.load(stream, yaml.FullLoader)

set_model_pose(model_pose['joints'], units='degrees')

# Fix joints to reduce model complexity
for joint, _ in objs_of_farms_types(joint=True):
    if any((name == joint["farms_name"] for name in model_pose["joints"])):
        joint["farms_joint_type"] = "fixed"

# Remove collision shapes from the body
remove_collision_names = (
    "Wing",
    "A1A2", "A3", "A4", "A5", "A6",
    "Thorax", "Haltere",
)

remove_collision_objs = [
    obj
    for obj, _ in objs_of_farms_types(collision=True)
    if any((name in obj.name for name in remove_collision_names))
]

remove_objects(remove_collision_objs)

# Export the model
export_sdf("neuromechfly_noLimits", os.path.join(SDF_MODEL_PATH, "neuromechfly_kinematic_replay.sdf"))
