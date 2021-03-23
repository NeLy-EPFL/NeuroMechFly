from farms_blender.core import pose
from farms_data.io import yaml

joint_poses = {'joints' : pose.get_current_pose("neuromechfly_noLimits")}

yaml.write_yaml(joint_poses, "../../config/test_pose.yaml")
