""" Drosophila simulation class for kinematic replay for the ball experiments. """

from NeuroMechFly.simulation.bullet_simulation import BulletSimulation
from NeuroMechFly.sdf.units import SimulationUnitScaling
import pybullet as p
import numpy as np
import pandas as pd

class DrosophilaSimulation(BulletSimulation):
    """ Drosophila Simulation Class for kinematic replay.

    Parameters
    ----------
    container: <Container>
        Instance of the Container class.
    sim_options: <dict>
        Dictionary containing the simulation options.
    fixed_positions: <dict>
        Dictionary containing the positions for the fixed joints that should be different from the zero pose.
    Kp: <float>
        Proportional gain of the position controller.
    Kv: <float>
        Derivative gain of the position controller.
    angles_path: <str>
        Path of the joint position .pkl file.
    velocity_path: <str>
        Path of the joint velocity .pkl file.
    units: <obj>
        Instance of SimulationUnitScaling object to scale up the units during calculations.
    """
    def __init__(
        self,
        container,
        sim_options,
        fixed_positions,
        Kp,
        Kv,
        angles_path,
        velocity_path,
        units=SimulationUnitScaling(
            meters=1000,
            kilograms=1000)):

        super().__init__(container, units, **sim_options)
        self.last_draw = []
        self.grf = []
        self.fixed_positions = fixed_positions
        self.kp = Kp
        self.kv = Kv
        self.pose = [0] * self.num_joints
        self.vel = [0] * self.num_joints
        self.angles = self.load_data(angles_path)
        self.velocities = self.load_data(velocity_path)      

    def load_data(self, data_path):
        """ Function that loads the pickle format joint angle or velocity gile.

        Parameters
        ----------
        data_path : <str>
            Path of the .pkl file.

        Returns
        -------
        dict
            Returns the joint angles in a dictionary.
        """
        names_equivalence = {
            'ThC_pitch':'Coxa',
            'ThC_yaw':'Coxa_yaw',
            'ThC_roll':'Coxa_roll',
            'CTr_pitch':'Femur',
            'CTr_roll':'Femur_roll',
            'FTi_pitch':'Tibia',
            'TiTa_pitch':'Tarsus1'
            }
        converted_dict = {}
        try:
            data = pd.read_pickle(data_path)
            for leg, joints in data.items():
                for joint_name, val in joints.items():
                    new_name = 'joint_'+ leg[:2] + names_equivalence[joint_name]
                    converted_dict[new_name] = val
            return converted_dict
        except BaseException:
            FileNotFoundError(f"File {data_path} not found!")

    def controller_to_actuator(self, t):
        """
        Code that glues the controller the actuator in the system.
        If there are muscles then contoller actuates the muscles.
        If not then the controller directly actuates the joints.

        Parameters
        ----------
        t : <int>
            Time running in the physics engine.
        """
        #: Setting the joint angular positions of the fixed joints
        for joint_name, joint_pos in self.fixed_positions.items():
            self.pose[self.joint_id[joint_name]] = np.deg2rad(joint_pos)

        #: Setting the joint angular positions of leg DOFs based on pose estimation
        for joint_name, joint_pos in self.angles.items():
            self.pose[self.joint_id[joint_name]] = joint_pos[t]

        #: Setting the joint angular velocities of leg DOFs based on pose estimation
        for joint_name, joint_vel in self.velocities.items():
            self.vel[self.joint_id[joint_name]] = joint_vel[t]

        #: Control the joints through position controller
        #: Velocity can be discarded if not available and gains can be changed
        for joint in range(self.num_joints):
            p.setJointMotorControl2(
                self.animal, joint,
                controlMode=p.POSITION_CONTROL,
                targetPosition=self.pose[joint],
                targetVelocity=self.vel[joint],
                positionGain=self.kp,
                velocityGain=self.kv,
            )

        #: Change the color of the colliding body segments
        if self.draw_collisions:
            draw = []
            if self.behavior == 'walking':
                links_contact = np.where(
                    np.linalg.norm(
                        self.ground_reaction_forces,
                        axis=1) > 0)[0]
                for i in links_contact:
                    link1 = self. ground_contacts[i][:-1]
                    if link1 not in draw:
                        draw.append(link1)
                        self.change_color(link1 + '5', self.color_collision)
                for link in self.last_draw:
                    if link not in draw:
                        self.change_color(link + '5', self.color_legs)

            elif self.behavior == 'grooming':
                links_contact = np.where(
                    np.linalg.norm(
                        self.collision_forces,
                        axis=1) > 0)[0]
                for i in links_contact:
                    link1 = self.self_collisions[i][0]
                    link2 = self.self_collisions[i][1]
                    if link1 not in draw:
                        draw.append(link1)
                        self.change_color(link1, self.color_collision)
                    if link2 not in draw:
                        draw.append(link2)
                        self.change_color(link2, self.color_collision)
                for link in self.last_draw:
                    if link not in draw:
                        if 'Antenna' in link:
                            self.change_color(link, self.color_body)
                        else:
                            self.change_color(link, self.color_legs)
            self.last_draw = draw

    def change_color(self, id, color):
        """ Change color of a given body segment. """
        p.changeVisualShape(self.animal, self.link_id[id], rgbaColor=color)

    def feedback_to_controller(self):
        """
        Code that glues the sensors/feedback to controller in the system.
        """

    def update_parameters(self, params):
        """ Update parameters. """

    def optimization_check(self):
        """ Optimization check. """
