""" Module to setup constrained kinematic replay experiment """
import numpy as np
import pandas as pd

import pybullet as p
from NeuroMechFly.sdf.units import SimulationUnitScaling
from NeuroMechFly.simulation.bullet_simulation import BulletSimulation


class DrosophilaSimulation(BulletSimulation):
    """[summary]

    Parameters
    ----------
    BulletSimulation : [type]
        [description]
    """

    def __init__(
        self,
        container,
        sim_options,
        Kp,
        Kv,
        units=SimulationUnitScaling(
            meters=1000,
            kilograms=1000)):
        super().__init__(container, units, **sim_options)
        self.last_draw = []
        self.grf = []
        self.kp = Kp
        self.kv = Kv
        self.pose = [0] * self.num_joints
        self.vel = [0] * self.num_joints
        self.angles = self.load_angles(
            f'../data/joint_kinematics/{self.behavior}/{self.behavior}_converted_joint_angles.pkl')
        self.velocities = self.load_angles(
            f'../data/joint_kinematics/{self.behavior}/{self.behavior}_converted_joint_velocities.pkl')

    def load_angles(self, data_path):
        """[summary]

        Parameters
        ----------
        data_path : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        try:
            return pd.read_pickle(data_path)
        except BaseException:
            FileNotFoundError(f"File {data_path} not found!")

    def controller_to_actuator(self, t):
        """
        Code that glues the controller the actuator in the system.
        If there are muscles then contoller actuates the muscles.
        If not then the controller directly actuates the joints.

        Parameters
        ----------
        t : [type]
            [description]
        """

        #: Setting the fixed joint positions
        fixed_positions = {
            'joint_A3': -15,
            'joint_A4': -15,
            'joint_A5': -15,
            'joint_A6': -15,
            'joint_LAntenna': 33,
            'joint_RAntenna': -33,
            'joint_Rostrum': 90,
            'joint_Haustellum': -60,
            'joint_LWing_roll': 90,
            'joint_LWing_yaw': -17,
            'joint_RWing_roll': -90,
            'joint_RWing_yaw': 17,
            'joint_Head': 10
        }

        for joint_name, joint_pos in fixed_positions.items():
            self.pose[self.joint_id[joint_name]] = np.deg2rad(joint_pos)

        #: Setting the joint positions of leg DOFs based on pose estimation
        for joint_name, joint_pos in self.angles.items():
            self.pose[self.joint_id[joint_name]] = joint_pos[t]

        #: Setting the joint velocities of leg DOFs based on pose estimation
        for joint_name, joint_vel in self.velocities.items():
            self.vel[self.joint_id[joint_name]] = joint_vel[t]

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
                    link1 = self. GROUND_CONTACTS[i][:-1]
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
