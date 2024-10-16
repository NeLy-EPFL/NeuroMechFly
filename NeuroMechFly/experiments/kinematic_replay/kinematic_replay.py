""" Drosophila simulation class for kinematic replay for the ball experiments. """

import numpy as np
import pandas as pd
import pybullet as p

from shapely.geometry import LinearRing, Point, Polygon
from NeuroMechFly.sdf.units import SimulationUnitScaling
from NeuroMechFly.simulation.bullet_simulation import BulletSimulation
from NeuroMechFly.experiments.network_optimization.neuromuscular_control import DrosophilaSimulation as ds


class DrosophilaSimulation(BulletSimulation):
    """ Drosophila Simulation Class for kinematic replay.

    Parameters
    ----------
    container: <Container>
        Instance of the Container class.
    sim_options: <dict>
        Dictionary containing the simulation options.
    kp: <float>
        Proportional gain of the position controller.
    kv: <float>
        Derivative gain of the position controller.
    angles_path: <str>
        Path of the joint position .pkl file.
    velocity_path: <str>
        Path of the joint velocity .pkl file.
    fixed_positions: <dict>
        Dictionary containing the positions for the fixed joints that should be different from the zero pose.
    units: <obj>
        Instance of SimulationUnitScaling object to scale up the units during calculations.
    """

    def __init__(
        self,
        container,
        sim_options,
        kp,
        kv,
        angles_path,
        velocity_path,
        starting_time=0.0,
        fixed_positions=None,
        units=SimulationUnitScaling(
            meters=1000,
            kilograms=1000)):

        self.last_draw = []
        self.grf = []
        self.kp = kp
        self.kv = kv
        self.angles_path = angles_path
        self.fixed_positions = fixed_positions

        super().__init__(container, units, **sim_options)

        self.pose = [0] * self.num_joints
        self.vel = [0] * self.num_joints
        self.angles = self.load_data(angles_path, starting_time)
        self.velocities = self.load_data(velocity_path, starting_time)

        # Debug parameter
        self.draw_ss_line_ids = [
            p.addUserDebugLine(
                (0., 0., 0.), (0., 0., 0.), lineColorRGB=[1, 0, 0]
            )
            for j in range(6)
        ]
        self.draw_com_line_vert_id = p.addUserDebugLine(
            (0., 0., 0.), (0., 0., 0.), lineColorRGB=[1, 0, 0]
        )
        self.draw_com_line_horz_id = p.addUserDebugLine(
            (0., 0., 0.), (0., 0., 0.), lineColorRGB=[1, 0, 0]
        )

    def load_data(self, data_path, starting_time):
        """ Function that loads the pickle format joint angle or velocity gile.

        Parameters
        ----------
        data_path : <str>
            Path of the .pkl file.

        starting_time : <float>
            Experiment's time from which the simulation will start.

        Returns
        -------
        dict
            Returns the joint angles in a dictionary.
        """
        names_equivalence = {
            'ThC_pitch': 'Coxa',
            'ThC_yaw': 'Coxa_yaw',
            'ThC_roll': 'Coxa_roll',
            'CTr_pitch': 'Femur',
            'CTr_roll': 'Femur_roll',
            'FTi_pitch': 'Tibia',
            'TiTa_pitch': 'Tarsus1'
        }
        converted_dict = {}
        try:
            data = pd.read_pickle(data_path)
            start = int(np.round(starting_time / self.time_step))
            for leg, joints in data.items():
                for joint_name, val in joints.items():
                    new_name = 'joint_' + leg[:2] + \
                        names_equivalence[joint_name]
                    converted_dict[new_name] = val[start:]
            return converted_dict
        except BaseException:
            FileNotFoundError(f"File {data_path} not found!")

    def load_ball_info(self):
        to_replace = self.angles_path[self.angles_path.find(
            'joint_angles'):self.angles_path.find('__')]
        data_path = self.angles_path.replace(to_replace, 'treadmill_info')

        try:
            data = pd.read_pickle(data_path)
            ball_rad = data['radius']
            ball_pos = data['position']

            return ball_rad, ball_pos

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
        # Setting the joint angular positions of the fixed joints
        if not self.fixed_positions:
            self.fixed_positions = {
                'joint_LAntenna': 35,
                'joint_RAntenna': -35,
            }

        # Setting the joint angular positions of the fixed joints
        for joint_name, joint_pos in self.fixed_positions.items():
            self.pose[self.joint_id[joint_name]] = np.deg2rad(joint_pos)

        # Setting the joint angular positions of leg DOFs based on pose estimation
        for joint_name, joint_pos in self.angles.items():
            self.pose[self.joint_id[joint_name]] = joint_pos[t]

        # Setting the joint angular velocities of leg DOFs based on pose estimation
        for joint_name, joint_vel in self.velocities.items():
            self.vel[self.joint_id[joint_name]] = joint_vel[t]

        # Control the joints through position controller
        # Velocity can be discarded if not available and gains can be changed
        for joint in range(self.num_joints):
            p.setJointMotorControl2(
                self.animal, joint,
                controlMode=p.POSITION_CONTROL,
                targetPosition=self.pose[joint],
                targetVelocity=self.vel[joint],
                positionGain=self.kp,
                velocityGain=self.kv,
                maxVelocity=1e8
            )
            p.changeDynamics(self.animal, joint, maxJointVelocity=1e8)

        # Change the color of the colliding body segments
        if self.draw_collisions:
            draw = []
            if self.behavior == 'walking':
                links_contact = self.get_current_contacts()
                link_names = list(self.link_id.keys())
                link_ids = list(self.link_id.values())
                for i in links_contact:
                    link1 = link_names[link_ids.index(i)]
                    if link1 not in draw:
                        draw.append(link1)
                        self.change_color(link1, self.color_collision)
                for link in self.last_draw:
                    if link not in draw:
                        self.change_color(link, self.color_legs)

            elif self.behavior == 'grooming':
                # Don't consider the ground sensors
                collision_forces = self.contact_normal_force[len(
                    self.ground_contacts):, :]
                links_contact = np.where(
                    np.linalg.norm(collision_forces, axis=1) > 0
                )[0]
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

    def change_color(self, identity, color):
        """ Change color of a given body segment. """
        p.changeVisualShape(
            self.animal,
            self.link_id[identity],
            rgbaColor=color)

    def feedback_to_controller(self):
        """
        Code that glues the sensors/feedback to controller in the system.
        """

    def update_parameters(self, params):
        """ Update parameters. """

    def optimization_check(self):
        """ Optimization check. """
