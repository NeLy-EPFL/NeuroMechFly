""" Drosophila simulation class for kinematic replay without body support. """

import numpy as np
import pandas as pd

import pybullet as p
from NeuroMechFly.sdf.units import SimulationUnitScaling
from NeuroMechFly.simulation.bullet_simulation import BulletSimulation

# Random number seed
np.random.seed(seed=321)


def add_perturbation(
        size, initial_position, target_position, time, units
):
    """ Shoot a ball to perturb the target system at a specified
    velocity

    Parameters
    ----------
    size: <float>
        Radius of the ball
    initial_position: <array>
        3D position of the ball
    target_position: <array>
        3D position of the target
    target_velocity: <float>
        Final velocity during impact

    Returns
    -------
    ball : <int>
    Pybullet ID for the ball

    """
    # Init
    initial_position = np.asarray(initial_position) * units.meters
    target_position = np.asarray(target_position) * units.meters
    # Load ball
    ball = p.loadURDF(
        "../data/design/sdf/sphere_1cm.urdf", initial_position,
        globalScaling=size * units.meters,
        useMaximalCoordinates=True
    )
    # Change dynamics to remove damping and friction
    p.changeDynamics(
        ball, -1, linearDamping=0, angularDamping=0,
        rollingFriction=0, spinningFriction=0
    )
    p.changeVisualShape(ball, -1, rgbaColor=[0.8, 0.8, 0.8, 1])
    # Compute initial velocity
    velocity = (
        target_position - initial_position -
        0.5 * np.asarray([0, 0, -9.81 * units.gravity]) * time**2
    ) / time
    # Reset base velocity
    p.resetBaseVelocity(ball, velocity)
    return ball


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
    position_path: <str>
        Path of the joint position .pkl file.
    velocity_path: <str>
        Path of the joint velocity .pkl file.
    add_perturbation: <bool>
        Activate/deactivate the ball perturbation.
    units: <obj>
        Instance of SimulationUnitScaling object to scale up the units during calculations.
    """

    def __init__(
            self, container, sim_options, kp, kv,
            angles_path, velocity_path,
            add_perturbation,
            starting_time = 0.0,
            fixed_positions = None,
            units=SimulationUnitScaling(meters=1000, kilograms=1000)
    ):
        super().__init__(container, units, **sim_options)
        self.last_draw = []
        self.kp = kp
        self.kv = kv
        self.pose = [0] * self.num_joints
        self.vel = [0] * self.num_joints
        self.angles = self.load_data(angles_path,starting_time)
        self.velocities = self.load_data(velocity_path,starting_time)
        self.impulse_sign = 1
        self.add_perturbation = add_perturbation
        self.pball = None
        self.fixed_positions = fixed_positions

    def load_data(self, data_path, starting_time):
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
            start = int(np.round(starting_time/self.time_step))
            for leg, joints in data.items():
                for joint_name, val in joints.items():
                    new_name = 'joint_' + leg[:2] + \
                        names_equivalence[joint_name]
                    converted_dict[new_name] = val[start:]
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
        t : int
            Time running in the physics engine.
        """
        # Throw mini balls at the fly during kinematic replay
        if self.add_perturbation:
            if ((t + 1) % 500) == 0:
                print("Adding perturbation")
                self.pball = add_perturbation(
                    size=5e-2,
                    initial_position=np.asarray(
                        [0, self.impulse_sign * 2e-3, 0.0]) + self.base_position,
                    target_position=self.base_position,
                    time=20e-3, units=self.units
                )
                self.impulse_sign *= -1

            if ((t + 1) % 3000) == 0 and t < 3012:
                radius = 10e-2
                self.pball = add_perturbation(
                    size=radius,
                    initial_position=np.asarray(
                        [radius * 0.05, radius * 0.05, 1e-3]) + self.base_position,
                    target_position=[self.base_position[0], self.base_position[1], 0.0],
                    time=20e-3, units=self.units
                )
                p.changeDynamics(self.pball, -1, 0.3)

        for joint_name, joint_pos in self.fixed_positions.items():
            self.pose[self.joint_id[joint_name]] = np.deg2rad(joint_pos)

        # Setting the joint angular positions of leg DOFs based on pose
        # estimation
        for joint_name, joint_pos in self.angles.items():
            self.pose[self.joint_id[joint_name]] = joint_pos[t]

        # Setting the joint angular velocities of leg DOFs based on pose
        # estimation
        for joint_name, joint_vel in self.velocities.items():
            self.vel[self.joint_id[joint_name]] = joint_vel[t]


        # Reset joint states to prevent explosion at high gains
        if t==0:
            for joint in range(self.num_joints):
                p.resetJointState(
                    self.animal, joint,
                    targetValue=self.pose[joint],
                    targetVelocity=self.vel[joint]
                )

        # Control the joints through position controller
        # Velocity can be discarded if not available and gains can be changed
        for joint in range(self.num_joints):
            p.setJointMotorControl2(
                self.animal, joint,
                controlMode=p.POSITION_CONTROL,
                targetPosition=self.pose[joint],
                targetVelocity=self.vel[joint],
                positionGain= self.kp,
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
                # Don't consider the ground sensors
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
