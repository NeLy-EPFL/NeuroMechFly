""" Drosophila simulation class for kinematic replay without body support. """
from NeuroMechFly.simulation.bullet_simulation import BulletSimulation
from NeuroMechFly.container import Container
from NeuroMechFly.sdf.units import SimulationUnitScaling
from random import random
import numpy as np
import pandas as pd
import pybullet as p

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
    container: <obj>
        Instance of the Container class.
    sim_options: <dict>
        Dictionary containing the simulation options.
    Kp: <float>
        Proportional gain of the position controller.  
    Kv: <float>
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
            self, container, sim_options, Kp, Kv, 
            position_path, velocity_path,
            add_perturbation,
            units=SimulationUnitScaling(meters=1000, kilograms=1000)
    ):    
        super().__init__(container, units, **sim_options)
        self.kp = Kp
        self.kv = Kv
        self.pose = [0] * self.num_joints
        self.vel = [0] * self.num_joints
        self.angles = self.load_angles(position_path)
        self.velocities = self.load_angles(velocity_path)
        self.impulse_sign = 1
        self.add_perturbation = add_perturbation

    def load_angles(self, data_path):
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
        t : int
            Time running in the physics engine. 
        """
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

        #: Setting the fixed joint angles, can be altered to change the appearance of the fly
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

        #: Leg joint indices
        joint_control_middle = list(
            np.arange(42, 49)) + list(np.arange(81, 88))
        joint_control_front = list(np.arange(17, 23)) + list(np.arange(56, 63))
        joint_control_hind = list(np.arange(28, 35)) + list(np.arange(67, 74))
        joint_control = joint_control_hind + joint_control_middle + joint_control_front
        
        #: Control the joints through position controller
        #: Velocity can be discarded if not available and gains can be changed
        for joint in range(self.num_joints):
            if joint in joint_control:
                p.setJointMotorControl2(
                    self.animal, joint,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=self.pose[joint],
                    targetVelocity=self.vel[joint],
                    positionGain=self.kp,
                    velocityGain=self.kv,
                )
            else:
                p.setJointMotorControl2(
                    self.animal, joint,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=self.pose[joint],
                )

    def feedback_to_controller(self):
        """
        Code that glues the sensors/feedback to controller in the system.
        """

    def update_parameters(self, params):
        """ Update parameters. """

    def optimization_check(self):
        """ Optimization check. """
