import math
import os
import pickle
import sys
import time
from random import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython import embed

import pybullet as p
from bullet_simulation_KM_noSelfCollisions_SA import BulletSimulation
from NeuroMechFly.container import Container
from NeuroMechFly.sdf.units import SimulationUnitScaling


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
    out :

    """
    # Init
    initial_position = np.asarray(initial_position)*units.meters
    target_position = np.asarray(target_position)*units.meters
    # Load ball
    ball = p.loadURDF(
        "../../design/sdf/sphere_1cm.urdf", initial_position,
        globalScaling=size*units.meters
    )
    p.changeDynamics(
        ball, -1, linearDamping=0, angularDamping=0,
        rollingFriction=1e-5, spinningFriction=1e-4
    )
    p.changeVisualShape(ball, -1, rgbaColor=[0.8, 0.8, 0.8, 1])
    # Compute distance
    dist = np.linalg.norm(initial_position -target_position)
    # Compute direction vector
    dir_vector = (target_position - initial_position)/dist
    # Compute initial velocity
    velocity = (
        target_position - initial_position -
        0.5*np.asarray([0, 0, -9.81*units.gravity])*time**2
    )/time
    p.resetBaseVelocity(ball, velocity)
    return ball


class DrosophilaSimulation(BulletSimulation):

    def __init__(
            self, container, sim_options, Kp, Kv,
            units=SimulationUnitScaling(meters=1000, kilograms=1000)
    ):
        super().__init__(container, units, **sim_options)
        # Transferred to Container, delete later
        self.torques = []
        self.grf = []
        self.collision_forces = []
        self.ball_rot = []
        self.lateral_force1 = []
        self.lateral_force2 = []
        self.kpId = p.addUserDebugParameter("Kp", 0, 1, 0)
        self.kvId = p.addUserDebugParameter("Kv", 0, 1, 0)
        self.kp = Kp
        self.kv = Kv
        self.claw_track = {'LF': list(), 'LM': list(
        ), 'LH': list(), 'RF': list(), 'RM': list(), 'RH': list()}
        self.angles = self.load_angles(
            './angles/walking_joint_angles_smoothed.pkl')
        self.velocities = self.load_angles(
            './angles/walking_joint_velocities.pkl')

    def load_angles(self, data_path):
        try:
            with open(data_path, 'rb') as f:
                return pickle.load(f)
        except:
            FileNotFoundError(f"File {data_path} not found!")

    def controller_to_actuator(self, t):
        """
        Code that glues the controller the actuator in the system.
        If there are muscles then contoller actuates the muscles.
        If not then the controller directly actuates the joints
        """

        if t == 10:
            print("Adding ball")
            add_perturbation(
                 size=5e-2,
                 initial_position=np.asarray([1e-2, 0, 1e-2]),
                 target_position=self.base_position,
                 time=0.1, units=self.units
                 )
        if t == 150:
            print(f"Adding ball {self.base_position}")
            add_perturbation(
                 size=5e-2,
                 initial_position=np.asarray([-1e-2, 0, 1e-2]),
                 target_position=self.base_position,
                 time=0.1, units=self.units
            )

        joints = [joint for joint in range(self.num_joints)]
        pose = [0]*self.num_joints
        vel = [0]*self.num_joints

        pose[self.joint_id['joint_A3']] = np.deg2rad(-15)
        pose[self.joint_id['joint_A4']] = np.deg2rad(-15)
        pose[self.joint_id['joint_A5']] = np.deg2rad(-15)
        pose[self.joint_id['joint_A6']] = np.deg2rad(-15)

        pose[self.joint_id['joint_LAntenna']] = np.deg2rad(35)
        pose[self.joint_id['joint_RAntenna']] = np.deg2rad(-35)

        pose[self.joint_id['joint_Rostrum']] = np.deg2rad(90)
        pose[self.joint_id['joint_Haustellum']] = np.deg2rad(-60)

        pose[self.joint_id['joint_LWing_roll']] = np.deg2rad(90)
        pose[self.joint_id['joint_LWing_yaw']] = np.deg2rad(-17)
        pose[self.joint_id['joint_RWing_roll']] = np.deg2rad(-90)
        pose[self.joint_id['joint_RWing_yaw']] = np.deg2rad(17)

        pose[self.joint_id['joint_Head']] = np.deg2rad(10)

        ind = t + 1500

        ####### Walk on floor#########
        '''
        init_lim = 25
        if ind<init_lim:
            pose[self.joint_id['prismatic_support_2']] = (1.01*self.MODEL_OFFSET[2]-ind*self.MODEL_OFFSET[2]/init_lim)*self.units.meters
        else:
            pose[self.joint_id['prismatic_support_2']] = 0
        '''
        ####LEFT LEGS#######
        pose[self.joint_id['joint_LFCoxa_roll']
             ] = self.angles['LF_leg']['roll'][ind]
        pose[self.joint_id['joint_LFCoxa_yaw']
             ] = self.angles['LF_leg']['yaw'][ind]
        pose[self.joint_id['joint_LFCoxa']] = self.angles['LF_leg']['pitch'][ind]
        pose[self.joint_id['joint_LFFemur_roll']
             ] = self.angles['LF_leg']['roll_tr'][ind]
        pose[self.joint_id['joint_LFFemur']
             ] = self.angles['LF_leg']['th_fe'][ind]
        pose[self.joint_id['joint_LFTibia']
             ] = self.angles['LF_leg']['th_ti'][ind]
        pose[self.joint_id['joint_LFTarsus1']
             ] = self.angles['LF_leg']['th_ta'][ind]

        pose[self.joint_id['joint_LMCoxa_roll']
             ] = self.angles['LM_leg']['roll'][ind]
        pose[self.joint_id['joint_LMCoxa_yaw']
             ] = self.angles['LM_leg']['yaw'][ind]
        pose[self.joint_id['joint_LMCoxa']] = self.angles['LM_leg']['pitch'][ind]
        pose[self.joint_id['joint_LMFemur_roll']
             ] = self.angles['LM_leg']['roll_tr'][ind]
        pose[self.joint_id['joint_LMFemur']
             ] = self.angles['LM_leg']['th_fe'][ind]
        pose[self.joint_id['joint_LMTibia']
             ] = self.angles['LM_leg']['th_ti'][ind]
        pose[self.joint_id['joint_LMTarsus1']
             ] = self.angles['LM_leg']['th_ta'][ind]

        pose[self.joint_id['joint_LHCoxa_roll']
             ] = self.angles['LH_leg']['roll'][ind]
        pose[self.joint_id['joint_LHCoxa_yaw']
             ] = self.angles['LH_leg']['yaw'][ind]
        pose[self.joint_id['joint_LHCoxa']] = self.angles['LH_leg']['pitch'][ind]
        pose[self.joint_id['joint_LHFemur_roll']
             ] = self.angles['LH_leg']['roll_tr'][ind]
        pose[self.joint_id['joint_LHFemur']
             ] = self.angles['LH_leg']['th_fe'][ind]
        pose[self.joint_id['joint_LHTibia']
             ] = self.angles['LH_leg']['th_ti'][ind]
        pose[self.joint_id['joint_LHTarsus1']
             ] = self.angles['LH_leg']['th_ta'][ind]

        #####RIGHT LEGS######
        pose[self.joint_id['joint_RFCoxa_roll']
             ] = self.angles['RF_leg']['roll'][ind]
        pose[self.joint_id['joint_RFCoxa_yaw']
             ] = self.angles['RF_leg']['yaw'][ind]
        pose[self.joint_id['joint_RFCoxa']] = self.angles['RF_leg']['pitch'][ind]
        pose[self.joint_id['joint_RFFemur_roll']
             ] = self.angles['RF_leg']['roll_tr'][ind]
        pose[self.joint_id['joint_RFFemur']
             ] = self.angles['RF_leg']['th_fe'][ind]
        pose[self.joint_id['joint_RFTibia']
             ] = self.angles['RF_leg']['th_ti'][ind]
        pose[self.joint_id['joint_RFTarsus1']
             ] = self.angles['RF_leg']['th_ta'][ind]

        pose[self.joint_id['joint_RMCoxa_roll']
             ] = self.angles['RM_leg']['roll'][ind]
        pose[self.joint_id['joint_RMCoxa_yaw']
             ] = self.angles['RM_leg']['yaw'][ind]
        pose[self.joint_id['joint_RMCoxa']] = self.angles['RM_leg']['pitch'][ind]
        pose[self.joint_id['joint_RMFemur_roll']
             ] = self.angles['RM_leg']['roll_tr'][ind]
        pose[self.joint_id['joint_RMFemur']
             ] = self.angles['RM_leg']['th_fe'][ind]
        pose[self.joint_id['joint_RMTibia']
             ] = self.angles['RM_leg']['th_ti'][ind]
        pose[self.joint_id['joint_RMTarsus1']
             ] = self.angles['RM_leg']['th_ta'][ind]

        pose[self.joint_id['joint_RHCoxa_roll']
             ] = self.angles['RH_leg']['roll'][ind]
        pose[self.joint_id['joint_RHCoxa_yaw']
             ] = self.angles['RH_leg']['yaw'][ind]
        pose[self.joint_id['joint_RHCoxa']] = self.angles['RH_leg']['pitch'][ind]
        pose[self.joint_id['joint_RHFemur_roll']
             ] = self.angles['RH_leg']['roll_tr'][ind]
        pose[self.joint_id['joint_RHFemur']
             ] = self.angles['RH_leg']['th_fe'][ind]
        pose[self.joint_id['joint_RHTibia']
             ] = self.angles['RH_leg']['th_ti'][ind]
        pose[self.joint_id['joint_RHTarsus1']
             ] = self.angles['RH_leg']['th_ta'][ind]
        #################### VELOCITY SET ####################

        ####LEFT LEGS#######
        vel[self.joint_id['joint_LFCoxa_roll']
            ] = self.velocities['LF_leg']['roll'][ind+0]
        vel[self.joint_id['joint_LFCoxa_yaw']
            ] = self.velocities['LF_leg']['yaw'][ind+0]
        vel[self.joint_id['joint_LFCoxa']
            ] = self.velocities['LF_leg']['pitch'][ind+0]
        vel[self.joint_id['joint_LFFemur_roll']
            ] = self.velocities['LF_leg']['roll_tr'][ind+0]
        vel[self.joint_id['joint_LFFemur']
            ] = self.velocities['LF_leg']['th_fe'][ind+0]
        vel[self.joint_id['joint_LFTibia']
            ] = self.velocities['LF_leg']['th_ti'][ind+0]
        vel[self.joint_id['joint_LFTarsus1']
            ] = self.velocities['LF_leg']['th_ta'][ind+0]

        vel[self.joint_id['joint_LMCoxa_roll']
            ] = self.velocities['LM_leg']['roll'][ind+0]
        vel[self.joint_id['joint_LMCoxa_yaw']
            ] = self.velocities['LM_leg']['yaw'][ind+0]
        vel[self.joint_id['joint_LMCoxa']
            ] = self.velocities['LM_leg']['pitch'][ind+0]
        vel[self.joint_id['joint_LMFemur_roll']
            ] = self.velocities['LM_leg']['roll_tr'][ind+0]
        vel[self.joint_id['joint_LMFemur']
            ] = self.velocities['LM_leg']['th_fe'][ind+0]
        vel[self.joint_id['joint_LMTibia']
            ] = self.velocities['LM_leg']['th_ti'][ind+0]
        vel[self.joint_id['joint_LMTarsus1']
            ] = self.velocities['LM_leg']['th_ta'][ind+0]

        vel[self.joint_id['joint_LHCoxa_roll']
            ] = self.velocities['LH_leg']['roll'][ind+0]
        vel[self.joint_id['joint_LHCoxa_yaw']
            ] = self.velocities['LH_leg']['yaw'][ind+0]
        vel[self.joint_id['joint_LHCoxa']
            ] = self.velocities['LH_leg']['pitch'][ind+0]
        vel[self.joint_id['joint_LHFemur_roll']
            ] = self.velocities['LH_leg']['roll_tr'][ind+0]
        vel[self.joint_id['joint_LHFemur']
            ] = self.velocities['LH_leg']['th_fe'][ind+0]
        vel[self.joint_id['joint_LHTibia']
            ] = self.velocities['LH_leg']['th_ti'][ind+0]
        vel[self.joint_id['joint_LHTarsus1']
            ] = self.velocities['LH_leg']['th_ta'][ind+0]

        #####RIGHT LEGS######
        vel[self.joint_id['joint_RFCoxa_roll']
            ] = self.velocities['RF_leg']['roll'][ind+0]
        vel[self.joint_id['joint_RFCoxa_yaw']
            ] = self.velocities['RF_leg']['yaw'][ind+0]
        vel[self.joint_id['joint_RFCoxa']
            ] = self.velocities['RF_leg']['pitch'][ind+0]
        vel[self.joint_id['joint_RFFemur_roll']
            ] = self.velocities['RF_leg']['roll_tr'][ind+0]
        vel[self.joint_id['joint_RFFemur']
            ] = self.velocities['RF_leg']['th_fe'][ind+0]
        vel[self.joint_id['joint_RFTibia']
            ] = self.velocities['RF_leg']['th_ti'][ind+0]
        vel[self.joint_id['joint_RFTarsus1']
            ] = self.velocities['RF_leg']['th_ta'][ind+0]

        vel[self.joint_id['joint_RMCoxa_roll']
            ] = self.velocities['RM_leg']['roll'][ind+0]
        vel[self.joint_id['joint_RMCoxa_yaw']
            ] = self.velocities['RM_leg']['yaw'][ind+0]
        vel[self.joint_id['joint_RMCoxa']
            ] = self.velocities['RM_leg']['pitch'][ind+0]
        vel[self.joint_id['joint_RMFemur_roll']
            ] = self.velocities['RM_leg']['roll_tr'][ind+0]
        vel[self.joint_id['joint_RMFemur']
            ] = self.velocities['RM_leg']['th_fe'][ind+0]
        vel[self.joint_id['joint_RMTibia']
            ] = self.velocities['RM_leg']['th_ti'][ind+0]
        vel[self.joint_id['joint_RMTarsus1']
            ] = self.velocities['RM_leg']['th_ta'][ind+0]

        vel[self.joint_id['joint_RHCoxa_roll']
            ] = self.velocities['RH_leg']['roll'][ind+0]
        vel[self.joint_id['joint_RHCoxa_yaw']
            ] = self.velocities['RH_leg']['yaw'][ind+0]
        vel[self.joint_id['joint_RHCoxa']
            ] = self.velocities['RH_leg']['pitch'][ind+0]
        vel[self.joint_id['joint_RHFemur_roll']
            ] = self.velocities['RH_leg']['roll_tr'][ind+0]
        vel[self.joint_id['joint_RHFemur']
            ] = self.velocities['RH_leg']['th_fe'][ind+0]
        vel[self.joint_id['joint_RHTibia']
            ] = self.velocities['RH_leg']['th_ti'][ind+0]
        vel[self.joint_id['joint_RHTarsus1']
            ] = self.velocities['RH_leg']['th_ta'][ind+0]

        joint_control_middle = list(
            np.arange(42, 49)) + list(np.arange(81, 88))
        joint_control_front = list(np.arange(17, 23)) + list(np.arange(56, 63))
        joint_control_hind = list(np.arange(28, 35)) + list(np.arange(67, 74))
        joint_control = joint_control_hind + joint_control_middle + joint_control_front
        #kp = p.readUserDebugParameter(self.kpId)
        #kv = p.readUserDebugParameter(self.kpId)

        for joint in range(self.num_joints):
            if joint in joint_control:
                p.setJointMotorControl2(
                    self.animal, joint,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=pose[joint],
                    targetVelocity=vel[joint],
                    positionGain=self.kp,
                    velocityGain=self.kv,
                    #maxVelocity = 50
                    #force = 0.55
                )
            else:
                p.setJointMotorControl2(
                self.animal, joint,
                controlMode=p.POSITION_CONTROL,
                targetPosition=pose[joint],
                )

    def feedback_to_controller(self):
        """
        Code that glues the sensors/feedback to controller in the system.
        """
        pass

    def update_parameters(self, params):
        """ Update parameters. """
        pass


def main():
    """ Main """
    run_time = 4.0
    time_step = 0.001
    behavior = 'walking'

    side = ['L', 'R']
    pos = ['F', 'M', 'H']
    leg_segments = ['Tibia']+['Tarsus' + str(i) for i in range(1, 6)]
    left_front_leg = ['LF'+name for name in leg_segments]
    right_front_leg = ['RF'+name for name in leg_segments]
    body_segments = [s+b for s in side for b in ['Eye', 'Antenna']]
    col_hind_leg = [s+'H'+leg for s in side for leg in ['Coxa',
                                                        'Coxa_roll', 'Femur', 'Femur_roll', 'Tibia']]
    col_body_abd = ['prismatic_support_1',
                    'prismatic_support_2', 'A1A2', 'A3', 'A4', 'A5', 'A6']
    ground_contact = [
        s+p+name for s in side for p in pos for name in leg_segments if name != 'Tibia']

    self_collision = []
    for link0 in left_front_leg:
        for link1 in right_front_leg:
            self_collision.append([link0, link1])

    for link0 in left_front_leg+right_front_leg:
        for link1 in body_segments:
            if link0[0] == link1[0]:
                self_collision.append([link0, link1])

    for link0 in col_hind_leg:
        for link1 in col_body_abd:
            self_collision.append([link0, link1])

    positionGain = np.arange(0.1, 1.1, 0.1)
    velocityGain = np.arange(0.1, 1.1, 0.1)
    '''
    Gains = [[0.21972656, 0.09667969],
    [0.51855469, 0.09667969],
    [0.21972656, 0.67675781],
    [0.21972656, 0.67675781],
    [0.51855469, 0.09667969],
    [0.51855469, 0.67675781],
    [0.71972656, 0.59667969],
    [0.01855469, 0.59667969],
    [0.71972656, 0.17675781],
    [0.71972656, 0.17675781],
    [0.01855469, 0.59667969],
    [0.01855469, 0.17675781],
    [0.96972656, 0.34667969],
    [0.76855469, 0.34667969],
    [0.96972656, 0.92675781],
    [0.96972656, 0.92675781],
    [0.76855469, 0.34667969],
    [0.76855469, 0.92675781],
    [0.46972656, 0.84667969],
    [0.26855469, 0.84667969],
    [0.46972656, 0.42675781],
    [0.46972656, 0.42675781],
    [0.26855469, 0.84667969],
    [0.26855469, 0.42675781],
    [0.34472656, 0.47167969],
    [0.14355469, 0.47167969],
    [0.34472656, 0.30175781],
    [0.34472656, 0.30175781],
    [0.14355469, 0.47167969],
    [0.14355469, 0.30175781],
    [0.84472656, 0.97167969],
    [0.64355469, 0.97167969],
    [0.84472656, 0.80175781],
    [0.84472656, 0.80175781],
    [0.64355469, 0.97167969],
    [0.64355469, 0.80175781],
    [0.59472656, 0.22167969],
    [0.39355469, 0.22167969],
    [0.59472656, 0.05175781],
    [0.59472656, 0.05175781],
    [0.39355469, 0.22167969],
    [0.39355469, 0.05175781],
    [0.09472656, 0.72167969],
    [0.89355469, 0.72167969],
    [0.09472656, 0.55175781],
    [0.09472656, 0.55175781],
    [0.89355469, 0.72167969],
    [0.89355469, 0.55175781],
    [0.06347656, 0.19042969],
    [0.04980469, 0.19042969],
    [0.06347656, 0.45800781],
    [0.06347656, 0.45800781],
    [0.04980469, 0.19042969],
    [0.04980469, 0.45800781],
    [0.56347656, 0.69042969],
    [0.54980469, 0.69042969],
    [0.56347656, 0.95800781],
    [0.56347656, 0.95800781],
    [0.54980469, 0.69042969],
    [0.54980469, 0.95800781]]
    '''
    for Kp in [0.4]:
        for Kv in [0.9]:
            sim_options = {
                "headless": False,
                "model": "../../design/sdf/neuromechfly_noLimits_noSupport.sdf",
                #"model_offset": [0., -0.1, 1.12],
                "model_offset": [0, 0.,1.4e-3],
                "run_time": run_time,
                "pose": '../../config/pose_optimization.yaml',
                "base_link": 'Thorax',
                # "controller": '../config/locomotion_trot.graphml',
                "ground_contacts": ground_contact,
                "self_collisions": self_collision,
                "draw_collisions": False,
                "record": False,
                # 'camera_distance': 0.35,
                'camera_distance': 6.0,
                'track': False,
                'moviename': './040421_walking_contacterp0.1_noSupport_perturbation.mp4',
                'moviefps': 80,
                'slow_down': False,
                'sleep_time': 0.001,
                'rot_cam': False,
                'behavior': behavior,
                'ground': 'floor'
            }
            container = Container(run_time/time_step)
            animal = DrosophilaSimulation(container, sim_options, Kp=Kp, Kv=Kv)
            animal.run(optimization=False)
            animal.container.dump(
                dump_path="./basepositionrecorded", overwrite=True)


if __name__ == '__main__':
    main()