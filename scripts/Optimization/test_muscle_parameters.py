import os
import pickle
import time

import numpy as np
import pandas as pd
from IPython import embed

import pybullet as p
import pybullet_data
from bullet_simulation_opt import BulletSimulation
from NeuroMechFly.container import Container
from NeuroMechFly.sdf.units import SimulationUnitScaling
from spring_damper_muscles import Parameters, SDAntagonistMuscle


class DrosophilaSimulation(BulletSimulation):
    """Drosophila Simulation Class
    """

    def __init__(
            self, container, sim_options,
            units=SimulationUnitScaling(meters=1000, kilograms=1000)
    ):
        ########## Container ##########
        container.add_namespace('muscle')
        container.muscle.add_table('parameters', table_type='CONSTANT')
        container.muscle.add_table('outputs')
        container.muscle.add_table('active_torques')
        container.muscle.add_table('passive_torques')
        ########## Initialize bullet simulation ##########
        super().__init__(container, units, **sim_options)
        ########## Parameters ##########
        self.sides = ('L', 'R')
        # FIXME: positions is a very bad name for this
        self.positions = ('F', 'M', 'H')
        self.feet_links = tuple([
            '{}{}Tarsus{}'.format(side, pos, seg)
            for side in self.sides
            for pos in self.positions
            for seg in range(1, 6)
        ])
        _joints = ('Coxa', 'Femur', 'Tibia')
        self.actuated_joints = tuple([
            f'joint_{side}{pos}{joint}_roll'
            if (pos+joint == "MCoxa") or  (pos+joint == "HCoxa")
            else f'joint_{side}{pos}{joint}'
            for side in self.sides
            for pos in self.positions
            for joint in _joints
        ])

        self.num_oscillators = self.controller.graph.number_of_nodes()
        self.active_muscles = {}
        self.neural = self.container.neural
        self.physics = self.container.physics
        self.muscle = self.container.muscle
        ########## Initialize joint muscles ##########
        self.debug_joint = 'joint_RFCoxa'
        for joint in [self.debug_joint,]:
            fmn = self.neural.states.get_parameter(
                'phase_' + joint + '_flexion')
            emn = self.neural.states.get_parameter(
                'phase_' + joint + '_extension')
            fmn_amp = self.neural.states.get_parameter(
                'amp_' + joint + '_flexion')
            emn_amp = self.neural.states.get_parameter(
                'amp_' + joint + '_extension')
            jpos = self.physics.joint_positions.get_parameter(joint)
            jvel = self.physics.joint_velocities.get_parameter(joint)
            joint_info = p.getJointInfo(self.animal, self.joint_id[joint])
            lower_limit = joint_info[8]
            upper_limit = joint_info[9]
            self.active_muscles[joint] = SDAntagonistMuscle(
                self.container,
                name=joint,
                joint_pos=jpos,
                joint_vel=jvel,
                rest_pos=(lower_limit + upper_limit)*0.5,
                flexor_mn=fmn,
                extensor_mn=emn,
                flexor_amp=fmn_amp,
                extensor_amp=fmn_amp,
            )

        ########## Initialize container ##########
        # FIXME: THIS IS BAD!!!
        self.container.initialize()

        #: Set
        for link, idx in self.link_id.items():
            p.changeDynamics(
                self.animal,
                idx,
                lateralFriction=1.0,
                restitution=1e-3,
                spinningFriction=0.0,
                rollingFriction=0.0,
                linearDamping=0.0,
                angularDamping=0.0,
            )
            p.changeDynamics(
                self.animal,
                idx,
                maxJointVelocity=10000.0
            )

        # ########## DISABLE COLLISIONS ##########
        # p.setCollisionFilterPair(
        #    self.animal, self.plane, self.link_id['Head'], -1, 0
        # )
        ########## DEBUG PARAMETER ##########
        self.debug_joint_id = self.joint_id[self.debug_joint]
        self.debug_parameters = {}
        self.debug_muscle_act = {}
        self.debug_parameters['alpha'] = p.addUserDebugParameter(
            'alpha', 1e-2, 1e1, 1e-1)
        self.debug_parameters['beta'] = p.addUserDebugParameter(
            'beta', 1e-2, 1e1, 1e-1)
        self.debug_parameters['gamma'] = p.addUserDebugParameter(
            'gamma', 1e-3, 1e1, 1e-3)
        self.debug_parameters['delta'] = p.addUserDebugParameter(
            'delta', 1e-5, 1e-3, 1e-5)
        self.debug_parameters['rest_pos'] = p.addUserDebugParameter(
            'rest_position',
            p.getJointInfo(self.animal, self.debug_joint_id)[8],
            p.getJointInfo(self.animal, self.debug_joint_id)[9],
        )
        self.debug_muscle_act['flexion'] = p.addUserDebugParameter(
            'flexion', 0, 2, 0.5)
        self.debug_muscle_act['extension'] = p.addUserDebugParameter(
            'extension', 0, 2, 0.5)

        ########## Data variables ###########
        self.torques = []
        self.grf = []
        self.collision_forces = []
        self.ball_rot = []
        self.stability_coef = 0
        self.stance_count = 0
        self.lastDraw = []

    def fixed_joints_controller(self):
        """Controller for fixed joints"""
        for joint in range(self.num_joints):
            joint_name = [name for name, ind_num in self.joint_id.items() if joint == ind_num][0]
            # FIXME: Resort to the pose file
            if joint_name not in self.actuated_joints and 'support' not in joint_name:
                if joint_name == 'joint_A3' or joint_name == 'joint_A4' or joint_name == 'joint_A5' or joint_name == 'joint_A6':
                    pos = np.deg2rad(-15)
                elif joint_name == 'joint_LAntenna':
                    pos = np.deg2rad(33)
                elif joint_name == 'joint_RAntenna':
                    pos = np.deg2rad(-33)
                elif joint_name == 'joint_Rostrum' or joint_name == 'joint_LWing_roll':
                    pos = np.deg2rad(90)
                elif joint_name == 'joint_Haustellum':
                    pos = np.deg2rad(-60)
                elif joint_name == 'joint_RWing_roll':
                    pos = np.deg2rad(-90)
                elif joint_name == 'joint_LWing_yaw':
                    pos = np.deg2rad(-17)
                elif joint_name == 'joint_RWing_yaw':
                    pos = np.deg2rad(17)
                elif joint_name == 'joint_Head':
                    pos = np.deg2rad(10)
                #elif joint_name == 'joint_LFCoxa_yaw':
                #    pos = np.deg2rad(-5)
                #elif joint_name == 'joint_RFCoxa_yaw':
                #    pos = np.deg2rad(5)
                elif joint_name == 'joint_LFCoxa_roll':
                    pos = np.deg2rad(10)
                elif joint_name == 'joint_RFCoxa_roll':
                    pos = np.deg2rad(-10)
                #elif joint_name == 'joint_LFFemur_roll':
                #    pos = np.deg2rad(-26)
                #elif joint_name == 'joint_RFFemur_roll':
                #    pos = np.deg2rad(26)
                elif joint_name == 'joint_LFTarsus1':
                    pos = np.deg2rad(-43)
                elif joint_name == 'joint_RFTarsus1':
                    pos = np.deg2rad(-49)
                elif joint_name == 'joint_LMCoxa_yaw':
                    pos = np.deg2rad(4)
                elif joint_name == 'joint_RMCoxa_yaw':
                    pos = np.deg2rad(0.5)
                elif joint_name == 'joint_LMCoxa':
                    pos = np.deg2rad(-2)
                elif joint_name == 'joint_RMCoxa':
                    pos = np.deg2rad(-4.5)
                #elif joint_name == 'joint_LMFemur_roll':
                #    pos = np.deg2rad(-7)
                #elif joint_name == 'joint_RMFemur_roll':
                #    pos = np.deg2rad(7)
                elif joint_name == 'joint_LMTarsus1':
                    pos = np.deg2rad(-52)
                elif joint_name == 'joint_RMTarsus1':
                    pos = np.deg2rad(-56)
                elif joint_name == 'joint_LHCoxa_yaw':
                    pos = np.deg2rad(0.6)
                elif joint_name == 'joint_RHCoxa_yaw':
                    pos = np.deg2rad(6.2)
                elif joint_name == 'joint_LHCoxa':
                    pos = np.deg2rad(13)
                elif joint_name == 'joint_RHCoxa':
                    pos = np.deg2rad(11.4)
                #elif joint_name == 'joint_LHFemur_roll':
                #    pos = np.deg2rad(9)
                #elif joint_name == 'joint_RHFemur_roll':
                #    pos = np.deg2rad(-9)
                elif joint_name == 'joint_LHTarsus1':
                    pos = np.deg2rad(-45)
                elif joint_name == 'joint_RHTarsus1':
                    pos = np.deg2rad(-50)
                else:
                    pos = 0

                p.setJointMotorControl2(
                    self.animal, joint,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=pos,
                    force=1e36)

    def controller_to_actuator(self, t):
        """ Implementation of abstractmethod. """

        self.fixed_joints_controller()

        # p.setJointMotorControlArray(
        #     self.animal,
        #     [j for j in range(self.num_joints) if j != self.debug_joint_id],
        #     controlMode=p.POSITION_CONTROL,
        #     targetPositions=np.zeros((self.num_joints-1,))
        # )
        #: Update muscle parameters
        self.active_muscles[self.debug_joint].flexor_mn.value = p.readUserDebugParameter(
            self.debug_muscle_act['flexion']
        )
        self.active_muscles[self.debug_joint].extensor_mn.value = p.readUserDebugParameter(
            self.debug_muscle_act['extension']
        )
        self.active_muscles[self.debug_joint].update_parameters(
            Parameters(**{
                key: p.readUserDebugParameter(value)
                for key, value in self.debug_parameters.items()
            })
        )
        torque = self.active_muscles[self.debug_joint].compute_torque(
            only_passive=False
        )
        p.setJointMotorControl2(
            self.animal,
            self.debug_joint_id,
            controlMode=p.TORQUE_CONTROL,
            force=torque
        )
        # print(torque)
        # print(
        #     self.active_muscles[self.debug_joint].active_torque.value,
        #     self.active_muscles[self.debug_joint].passive_torque.value,
        # )

    def feedback_to_controller(self):
        """ Implementation of abstractmethod. """
        pass

    def joint_torques(self):
        """ Get the joint torques in the animal  """
        _joints = np.arange(0, p.getNumJoints(self.animal))
        return tuple(
            state[-1] for state in p.getJointStates(self.animal, _joints))

    def ball_reaction_forces(self):
        """Get the ground reaction forces.  """
        return list(
            map(self._get_contact_force_ball, self.ground_sensors.values())
        )

    def ball_rotations(self):
        return tuple(
            state[0] for state in p.getJointStates(
                self.ball_id,
                np.arange(0, p.getNumJoints(self.ball_id))
            )
        )

    def stance_polygon_dist(self):
        return None

    def optimization_check(self):
        """ Check optimization status. """
        pass

    def update_parameters(self, params):
        """ Implementation of abstractmethod. """
        pass


def main():
    """ Main """
    run_time = 50.
    time_step = 0.001

    side = ['L', 'R']
    pos = ['F', 'M', 'H']
    leg_segments = ['Femur', 'Tibia']+['Tarsus' + str(i) for i in range(1, 6)]

    ground_contact = [
        s+p+name for s in side for p in pos for name in leg_segments if 'Tarsus' in name]

    left_front_leg = ['LF'+name for name in leg_segments]
    left_middle_leg = ['LM'+name for name in leg_segments]
    left_hind_leg = ['LH'+name for name in leg_segments]

    right_front_leg = ['RF'+name for name in leg_segments]
    right_middle_leg = ['RM'+name for name in leg_segments]
    right_hind_leg = ['RH'+name for name in leg_segments]

    body_segments = ['A1A2', 'A3', 'A4', 'A5', 'A6', 'Thorax', 'Head']

    self_collision = []
    for link0 in left_front_leg:
        for link1 in left_middle_leg:
            self_collision.append([link0, link1])
    for link0 in left_middle_leg:
        for link1 in left_hind_leg:
            self_collision.append([link0, link1])
    for link0 in left_front_leg:
        for link1 in body_segments:
            self_collision.append([link0, link1])
    for link0 in left_middle_leg:
        for link1 in body_segments:
            self_collision.append([link0, link1])
    for link0 in left_hind_leg:
        for link1 in body_segments:
            self_collision.append([link0, link1])

    for link0 in right_front_leg:
        for link1 in right_middle_leg:
            self_collision.append([link0, link1])
    for link0 in right_middle_leg:
        for link1 in right_hind_leg:
            self_collision.append([link0, link1])
    for link0 in right_front_leg:
        for link1 in body_segments:
            self_collision.append([link0, link1])
    for link0 in right_middle_leg:
        for link1 in body_segments:
            self_collision.append([link0, link1])
    for link0 in right_hind_leg:
        for link1 in body_segments:
            self_collision.append([link0, link1])

    gen = '49'
    exp = '1211_0208'

    sim_options = {
        "headless": False,
        # Scaled SDF model
        "model": "../../design/sdf/neuromechfly_limitsFromData_minMax.sdf",
        "model_offset": [0., 0., 4e-4],
        "run_time": run_time,
        "time_step" : 5e-4,
        "pose": '../../config/test_pose_tripod.yaml',
        "base_link": 'Thorax',
        "controller": '../../config/locomotion_tripod.graphml',
        "ground_contacts": ground_contact,
        'self_collisions': self_collision,
        "draw_collisions": False,
        "record": False,
        'camera_distance': 3.5,
        'track': False,
        'moviename': 'stability_'+exp+'_gen_'+gen+'.mp4',
        'moviefps': 50,
        'slow_down': False,
        'sleep_time': 0.1,
        'rot_cam': False,
        "is_ball": False
    }

    container = Container(run_time/time_step)
    animal = DrosophilaSimulation(container, sim_options)

    animal.run(optimization=False)
    animal.container.dump(overwrite=True)


if __name__ == '__main__':
    main()
