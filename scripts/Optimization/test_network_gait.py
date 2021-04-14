import os
import pickle
import time

import farms_pylog as pylog
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

    def __init__(self, container, sim_options, units=SimulationUnitScaling(meters=1000,kilograms=1000)):
        ########## Initialize bullet simulation ##########
        super().__init__(container, units, **sim_options)

        ########## Initialize container ##########
        #: FUCK : THIS IS BAD!!!
        self.container.initialize()

        self.sides = ('L', 'R')
        self.positions = ('F', 'M', 'H')
        self.feet_links = tuple([
            '{}{}Tarsus{}'.format(side, pos,seg)
            for side in self.sides
            for pos in self.positions
            for seg in range(1,6)
        ])
        _joints = ('Coxa', 'Femur', 'Tibia')
        self.actuated_joints = [
            'joint_{}{}{}'.format(side, pos, joint)
            for side in self.sides
            for pos in self.positions
            for joint in _joints
        ]
        for j, joint in enumerate(self.actuated_joints):
            pos = joint.split('_')[1][1]
            if (('M' == pos) or ('H' == pos)) and ('Coxa' in joint):
                self.actuated_joints[j] = joint.replace('Coxa', 'Coxa_roll')

        #: Set
        for link, idx in self.link_id.items():
            p.changeDynamics(
                self.animal,
                idx,
                lateralFriction=1.0,
                restitution=0.1,
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
        #p.setCollisionFilterPair(
        #    self.animal, self.plane, self.link_id['Head'], -1, 0
        #)
        ########## DEBUG PARAMETER ##########
        self.debug = p.addUserDebugParameter('debug', -1, 1, 0.0)

        ########## Data variables ###########
        self.torques=[]
        self.grf=[]
        self.collision_forces=[]
        self.ball_rot=[]
        self.stability_coef = 0
        self.stance_count = 0
        self.lastDraw=[]

    def fixed_joints_controller(self):
        """Controller for fixed joints"""
        for joint in range(self.num_joints):
            joint_name = [name for name, ind_num in self.joint_id.items() if joint == ind_num][0]
            if joint_name not in self.actuated_joints:# and 'support' not in joint_name:
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

                #print('lava: ',self.is_lava())
                #print('fly: ',self.is_flying())
                #print('touch: ',self.is_touch())
                #print('Ball rotations: ', self.ball_rotations(),'\n')
                #print('Dist stance_polygon:', self.stance_polygon_dist())
                #print(np.array(self.get_link_position('Thorax')[:2]))


                p.setJointMotorControl2(
                    self.animal, joint,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=pos,
                    force=1e36)


    def controller_to_actuator(self, t):
        """ Implementation of abstractmethod. """
        self.fixed_joints_controller()

        outputs = self.container.neural.outputs
        for name in outputs.names:
            if "flexion" in name:
                joint_id = self.joint_id['_'.join(name.split('_')[1:-1])]
                min_val, max_val = p.getJointInfo(self.animal, joint_id)[8:10]
                position = min_val + (max_val-min_val)*(
                    1+np.sin(outputs.values[outputs.name_index[name]])
                )
                p.setJointMotorControl2(
                    self.animal,
                    joint_id,
                    p.POSITION_CONTROL,
                    targetPosition=position
                )

    def feedback_to_controller(self):
        """ Implementation of abstractmethod. """
        pass

    def joint_torques(self):
        """ Get the joint torques in the animal  """
        _joints = np.arange(0, p.getNumJoints(self.animal))
        return tuple(
            state[-1] for state in p.getJointStates(self.animal, _joints))

    def optimization_check(self):
        """ Check optimization status. """
        pass

    def update_parameters(self, params):
        """ Implementation of abstractmethod. """
        pass


def main():
    """ Main """
    run_time = 5.
    time_step = 0.001

    side = ['L','R']
    pos = ['F','M','H']
    leg_segments = ['Femur','Tibia']+['Tarsus' + str(i) for i in range(1, 6)]

    ground_contact = [s+p+name for s in side for p in pos for name in leg_segments if 'Tarsus' in name]

    left_front_leg = ['LF'+name for name in leg_segments]
    left_middle_leg = ['LM'+name for name in leg_segments]
    left_hind_leg = ['LH'+name for name in leg_segments]

    right_front_leg = ['RF'+name for name in leg_segments]
    right_middle_leg = ['RM'+name for name in leg_segments]
    right_hind_leg = ['RH'+name for name in leg_segments]

    body_segments = ['A1A2','A3','A4','A5','A6','Thorax','Head']

    self_collision = []
    for link0 in left_front_leg:
        for link1 in left_middle_leg:
            self_collision.append([link0,link1])
    for link0 in left_middle_leg:
        for link1 in left_hind_leg:
            self_collision.append([link0,link1])
    for link0 in left_front_leg:
        for link1 in body_segments:
            self_collision.append([link0,link1])
    for link0 in left_middle_leg:
        for link1 in body_segments:
            self_collision.append([link0,link1])
    for link0 in left_hind_leg:
        for link1 in body_segments:
            self_collision.append([link0,link1])

    for link0 in right_front_leg:
        for link1 in right_middle_leg:
            self_collision.append([link0,link1])
    for link0 in right_middle_leg:
        for link1 in right_hind_leg:
            self_collision.append([link0,link1])
    for link0 in right_front_leg:
        for link1 in body_segments:
            self_collision.append([link0,link1])
    for link0 in right_middle_leg:
        for link1 in body_segments:
            self_collision.append([link0,link1])
    for link0 in right_hind_leg:
        for link1 in body_segments:
            self_collision.append([link0,link1])

    gen = '10'
    exp = 'run_Drosophila_var_71_obj_2_pop_20_gen_100_0407_1744'

    sim_options = {
        "headless": False,
        # Scaled SDF model
        "model": "../../design/sdf/neuromechfly_limitsFromData_minMax.sdf",
        "model_offset": [0., 0., 11.2e-3],
        "run_time": run_time,
        "pose": '../../config/test_pose_tripod.yaml',
        "base_link": 'Thorax',
        "controller": "../../config/locomotion_tripod.graphml",
        "ground_contacts": ground_contact,
        'self_collisions':self_collision,
        "draw_collisions": True,
        "record": False,
        'camera_distance': 3.5,
        'track': False,
        'moviename': 'stability_'+exp+'_gen_'+gen+'.mp4',
        'moviefps': 50,
        'slow_down': False,
        'sleep_time': 10.0,
        'rot_cam': False,
        'is_ball' : False
        }

    container = Container(run_time/time_step)
    animal = DrosophilaSimulation(container, sim_options)
    animal.run()


if __name__ == '__main__':
    main()
