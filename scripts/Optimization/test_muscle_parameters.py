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
        ########## Initialize bullet simulation ##########
        super().__init__(container, units, **sim_options)
        ########## Parameters ##########
        self.sides = ('L', 'R')
        self.positions = ('F', 'M', 'H')
        self.feet_links = tuple([
            '{}{}Tarsus{}'.format(side, pos, seg)
            for side in self.sides
            for pos in self.positions
            for seg in range(1, 6)
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

        self.num_oscillators = self.controller.graph.number_of_nodes()
        self.active_muscles = {}
        self.neural = self.container.neural
        self.physics = self.container.physics
        self.muscle = self.container.muscle
        ########## Initialize joint muscles ##########
        self.debug_joint = 'joint_RFTibia'
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
        # p.setCollisionFilterPair(
        #    self.animal, self.plane, self.link_id['Head'], -1, 0
        # )
        ########## DEBUG PARAMETER ##########
        self.debug_joint_id = self.joint_id[self.debug_joint]
        self.debug_parameters = {}
        self.debug_muscle_act = {}
        self.debug_parameters['alpha'] = p.addUserDebugParameter(
            'alpha', 1e-2, 1e-1, 1e-2)
        self.debug_parameters['beta'] = p.addUserDebugParameter(
            'beta', 1e-2, 1e-1, 1e-2)
        self.debug_parameters['gamma'] = p.addUserDebugParameter(
            'gamma', 1e-3, 1e-1, 1e-3)
        self.debug_parameters['delta'] = p.addUserDebugParameter(
            'delta', 1e-4, 1e-3, 1e-4)
        self.debug_parameters['rest_pos'] = p.addUserDebugParameter(
            'rest_position',
            p.getJointInfo(self.animal, self.debug_joint_id)[8],
            p.getJointInfo(self.animal, self.debug_joint_id)[9],
        )
        self.debug_muscle_act['flexion'] = p.addUserDebugParameter(
            'flexion', 0, 1, 0.0)
        self.debug_muscle_act['extension'] = p.addUserDebugParameter(
            'extension', 0, 1, 0.0)

        ########## Data variables ###########
        self.torques = []
        self.grf = []
        self.collision_forces = []
        self.ball_rot = []
        self.stability_coef = 0
        self.stance_count = 0
        self.lastDraw = []

    def controller_to_actuator(self, t):
        """ Implementation of abstractmethod. """
        p.setJointMotorControlArray(
            self.animal,
            [j for j in range(self.num_joints) if j != self.debug_joint_id],
            controlMode=p.POSITION_CONTROL,
            targetPositions=np.zeros((self.num_joints-1,))
        )
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

        p.setJointMotorControl2(
            self.animal,
            self.debug_joint_id,
            controlMode=p.TORQUE_CONTROL,
            force=self.active_muscles[self.debug_joint].compute_torque(
                only_passive=False
            )
        )
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
    run_time = 5.
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
        "model": "../../design/sdf/neuromechfly_limitsFromData.sdf",
        "model_offset": [0., 0., 11.2e-3],
        "run_time": run_time,
        "pose": '../../config/pose_tripod.yaml',
        "base_link": 'Thorax',
        "controller": '../../config/locomotion_ball.graphml',
        "ground_contacts": ground_contact,
        'self_collisions': self_collision,
        "draw_collisions": False,
        "record": False,
        'camera_distance': 3.5,
        'track': False,
        'moviename': 'stability_'+exp+'_gen_'+gen+'.mp4',
        'moviefps': 50,
        'slow_down': True,
        'sleep_time': 0.001,
        'rot_cam': False,
        "is_ball": False
    }

    container = Container(run_time/time_step)
    animal = DrosophilaSimulation(container, sim_options)

    animal.run(optimization=False)
    animal.container.dump(overwrite=True)


if __name__ == '__main__':
    main()
