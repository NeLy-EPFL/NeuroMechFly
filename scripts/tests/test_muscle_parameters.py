import os

import farms_pylog as pylog
import numpy as np
import pybullet as p
from farms_container import Container
from NeuroMechFly.control.spring_damper_muscles import (Parameters,
                                                        SDAntagonistMuscle)
from NeuroMechFly.sdf.units import SimulationUnitScaling
from NeuroMechFly.simulation.bullet_simulation import BulletSimulation

pylog.set_level("error")


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
            'alpha', 1e-2, 1e-0, 1e-2)
        self.debug_parameters['beta'] = p.addUserDebugParameter(
            'beta', 1e-2, 1e-0, 1e-2)
        self.debug_parameters['gamma'] = p.addUserDebugParameter(
            'gamma', 1e-3, 1e-0, 1e-3)
        self.debug_parameters['delta'] = p.addUserDebugParameter(
            'delta', 1e-6, 1e-4, 1e-5)
        self.debug_parameters['rest_pos'] = p.addUserDebugParameter(
            'rest_position',
            p.getJointInfo(self.animal, self.debug_joint_id)[8],
            p.getJointInfo(self.animal, self.debug_joint_id)[9],
        )
        self.debug_muscle_act['flexion'] = p.addUserDebugParameter(
            'flexion', 0, 2, 0.0)
        self.debug_muscle_act['extension'] = p.addUserDebugParameter(
            'extension', 0, 2, 0.0)

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

    def feedback_to_controller(self):
        """ Implementation of abstractmethod. """
        pass

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

    sim_options = {
        "headless": False,
        "time_step": 5e-4,
        # Scaled SDF model
        "model": "../../data/design/sdf/fly_locomotion.sdf",
        "model_offset": [0., 0., 11.2e-3],
        "run_time": run_time,
        "pose": '../../data/config/pose/pose_tripod.yaml',
        "base_link": 'Thorax',
        "controller": '../../data/config/network/locomotion_network.graphml',
        "draw_collisions": False,
        'camera_distance': 3.5,
        'slow_down': False,
        "ground": "floor",
        'sleep_time': 1e-3,
        "is_ball": False
    }

    container = Container(run_time/time_step)
    animal = DrosophilaSimulation(container, sim_options)

    animal.run(optimization=False)


if __name__ == '__main__':
    main()
