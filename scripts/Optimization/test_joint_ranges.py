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
        ########## Initialize bullet simulation ##########
        super().__init__(container, units, **sim_options)

        self.actuated_joints = [
            f'joint_{side}{pos}{joint}_roll'
            if (pos+joint == "MCoxa") or  (pos+joint == "HCoxa")
            else f'joint_{side}{pos}{joint}'
            for side in ('L', 'R')
            for pos in ('F', 'M', 'H')
            for joint in ('Coxa', 'Femur', 'Tibia')
        ]

        self.fixed_joints = [
            joint
            for joint in self.joint_id.keys()
            if joint not in self.actuated_joints
        ]
        self.num_fixed_joints = len(self.fixed_joints)

        ########## DEBUG PARAMETER ##########
        self.debug_parameters = {}
        for joint in self.actuated_joints:
            limits = p.getJointInfo(self.animal, self.joint_id[joint])[8:10]
            self.debug_parameters[joint] = p.addUserDebugParameter(
            joint, limits[0], limits[1], limits[1]
            )

    def controller_to_actuator(self, t):
        """ Implementation of abstractmethod. """

        p.setJointMotorControlArray(
            self.animal,
            [self.joint_id[joint] for joint in self.fixed_joints],
            controlMode=p.POSITION_CONTROL,
            targetPositions=np.zeros((self.num_fixed_joints,))
        )
        #: set joint positions
        p.setJointMotorControlArray(
            self.animal,
            [self.joint_id[joint] for joint in self.debug_parameters.keys()],
            controlMode=p.POSITION_CONTROL,
            targetPositions=[
                p.readUserDebugParameter(value)
                for value in self.debug_parameters.values()
            ]
        )

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

    sim_options = {
        "headless": False,
        # Scaled SDF model
        "model": "../../design/sdf/neuromechfly_limitsFromData.sdf",
        "model_offset": [0., 0., 11.2e-3],
        "run_time": run_time,
        "pose": '../../config/pose_optimization_2.yaml',
        "record": False,
        'camera_distance': 3.5,
        'track': False,
        'moviefps': 50,
        'slow_down': False,
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
