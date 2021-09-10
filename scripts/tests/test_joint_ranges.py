import os

import farms_pylog as pylog
import numpy as np
import pybullet as p
from farms_container import Container
from NeuroMechFly.control.spring_damper_muscles import (Parameters,
                                                        SDAntagonistMuscle)
from NeuroMechFly.sdf.units import SimulationUnitScaling
from NeuroMechFly.simulation.bullet_simulation import BulletSimulation


class DrosophilaSimulation(BulletSimulation):
    """Drosophila Simulation Class
    """

    def __init__(
            self, container, sim_options,
            units=SimulationUnitScaling(meters=1000, kilograms=1000)
    ):
        ########## Initialize bullet simulation ##########
        super().__init__(container, units, **sim_options)

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
            if (pos in ('M', 'H')) and ('Coxa' in joint):
                self.actuated_joints[j] = joint.replace('Coxa', 'Coxa_roll')

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
        "model": "../../data/design/sdf/fly_locomotion.sdf",
        "model_offset": [0., 0., 11.2e-3],
        "run_time": run_time,
        'camera_distance': 3.5,
        'slow_down': True,
        'sleep_time': 1e-3,
        'ground': 'floor',
    }

    container = Container(run_time/time_step)
    animal = DrosophilaSimulation(container, sim_options)

    animal.run(optimization=False)
    animal.container.dump(overwrite=False)


if __name__ == '__main__':
    main()
