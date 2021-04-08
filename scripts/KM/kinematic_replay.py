import pybullet as p
import numpy as np
import pickle

from NeuroMechFly.simulation.bullet_simulation import BulletSimulation
from NeuroMechFly.container import Container
from NeuroMechFly.sdf.units import SimulationUnitScaling


class DrosophilaSimulation(BulletSimulation):

    def __init__(
        self,
        container,
        sim_options,
        Kp,
        Kv,
        units=SimulationUnitScaling(
            meters=1000,
            kilograms=1000)):
        super().__init__(container, units, **sim_options)
        self.kp = Kp
        self.kv = Kv
        self.pose = [0] * self.num_joints
        self.vel = [0] * self.num_joints
        self.angles = self.load_angles(
            f'./new_angles/{self.behavior}_converted_joint_angles.pkl')
        self.velocities = self.load_angles(
            f'./new_angles/{self.behavior}_converted_joint_velocities.pkl')

    def load_angles(self, data_path):
        try:
            with open(data_path, 'rb') as f:
                return pickle.load(f)
        except BaseException:
            FileNotFoundError(f"File {data_path} not found!")

    def controller_to_actuator(self, t):
        """
        Code that glues the controller the actuator in the system.
        If there are muscles then contoller actuates the muscles.
        If not then the controller directly actuates the joints.
        """

        #: Setting the fixed joint positions
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

        for joint in range(self.num_joints):
            p.setJointMotorControl2(
                self.animal, joint,
                controlMode=p.POSITION_CONTROL,
                targetPosition=self.pose[joint],
                targetVelocity=self.vel[joint],
                positionGain=self.kp,
                velocityGain=self.kv,
            )

    def feedback_to_controller(self):
        """
        Code that glues the sensors/feedback to controller in the system.
        """
        pass

    def update_parameters(self, params):
        """ Update parameters. """
        pass

    def optimization_check(self):
        """ Optimization check. """
        pass


def main():
    """ Main """
    run_time = 4.0
    time_step = 0.001
    behavior = 'walking'

    #: Setting up the collision and ground sensors
    side = ['L', 'R']
    pos = ['F', 'M', 'H']
    leg_segments = ['Tibia'] + ['Tarsus' + str(i) for i in range(1, 6)]
    left_front_leg = ['LF' + name for name in leg_segments]
    right_front_leg = ['RF' + name for name in leg_segments]
    body_segments = [s + b for s in side for b in ['Eye', 'Antenna']]

    self_collision = []
    for link0 in left_front_leg:
        for link1 in right_front_leg:
            self_collision.append([link0, link1])

    for link0 in left_front_leg + right_front_leg:
        for link1 in body_segments:
            if link0[0] == link1[0]:
                self_collision.append([link0, link1])

    ground_contact = [
        s +
        p +
        name for s in side for p in pos for name in leg_segments if name != 'Tibia']

    sim_options = {
        "headless": False,
        "model": "../../data/design/sdf/neuromechfly_noLimits.sdf",
        "model_offset": [0., -0.1e-3, 11.2e-3],
        "run_time": run_time,
        "base_link": 'Thorax',
        "ground_contacts": ground_contact,
        "self_collisions": self_collision,
        "draw_collisions": False,
        "record": False,
        'camera_distance': 6.0,
        'track': False,
        'moviename': './videos/kinematic_replay_video.mp4',
        'moviefps': 80,
        'slow_down': False,
        'sleep_time': 0.001,
        'rot_cam': False,
        'behavior': behavior,
        'ground': 'ball'
    }

    container = Container(run_time / time_step)
    animal = DrosophilaSimulation(container, sim_options, Kp=0.4, Kv=0.9)
    animal.run(optimization=False)
    animal.container.dump(
        dump_path="./kinematic_replay",
        overwrite=False)


if __name__ == '__main__':
    main()
