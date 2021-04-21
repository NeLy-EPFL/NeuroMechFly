import pybullet as p
import numpy as np
import pandas as pd
from NeuroMechFly.simulation.bullet_simulation import BulletSimulation
from NeuroMechFly.container import Container
from NeuroMechFly.sdf.units import SimulationUnitScaling


class DrosophilaSimulation(BulletSimulation):
    """[summary]

    Parameters
    ----------
    BulletSimulation : [type]
        [description]
    """

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
        self.last_draw = []
        self.grf = []
        self.kp = Kp
        self.kv = Kv
        self.pose = [0] * self.num_joints
        self.vel = [0] * self.num_joints
        self.angles = self.load_angles(
            f'../../data/joint_kinematics/{self.behavior}/{self.behavior}_converted_joint_angles.pkl')
        self.velocities = self.load_angles(
            f'../../data/joint_kinematics/{self.behavior}/{self.behavior}_converted_joint_velocities.pkl')

    def load_angles(self, data_path):
        """[summary]

        Parameters
        ----------
        data_path : [type]
            [description]

        Returns
        -------
        [type]
            [description]
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
        t : [type]
            [description]
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

        #: Change the color of the colliding body segments
        if self.draw_collisions:
            draw = []
            if self.behavior == 'walking':
                links_contact = np.where(
                    np.linalg.norm(
                        self.ground_reaction_forces,
                        axis=1) > 0)[0]
                for i in links_contact:
                    link1 = self. GROUND_CONTACTS[i][:-1]
                    if link1 not in draw:
                        draw.append(link1)
                        self.change_color(link1 + '5', self.color_collision)
                for link in self.last_draw:
                    if link not in draw:
                        self.change_color(link + '5', self.color_legs)

            elif self.behavior == 'grooming':
                links_contact = np.where(
                    np.linalg.norm(
                        self.collision_forces,
                        axis=1) > 0)[0]
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

    def change_color(self, id, color):
        """ Change color of a given body segment. """
        p.changeVisualShape(self.animal, self.link_id[id], rgbaColor=color)

    def feedback_to_controller(self):
        """
        Code that glues the sensors/feedback to controller in the system.
        """

    def update_parameters(self, params):
        """ Update parameters. """ 

    def optimization_check(self):
        """ Optimization check. """


def main():
    """ Main """
    run_time = 8.0
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
        "pose": '../../data/config/pose/pose_optimization_2.yaml',
        "model_offset": [0., 0, 11.2e-3],
        "run_time": run_time,
        "base_link": 'Thorax',
        "ground_contacts": ground_contact,
        "self_collisions": self_collision,
        "draw_collisions": True,
        "record": False,
        'camera_distance': 6.0,
        'track': False,
        'moviename': './videos/kinematic_replay_video.mp4',
        'moviespeed': 0.2,
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
        dump_path=f"./kinematic_replay_{behavior}",
        overwrite=False)


if __name__ == '__main__':
    main()
