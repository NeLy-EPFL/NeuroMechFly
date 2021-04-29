from NeuroMechFly.experiments.kinematic_replay import kinematic_replay_no_support
from NeuroMechFly.container import Container
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--behavior', default='walking')
    parser.add_argument('-p', '--perturbation', default=False, action='store_true')
    args = parser.parse_args()

    run_time = 8.0
    time_step = 0.001
    behavior = args.behavior.lower()

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
        "model": "../../data/design/sdf/neuromechfly_noLimits_noSupport.sdf",
        "model": "../data/design/sdf/neuromechfly_noLimits_noSupport.sdf",
        "pose": '../data/config/pose/pose_optimization_2.yaml',
        "model_offset": [0, 0., 2.1e-3],
        "run_time": run_time,
        "time_step": time_step,
        "base_link": 'Thorax',
        "ground_contacts": ground_contact,
        "self_collisions": self_collision,
        "record": True,
        'camera_distance':5.0,
        'track': False,
        'moviename': './kinematic_replay_ground.mp4',
        'moviespeed': 0.2,
        'slow_down': False,
        'sleep_time': 0.001,
        'rot_cam': False,
        'behavior': behavior,
        'ground': 'floor',
        'num_substep': 5
    }

    position_path = f'../data/joint_kinematics/{behavior}/{behavior}_converted_joint_angles.pkl'
    velocity_path = f'../data/joint_kinematics/{behavior}/{behavior}_converted_joint_velocities.pkl'

    container = Container(run_time / time_step)
    animal = kinematic_replay_no_support.DrosophilaSimulation(
        container, 
        sim_options, 
        Kp=0.4, Kv=0.9,
        position_path=position_path,
        velocity_path=velocity_path,
        add_perturbation=args.perturbation
    )
    animal.run(optimization=False)
    animal.container.dump(
        dump_path="./basepositionrecorded", overwrite=True)