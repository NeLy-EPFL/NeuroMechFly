from NeuroMechFly.experiments.kinematic_replay import kinematic_replay
from NeuroMechFly.container import Container

if __name__ == "__main__":
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
        "model": "../data/design/sdf/neuromechfly_noLimits.sdf",
        "pose": '../data/config/pose/pose_optimization_2.yaml',
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
    animal = kinematic_replay.DrosophilaSimulation(container, sim_options, Kp=0.4, Kv=0.9)
    animal.run(optimization=False)
    animal.container.dump(
        dump_path=f"./kinematic_replay_{behavior}",
        overwrite=False)