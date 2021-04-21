from NeuroMechFly.experiments.network_optimization import neuromuscular_control
from NeuroMechFly.container import Container
import numpy as np 
    
if __name__ == "__main__":
    """ Main """
    run_time = 5.
    time_step = 0.001

    side = ['L', 'R']
    pos = ['F', 'M', 'H']
    leg_segments = ['Femur', 'Tibia'] + \
        ['Tarsus' + str(i) for i in range(1, 6)]

    ground_contact = [
        s +
        p +
        name for s in side for p in pos for name in leg_segments if 'Tarsus' in name]

    left_front_leg = ['LF' + name for name in leg_segments]
    left_middle_leg = ['LM' + name for name in leg_segments]
    left_hind_leg = ['LH' + name for name in leg_segments]

    right_front_leg = ['RF' + name for name in leg_segments]
    right_middle_leg = ['RM' + name for name in leg_segments]
    right_hind_leg = ['RH' + name for name in leg_segments]

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

    gen = '10'
    exp = 'run_Drosophila_var_71_obj_2_pop_20_gen_100_0407_1744'

    sim_options = {
        "headless": False,
        # Scaled SDF model
        "model": "../data/design/sdf/neuromechfly_limitsFromData_minMax.sdf",
        "model_offset": [0., 0., 11.2e-3],
        "run_time": run_time,
        "pose": '../data/config/pose/test_pose_tripod.yaml',
        "base_link": 'Thorax',
        "controller": '../data/config/network/locomotion_ball.graphml',
        "ground_contacts": ground_contact,
        'self_collisions': self_collision,
        "draw_collisions": True,
        "record": False,
        'camera_distance': 3.5,
        'track': False,
        'moviename': 'stability_' + exp + '_gen_' + gen + '.mp4',
        'moviefps': 50,
        'slow_down': True,
        'sleep_time': 0.001,
        'rot_cam': False,
        'ground': 'ball'
    }

    container = Container(run_time / time_step)
    animal = neuromuscular_control.DrosophilaSimulation(container, sim_options)

    fun, var = np.loadtxt("./FUN.txt"), np.loadtxt("./VAR.txt")

    params = var[np.argmax(fun[:, 0] * fun[:, 1])]
    params = np.array(params)
    animal.update_parameters(params)

    animal.run(optimization=False)
    animal.container.dump(
        dump_path=f"./optimization_{exp}_gen_{gen}",
        overwrite=True
        )