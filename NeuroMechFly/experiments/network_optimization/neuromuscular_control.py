"""Optimisation simulation"""
from NeuroMechFly.simulation.bullet_simulation import BulletSimulation
from NeuroMechFly.sdf.units import SimulationUnitScaling
from NeuroMechFly.container import Container
from NeuroMechFly.control.spring_damper_muscles import Parameters, SDAntagonistMuscle

import farms_pylog as pylog
import argparse
import numpy as np
import pybullet as p
import pybullet_data


class DrosophilaSimulation(BulletSimulation):
    """Drosophila Simulation Class. """

    def __init__(
        self,
        container,
        sim_options,
        units=SimulationUnitScaling(
            meters=1000,
            kilograms=1000)):
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
        for joint in self.actuated_joints:
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
                rest_pos=(lower_limit + upper_limit) * 0.5,
                flexor_mn=fmn,
                extensor_mn=emn,
                flexor_amp=fmn_amp,
                extensor_amp=fmn_amp,
            )

        ########## Initialize container ##########
        #: FIXME: This is
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
        p.setCollisionFilterPair(
            self.animal, self.plane, self.link_id['Head'], -1, 0
        )
        ########## DEBUG PARAMETER ##########
        self.debug = p.addUserDebugParameter('debug', -1, 1, 0.0)

        ########## Data variables ###########
        self.stability_coef = 0
        self.stance_count = 0
        self.last_draw = []
        self.check_is_all_legs = np.asarray(
            [False
             for leg in self.feet_links
             if "Tarsus5" in leg
             ]
        )

    def muscle_controller(self):
        """ Muscle controller. """
        for key, value in self.active_muscles.items():
            p.setJointMotorControl2(
                self.animal,
                self.joint_id[key],
                controlMode=p.TORQUE_CONTROL,
                force=value.compute_torque(only_passive=False)
            )

    def fixed_joints_controller(self):
        """Controller for fixed joints"""
        fixed_positions = {
            'joint_A3': -15, 'joint_A4': -15,
            'joint_A5': -15, 'joint_A6': -15, 'joint_Head': 10,
            'joint_LAntenna': 33, 'joint_RAntenna': -33,
            'joint_Rostrum': 90, 'joint_Haustellum': -60,
            'joint_LWing_roll': 90, 'joint_LWing_yaw': -17,
            'joint_RWing_roll': -90, 'joint_RWing_yaw': 17,
            'joint_LFCoxa_roll': 10, 'joint_RFCoxa_roll': -10,
            'joint_LFTarsus1': -46, 'joint_RFTarsus1': -46,
            'joint_LMCoxa_yaw': 2, 'joint_RMCoxa_yaw': 2,
            'joint_LMCoxa': -3, 'joint_RMCoxa': -3,
            'joint_LMTarsus1': -56, 'joint_RMTarsus1': -56,
            'joint_LHCoxa_yaw': 3, 'joint_RHCoxa_yaw': 3,
            'joint_LHCoxa': 11, 'joint_RHCoxa': 11,
            'joint_LHTarsus1': -50, 'joint_RHTarsus1': -50
        }
        for joint in range(self.num_joints):
            joint_name = [
                name for name,
                ind_num in self.joint_id.items() if joint == ind_num][0]
            if joint_name not in self.actuated_joints:
                try:
                    pos = np.deg2rad(fixed_positions[joint_name])
                except KeyError:
                    pos = 0

                p.setJointMotorControl2(
                    self.animal, joint,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=pos,
                    force=1e36)

    def controller_to_actuator(self, t):
        """ Implementation of abstractmethod. """
        self.muscle_controller()
        self.fixed_joints_controller()

        #: Change the color of the colliding body segments
        if self.draw_collisions:
            draw = []
            links_contact = np.where(
                np.linalg.norm(
                    self.ground_reaction_forces,
                    axis=1) > 0)[0]
            for i in links_contact:
                link1 = self.GROUND_CONTACTS[i][:-1]
                if link1 not in draw:
                    draw.append(link1)
                    self.change_color(link1 + '5', self.color_collision)
            for link in self.last_draw:
                if link not in draw:
                    self.change_color(link + '5', self.color_legs)
            self.last_draw = draw

    def change_color(self, id, color):
        """ Change color of a given body segment. """
        p.changeVisualShape(self.animal, self.link_id[id], rgbaColor=color)

    def feedback_to_controller(self):
        """ Implementation of abstractmethod. """
        pass

    def stance_polygon_dist(self):
        contact_segments = [
            leg for leg in self.feet_links if self.is_contact(leg)]
        contact_legs = []
        sum_x = 0
        sum_y = 0
        for seg in contact_segments:
            if seg[:2] not in contact_legs:
                contact_legs.append(seg[:2])
                pos_tarsus = self.get_link_position(seg[:2] + 'Tarsus5')
                sum_x += pos_tarsus[0]
                sum_y += pos_tarsus[1]

        self.stance_count += len(contact_legs)

        if len(contact_legs) > 2:
            if ('LM' in contact_legs or 'RM' in contact_legs):
                if (set(contact_legs) == set(['RM', 'LF', 'LH'])) or (
                        set(contact_legs) == set(['LM', 'RF', 'RH'])):
                    reward = 5
                else:
                    reward = 0
                poly_centroid = np.array(
                    [sum_x / len(contact_legs), sum_y / len(contact_legs)])
                body_centroid = np.array(self.get_link_position('Thorax')[:2])
                dist = np.linalg.norm(body_centroid - poly_centroid) - reward
            else:
                dist = 5.0
        else:
            dist = 5.5 - len(contact_legs) * 0.25

        return dist

    def is_using_all_legs(self):
        """Check if the fly uses all its legs to locomote"""
        contact_segments = [
            self.is_contact(leg)
            for leg in self.feet_links
            if "Tarsus5" in leg
        ]
        self.check_is_all_legs += np.asarray(contact_segments)

    def is_lava(self):
        """ State of lava approaching the model. """
        dist_traveled = -1 * self.ball_rotations[0]
        moving_limit = (self.TIME / self.RUN_TIME) * 3.24 - 0.30
        return dist_traveled < moving_limit

    def is_in_not_bounds(self):
        """ Bounds of the pelvis. """
        return (
            (self.distance_z * self.units.meters > 0.5) or
            (self.distance_y * self.units.meters > 10) or
            (self.distance_y * self.units.meters < -0.5)
        )

    def is_touch(self):
        """ Check if certain links touch. """
        return np.any(
            [
                self.is_contact(link)
                for link in self.link_id.keys()
                if 'Tarsus' not in link
            ]
        )

    def is_velocity_limit(self):
        """ Check velocity limits. """
        return np.any(
            np.array(self.joint_velocities) > 1e3
        )

    def is_flying(self):
        # FIXME: This function does two things at the same time
        dist_to_centroid = self.stance_polygon_dist()
        self.stability_coef += dist_to_centroid
        return dist_to_centroid > 5.0

    def optimization_check(self):
        """ Check optimization status. """
        lava = self.is_lava()
        flying = self.is_flying()
        velocity_cap = self.is_velocity_limit()
        touch = self.is_touch()
        self.is_using_all_legs()
        if lava or velocity_cap or touch or flying:
            pylog.debug(
                "Lava {} | Flying {} | Vel {} | Touch {}".format(
                    lava, flying, velocity_cap, touch
                )
            )
            return False
        return True

    def update_parameters(self, params):
        """ Implementation of abstractmethod. """
        parameters = self.container.neural.parameters
        N = int(self.controller.graph.number_of_nodes() / 4)
        edges_joints = int(self.controller.graph.number_of_nodes() / 3)
        edges_anta = int(self.controller.graph.number_of_nodes() / 12)

        opti_active_muscle_gains = params[:7 * N]
        opti_joint_phases = params[7 * N:7 * N + edges_joints]
        #opti_antagonist_phases = params[6*N+edges_joints:6*N+edges_joints+edges_anta]
        #opti_base_phases = params[6*N+edges_joints+edges_anta:]

        # opti_base_phases = params[5*N+edges_joints:]

        # print(
        #    "Opti active muscle gains {}".format(
        #        opti_active_muscle_gains
        #    )
        # )
        #print("Opti joint phases {}".format(opti_joint_phases))
        #print("Opti antagonist phases {}".format(opti_antagonist_phases))
        #print("Opti base phases {}".format(opti_base_phases))

        #: update active muscle parameters
        symmetry_joints = filter(
            lambda x: x.split('_')[1][0] != 'R', self.actuated_joints
        )

        for j, joint in enumerate(symmetry_joints):
            #print(joint,joint.replace('L', 'R', 1),6*j,6*(j+1))
            # print(joint, Parameters(*opti_active_muscle_gains[7*j:7*(j+1)]))
            self.active_muscles[joint.replace('L', 'R', 1)].update_parameters(
                Parameters(*opti_active_muscle_gains[7 * j:7 * (j + 1)])
            )
            #: It is important to mirror the joint angles for rest position
            #: especially for coxa
            if "Coxa_roll" in joint:
                opti_active_muscle_gains[(7 * j) + 4] *= -1
            self.active_muscles[joint].update_parameters(
                Parameters(*opti_active_muscle_gains[7 * j:7 * (j + 1)])
            )
        #: Update phases
        #: Edges to set phases for
        phase_edges = [
            ['Coxa', 'Femur'],
            ['Femur', 'Tibia'],
        ]

        for side in ('L', 'R'):
            for j0, pos in enumerate(('F', 'M', 'H')):
                if pos != 'F':
                    coxa_label = 'Coxa_roll'
                else:
                    coxa_label = 'Coxa'
                for j1, ed in enumerate(phase_edges):
                    if ed[0] == 'Coxa':
                        from_node = coxa_label
                    else:
                        from_node = ed[0]
                    to_node = ed[1]
                    for j2, action in enumerate(('flexion', 'extension')):
                        node_1 = "joint_{}{}{}_{}".format(
                            side, pos, from_node, action)
                        node_2 = "joint_{}{}{}_{}".format(
                            side, pos, to_node, action)
                        parameters.get_parameter(
                            'phi_{}_to_{}'.format(node_1, node_2)
                        ).value = opti_joint_phases[4 * j0 + 2 * j1 + j2]
                        parameters.get_parameter(
                            'phi_{}_to_{}'.format(node_2, node_1)
                        ).value = -1 * opti_joint_phases[4 * j0 + 2 * j1 + j2]
        '''
        if len(params)>75:
            opti_base_phases = params[5*N+edges_joints+edges_anta:]
            coxae_edges =[
                 ['LFCoxa', 'RFCoxa'],
                 ['LFCoxa', 'RMCoxa_roll'],
                 ['RMCoxa_roll', 'LHCoxa_roll'],
                 ['RFCoxa', 'LMCoxa_roll'],
                 ['LMCoxa_roll', 'RHCoxa_roll']
             ]

            for j1, ed in enumerate(coxae_edges):
                for j2, action in enumerate(('flexion', 'extension')):
                    node_1 = "joint_{}_{}".format(ed[0], action)
                    node_2 = "joint_{}_{}".format(ed[1], action)
                    #print(node_1, node_2, j1)
                    parameters.get_parameter(
                        'phi_{}_to_{}'.format(node_1, node_2)
                    ).value = opti_base_phases[j1]
                    parameters.get_parameter(
                        'phi_{}_to_{}'.format(node_2, node_1)
                    ).value = -1*opti_base_phases[j1]
        '''

def read_optimization_results(fun, var):
    """ Read optimization results. """
    return (np.loadtxt(fun), np.loadtxt(var))


def parse_args():
    """Argument parser"""
    parser = argparse.ArgumentParser(
        description='Neuromechfly simulation of evolution results',
        formatter_class=(
            lambda prog:
            argparse.HelpFormatter(prog, max_help_position=50)
        ),
    )
    parser.add_argument(
        '--output_fun',
        type=str,
        default='FUN.txt',
        help='Results output of functions',
    )
    parser.add_argument(
        '--output_var',
        type=str,
        default='VAR.txt',
        help='Results output of variables',
    )
    parser.add_argument(
        '--runtime',
        type=float,
        default=2.,
        help='Simulation run time',
    )
    parser.add_argument(
        '--timestep',
        type=float,
        default=0.001,
        help='Simulation timestep',
    )
    return parser.parse_args()


def main():
    """ Main """

    clargs = parse_args()

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
        "model": "../../data/design/sdf/neuromechfly_limitsFromData_minMax.sdf",
        "model_offset": [0., 0., 11.2e-3],
        "run_time": clargs.runtime,
        "pose": '../../data/config/pose/test_pose_tripod.yaml',
        "base_link": 'Thorax',
        "controller": '../../data/config/network/locomotion_ball.graphml',
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

    container = Container(clargs.runtime / clargs.timestep)
    animal = DrosophilaSimulation(container, sim_options)

    '''
    fun, var = read_optimization_results(
        "./optimization_results/"+exp+"/FUN."+gen,
        "./optimization_results/"+exp+"/VAR."+gen
    )
    '''

    fun, var = read_optimization_results(
        "./FUN.ged3",
        "./VAR.ged3"
    )
    '''
    fun, var = read_optimization_results(
        "./FUN.txt",
        "./VAR.txt",
    )
    '''

    params = var[np.argmax(fun[:, 0] * fun[:, 1])]
    params = np.array(params)
    animal.update_parameters(params)

    animal.run(optimization=False)
    animal.container.dump(
        dump_path=f"./optimization_{exp}_gen_{gen}",
        overwrite=True
        )

if __name__ == '__main__':
    main()