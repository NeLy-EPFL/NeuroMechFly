"""Optimisation simulation"""

import os
import pickle
import time
import argparse

import farms_pylog as pylog
import numpy as np
import pandas as pd
from IPython import embed
import matplotlib.pyplot as plt
import pybullet as p
import pybullet_data
from bullet_simulation_opt import BulletSimulation
from NeuroMechFly.container import Container
from NeuroMechFly.sdf.units import SimulationUnitScaling
from spring_damper_muscles import Parameters, SDAntagonistMuscle


class DrosophilaSimulation(BulletSimulation):
    """Drosophila Simulation Class
    """

    def __init__(self, container, sim_options, units=SimulationUnitScaling(meters=1000,kilograms=1000)):
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
            '{}{}Tarsus{}'.format(side, pos,seg)
            for side in self.sides
            for pos in self.positions
            for seg in range(1,6)
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
                rest_pos = (lower_limit + upper_limit)*0.5,
                flexor_mn=fmn,
                extensor_mn=emn,
                flexor_amp=fmn_amp,
                extensor_amp=fmn_amp,
            )

        ########## Initialize container ##########
        #: FUCK : THIS IS BAD!!!
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
        #p.setCollisionFilterPair(
        #    self.animal, self.plane, self.link_id['Head'], -1, 0
        #)
        ########## DEBUG PARAMETER ##########
        self.debug = p.addUserDebugParameter('debug', -1, 1, 0.0)

        ########## Data variables ###########
        self.torques=[]
        self.grf=[]
        self.collision_forces=[]
        self.ball_rot=[]
        self.stability_coef = 0
        self.stance_count = 0
        self.lastDraw=[]
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
            for joint in range(self.num_joints):
                joint_name = [name for name, ind_num in self.joint_id.items() if joint == ind_num][0]
                if joint_name not in self.actuated_joints:# and 'support' not in joint_name:
                    if joint_name == 'joint_A3' or joint_name == 'joint_A4' or joint_name == 'joint_A5' or joint_name == 'joint_A6':
                        pos = np.deg2rad(-15)
                    elif joint_name == 'joint_LAntenna':
                        pos = np.deg2rad(35)
                    elif joint_name == 'joint_RAntenna':
                        pos = np.deg2rad(-35)
                    elif joint_name == 'joint_Rostrum' or joint_name == 'joint_LWing_roll':
                        pos = np.deg2rad(90)
                    elif joint_name == 'joint_Haustellum':
                        pos = np.deg2rad(-60)
                    elif joint_name == 'joint_RWing_roll':
                        pos = np.deg2rad(-90)
                    elif joint_name == 'joint_LWing_yaw':
                        pos = np.deg2rad(-17)
                    elif joint_name == 'joint_RWing_yaw':
                        pos = np.deg2rad(17)
                    elif joint_name == 'joint_Head':
                        pos = np.deg2rad(10)
                    #elif joint_name == 'joint_LFCoxa_yaw':
                    #    pos = np.deg2rad(-5)
                    #elif joint_name == 'joint_RFCoxa_yaw':
                    #    pos = np.deg2rad(5)
                    elif joint_name == 'joint_LFCoxa_roll':
                        pos = np.deg2rad(12)
                    elif joint_name == 'joint_RFCoxa_roll':
                        pos = np.deg2rad(-12)
                    #elif joint_name == 'joint_LFFemur_roll':
                    #    pos = np.deg2rad(-26)
                    #elif joint_name == 'joint_RFFemur_roll':
                    #    pos = np.deg2rad(26)
                    elif joint_name == 'joint_LFTarsus1':
                        pos = np.deg2rad(-46) #43
                    elif joint_name == 'joint_RFTarsus1':
                        pos = np.deg2rad(-46) #49
                    elif joint_name == 'joint_LMCoxa_yaw':
                        pos = np.deg2rad(2) #4
                    elif joint_name == 'joint_RMCoxa_yaw':
                        pos = np.deg2rad(2) #4
                    elif joint_name == 'joint_LMCoxa':
                        pos = np.deg2rad(-3) #-2
                    elif joint_name == 'joint_RMCoxa':
                        pos = np.deg2rad(-3) #-4.5
                    #elif joint_name == 'joint_LMFemur_roll':
                    #    pos = np.deg2rad(-7)
                    #elif joint_name == 'joint_RMFemur_roll':
                    #    pos = np.deg2rad(7)
                    elif joint_name == 'joint_LMTarsus1':
                        pos = np.deg2rad(-56) #-52
                    elif joint_name == 'joint_RMTarsus1':
                        pos = np.deg2rad(-56)
                    elif joint_name == 'joint_LHCoxa_yaw':
                        pos = np.deg2rad(3)
                    elif joint_name == 'joint_RHCoxa_yaw':
                        pos = np.deg2rad(3) #6.2
                    elif joint_name == 'joint_LHCoxa':
                        pos = np.deg2rad(11) #13
                    elif joint_name == 'joint_RHCoxa':
                        pos = np.deg2rad(11)
                    #elif joint_name == 'joint_LHFemur_roll':
                    #    pos = np.deg2rad(9)
                    #elif joint_name == 'joint_RHFemur_roll':
                    #    pos = np.deg2rad(-9)
                    elif joint_name == 'joint_LHTarsus1':
                        pos = np.deg2rad(-50) #-45
                    elif joint_name == 'joint_RHTarsus1':
                        pos = np.deg2rad(-50)
                    else:
                        pos = 0


                    #print('lava: ',self.is_lava())
                    #print('fly: ',self.is_flying())
                    #print('touch: ',self.is_touch())
                    #print('Ball rotations: ', self.ball_rotations(),'\n')
                    #print('Dist stance_polygon:', self.stance_polygon_dist())
                    #print(np.array(self.get_link_position('Thorax')[:2]))


                    p.setJointMotorControl2(
                        self.animal, joint,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=pos,
                        force=1e36)


    def controller_to_actuator(self, t):
        """ Implementation of abstractmethod. """
        self.muscle_controller()
        self.fixed_joints_controller()

        jointTorques = np.array(self.joint_torques())
        #print(jointTorques.shape)
        self.torques.append(jointTorques)


        grf = self.ball_reaction_forces()
        if self.draw_collisions:
            ind = np.where(np.array(grf).transpose()[0]>0)[0]
            draw=[]
            for i in ind:
                link1 = self.ground_contacts[i][:-1]
                if link1 not in draw:
                    draw.append(link1)
                    p.changeVisualShape(self.animal, self.link_id[link1+'5'],rgbaColor=self.color_collision)
            for link in self.lastDraw:
                if link not in draw:
                    p.changeVisualShape(self.animal, self.link_id[link+'5'], rgbaColor=self.color_legs)

            self.lastDraw = draw
        self.grf.append(grf)

        ball_rot = np.array(self.ball_rotations())
        ball_rot[:2] = ball_rot[:2]*self.ball_radius*10 # Distance in mm
        self.ball_rot.append(ball_rot)
        #print(ball_rot)


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
        contact_segments = [leg for leg in self.feet_links if self.is_contact_ball(leg)]
        contact_legs = []
        sum_x = 0
        sum_y = 0
        #print(contact_segments)
        for seg in contact_segments:
            if seg[:2] not in contact_legs:
                contact_legs.append(seg[:2])
                pos_tarsus = self.get_link_position(seg[:2]+'Tarsus5')
                sum_x += pos_tarsus[0]
                sum_y += pos_tarsus[1]

        #print(contact_legs)

        self.stance_count += len(contact_legs)

        if len(contact_legs)>2:
            if 'LM' in contact_legs or 'RM' in contact_legs:
                if (set(contact_legs) == set(['RM','LF','LH'])) or (set(contact_legs) == set(['LM','RF','RH'])):
                    penalty = 0
                else:
                    penalty = 0
                poly_centroid = np.array([sum_x/len(contact_legs),sum_y/len(contact_legs)])
                body_centroid = np.array(self.get_link_position('Thorax')[:2])
                dist = np.linalg.norm(body_centroid-poly_centroid) + penalty
            else:
                dist = 5.0
        else:
            dist = 6.0 - len(contact_legs)*0.25

        return dist

    def is_using_all_legs(self):
        """Check if the fly uses all its legs to locomote"""
        contact_segments = [
            self.is_contact_ball(leg)
            for leg in self.feet_links
            if "Tarsus5" in leg
        ]
        self.check_is_all_legs += np.asarray(contact_segments)

    def is_lava(self):
        """ State of lava approaching the model. """
        #return (self.distance_y < (((self.TIME)/self.RUN_TIME)*2)-0.25)
        ball_rot = np.array(self.ball_rotations())
        dist_traveled = -ball_rot[0]
        #print("BALL ROTATION", ball_rot)
        moving_limit = (((self.time)/self.run_time)*5.24)-0.40
        #print(ball_rot)
        return dist_traveled < moving_limit


    def is_in_not_bounds(self):
        """ Bounds of the pelvis. """
        return (
            (self.distance_z > 0.5) or \
            (self.distance_y > 10) or \
            (self.distance_y < -0.5)
        )

    def is_touch(self):
        """ Check if certain links touch. """
        return np.any(
            [
                self.is_contact_ball(link)
                for link in self.link_id.keys()
                if 'Tarsus' not in link
            ]
        )

    def is_velocity_limit(self):
        """ Check velocity limits. """
        return np.any(
            np.array(self.joint_velocities) > 5e3
        )

    def is_flying(self):
        # FIXME: This function does two things at the same time
        dist_to_centroid = self.stance_polygon_dist()
        self.stability_coef += dist_to_centroid
        # print(dist_to_centroid)
        return dist_to_centroid > 5.5

    def optimization_check(self):
        """ Check optimization status. """
        lava = self.is_lava()
        flying = self.is_flying()
        velocity_cap = self.is_velocity_limit()
        touch = self.is_touch()
        self.is_using_all_legs()
        if lava or velocity_cap or flying or touch:
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
        N = int(self.controller.graph.number_of_nodes()/4)
        edges_joints = int(self.controller.graph.number_of_nodes()/3)
        edges_anta = int(self.controller.graph.number_of_nodes()/12)

        opti_active_muscle_gains = params[:5*N]
        opti_joint_phases = params[5*N:5*N+edges_joints]
        #opti_antagonist_phases = params[6*N+edges_joints:6*N+edges_joints+edges_anta]
        #opti_base_phases = params[6*N+edges_joints+edges_anta:]

        opti_base_phases = params[5*N+edges_joints:]

        #print(
        #    "Opti active muscle gains {}".format(
        #        opti_active_muscle_gains
        #    )
        #)
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
                Parameters(*opti_active_muscle_gains[5*j:5*(j+1)])
            )
            #: It is important to mirror the joint angles for rest position
            #: especially for coxa
            if "Coxa_roll" in joint:
                opti_active_muscle_gains[(5*j)+0] *= -1
                opti_active_muscle_gains[(5*j)+4] *= -1
            self.active_muscles[joint].update_parameters(
                Parameters(*opti_active_muscle_gains[5*j:5*(j+1)])
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
                        node_1 = "joint_{}{}{}_{}".format(side, pos, from_node, action)
                        node_2 = "joint_{}{}{}_{}".format(side, pos, to_node, action)
                        #print(node_1, node_2, 4*j0 + 2*j1 + j2)
                        #print(len(opti_joint_phases))
                        parameters.get_parameter(
                            'phi_{}_to_{}'.format(node_1, node_2)
                        ).value = opti_joint_phases[4*j0 + 2*j1 + j2]
                        parameters.get_parameter(
                            'phi_{}_to_{}'.format(node_2, node_1)
                        ).value = -1*opti_joint_phases[4*j0 + 2*j1 + j2]

                #node_1 = "joint_{}{}{}_{}".format(side, pos, coxa_label, 'flexion')
                #node_2 = "joint_{}{}{}_{}".format(side, pos, coxa_label, 'extension')
                ##print(node_1,node_2)
                ##print()
                #parameters.get_parameter(
                #    'phi_{}_to_{}'.format(node_1, node_2)
                #).value = opti_antagonist_phases[j0]
                #parameters.get_parameter(
                #    'phi_{}_to_{}'.format(node_2, node_1)
                #).value = -1*opti_antagonist_phases[j0]


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


def read_optimization_results(fun, var):
    """ Read optimization results. """
    return (np.loadtxt(fun), np.loadtxt(var))

def save_data(fly, filename, exp=''):
    torques_dict = {}
    grf_dict = {}
    collisions_dict = {}
    data_torque = np.array(fly.torques).transpose()
    data_grf = np.array(fly.grf).transpose((1, 0, 2))
    #data_collisions = np.array(fly.collision_forces).transpose((1, 0, 2))
    currentDirectory = os.getcwd()

    #for i, joint in enumerate(fly.joint_id.keys()):
    #    torques_dict[joint] = data_torque[i]

    for i, joint in enumerate(fly.ground_contacts):
        grf_dict[joint] = data_grf[i]

    path_muscles = os.path.join(currentDirectory,'Results/muscle/outputs.h5')
    path_joint_pos = os.path.join(currentDirectory,'Results/physics/joint_positions.h5')

    path_angles = os.path.join(currentDirectory,'Output_data','angles',exp)
    path_act = os.path.join(currentDirectory,'Output_data','muscles',exp)
    path_grf = os.path.join(currentDirectory,'Output_data','grf',exp)
    path_ball_rot = os.path.join(currentDirectory,'Output_data','ballRotations',exp)

    if not os.path.exists(path_angles):
        os.makedirs(path_angles)
    if not os.path.exists(path_act):
        os.makedirs(path_act)
    if not os.path.exists(path_grf):
        os.makedirs(path_grf)
    if not os.path.exists(path_ball_rot):
        os.makedirs(path_ball_rot)

    #with open(path_torque+'/torques_'+filename,'wb') as f:
    #    pickle.dump(torques_dict,f)

    with open(path_grf+'/grf_'+filename+'.pkl','wb') as f:
        pickle.dump(grf_dict,f)

    with open(path_ball_rot+'/ballRot_'+filename+'.pkl','wb') as f:
        pickle.dump(fly.ball_rot,f)

    muscles_data = pd.read_hdf(path_muscles)
    muscles_data.to_hdf(path_act+'/outputs_'+filename+'.h5','muscles',mode='w')

    angles_data = pd.read_hdf(path_joint_pos)
    angles_data.to_hdf(path_angles+'/jointpos_'+filename+'.h5','angles',mode='w')


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
        default=5.,
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
    from sklearn.preprocessing import normalize

    clargs = parse_args()

    side = ['L','R']
    pos = ['F','M','H']
    leg_segments = ['Femur','Tibia']+['Tarsus' + str(i) for i in range(1, 6)]

    ground_contact = [s+p+name for s in side for p in pos for name in leg_segments if 'Tarsus' in name]

    left_front_leg = ['LF'+name for name in leg_segments]
    left_middle_leg = ['LM'+name for name in leg_segments]
    left_hind_leg = ['LH'+name for name in leg_segments]

    right_front_leg = ['RF'+name for name in leg_segments]
    right_middle_leg = ['RM'+name for name in leg_segments]
    right_hind_leg = ['RH'+name for name in leg_segments]

    body_segments = ['A1A2','A3','A4','A5','A6','Thorax','Head']

    self_collision = []
    for link0 in left_front_leg:
        for link1 in left_middle_leg:
            self_collision.append([link0,link1])
    for link0 in left_middle_leg:
        for link1 in left_hind_leg:
            self_collision.append([link0,link1])
    for link0 in left_front_leg:
        for link1 in body_segments:
            self_collision.append([link0,link1])
    for link0 in left_middle_leg:
        for link1 in body_segments:
            self_collision.append([link0,link1])
    for link0 in left_hind_leg:
        for link1 in body_segments:
            self_collision.append([link0,link1])

    for link0 in right_front_leg:
        for link1 in right_middle_leg:
            self_collision.append([link0,link1])
    for link0 in right_middle_leg:
        for link1 in right_hind_leg:
            self_collision.append([link0,link1])
    for link0 in right_front_leg:
        for link1 in body_segments:
            self_collision.append([link0,link1])
    for link0 in right_middle_leg:
        for link1 in body_segments:
            self_collision.append([link0,link1])
    for link0 in right_hind_leg:
        for link1 in body_segments:
            self_collision.append([link0,link1])

    gen = '70'
    exp = 'run_Drosophila_var_62_obj_2_0423_1801'
    exp = 'run_Drosophila_var_62_obj_2_0426_1221'

    sim_options = {
        "headless": False,
        # Scaled SDF model
        "model": "../../design/sdf/neuromechfly_limitsFromData_minMax.sdf",
        "model_offset": [0., 0., 11.2e-3],
        "run_time": clargs.runtime,
        "pose": '../../config/test_pose_tripod.yaml',
        "base_link": 'Thorax',
        "controller": '../../config/locomotion_ball.graphml',
        "ground_contacts": ground_contact,
        'self_collisions':self_collision,
        "draw_collisions": True,
        "record": False,
        'camera_distance': 4.5,
        'track': False,
        'moviename': 'stability_'+exp+'_gen_'+gen+'.mp4',
        'moviespeed': 0.1,
        'slow_down': False,
        'sleep_time': 10.0,
        'rot_cam': True
        }

    container = Container(clargs.runtime/clargs.timestep)
    animal = DrosophilaSimulation(container, sim_options)

    #: read results

    #fun, var = read_optimization_results(
    #    "./sim_files/fun/FUN_last_good.ged3",
    #    "./sim_files/var/VAR_last_good.ged3"
    #)
    
    fun, var = read_optimization_results(
        "./optimization_results/"+exp+"/FUN."+gen,
        "./optimization_results/"+exp+"/VAR."+gen
    )
    '''

    fun, var = read_optimization_results(
        "./release/run_Drosophila_var_80_obj_2_pop_100_gen_30_0411_1725/FUN.29",
        "./release/run_Drosophila_var_80_obj_2_pop_100_gen_30_0411_1725/VAR.29",
    )

    fun, var = read_optimization_results(
        "./FUN.txt",
        "./VAR.txt",
    )
    '''
    # fun, var = read_optimization_results(
    #     "./optimization_results/run_Drosophila_var_80_obj_2_pop_10_gen_4_0412_0316/FUN.3",
    #     "./optimization_results/run_Drosophila_var_80_obj_2_pop_10_gen_4_0412_0316/VAR.3",
    # )
    fun_normalized = normalize(fun)
    params = var[np.argmin(fun_normalized[:,0]*fun_normalized[:,1])]
    params = var[np.argmin(fun_normalized[:,0])]

    params = var[np.argmax(fun[:,0]*fun[:,1])]
    ind = np.argmax(fun[:,0]*fun[:,1])
    #params = var[np.argmin(fun[:,1])]
    params = np.array(params)
    print(params)
    animal.update_parameters(params)

    animal.run(optimization=False)
    animal.container.dump(overwrite=True)

    name_data = 'optimization_gen_'+ gen

    save_data(animal,name_data,exp)
    
    colors =    fun[:,0]*fun[:,1]
    plt.scatter(fun[:,0], fun[:,1], c=colors, cmap=plt.cm.winter)
    plt.scatter(fun[ind,0], fun[ind,1], c=colors, cmap=plt.cm.winter)
    plt.xlabel('Distance (negative)')
    plt.ylabel('Stability')
    # plt.savefig('./{}_generation_{}.pdf'.format(exp, gen))
    plt.show()

if __name__ == '__main__':
    main()
