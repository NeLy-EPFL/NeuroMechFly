from bullet_simulation_opt import BulletSimulation
from IPython import embed
import pybullet as p
import pybullet_data
import pandas as pd
import numpy as np
import time
import pickle
import farms_pylog as pylog
import numpy as np
from farms_container import Container
import os
from spring_damper_muscles import SDAntagonistMuscle, Parameters
from farms_sdf.units import SimulationUnitScaling
pylog.set_level('error')

class DrosophilaSimulation(BulletSimulation):
    """Drosophila Simulation Class
    """

    def __init__(self, container, sim_options, units=SimulationUnitScaling()):
        ########## Container ##########
        container.add_namespace('muscle')
        container.muscle.add_table('parameters', table_type='CONSTANT')
        container.muscle.add_table('outputs')
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
                maxJointVelocity=1000.0
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
                #else:
                #    pos = 0
                elif joint_name == 'joint_LFCoxa_roll':
                    pos = np.deg2rad(-25)
                elif joint_name == 'joint_RFCoxa_roll':
                    pos = np.deg2rad(25)
                elif joint_name == 'joint_LAntenna':
                    pos = np.deg2rad(25)
                elif joint_name == 'joint_RAntenna':
                    pos = np.deg2rad(-25)
                elif joint_name == 'joint_Proboscis' or joint_name == 'joint_RWing_roll':
                    pos = np.deg2rad(90)
                elif joint_name == 'joint_Labellum':
                    pos = np.deg2rad(60)
                elif joint_name == 'joint_LWing_roll':
                    pos = np.deg2rad(-90)
                elif joint_name == 'joint_LWing_yaw':
                    pos = np.deg2rad(-17)
                elif joint_name == 'joint_RWing_yaw':
                    pos = np.deg2rad(17)
                elif 'FTarsus1' in joint_name:
                    pos = np.deg2rad(-24)
                elif 'MTarsus1' in joint_name:
                    pos = np.deg2rad(-28)
                elif 'HTarsus1' in joint_name:
                    pos = np.deg2rad(-22)
                else:
                    pos = 0

                #print('lava: ',self.is_lava())
                #print('fly: ',self.is_flying())
                #print('touch: ',self.is_touch())
                #print('Ball rotations: ', self.ball_rotations(),'\n')
                
                p.setJointMotorControl2(
                    self.animal, joint,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=pos,
                    force=1e36)
        

    def controller_to_actuator(self, t):
        """ Implementation of abstractmethod. """
        self.muscle_controller()
        self.fixed_joints_controller()

        if t%10 == 0:
            jointTorques = np.array(self.joint_torques())
            #print(jointTorques.shape)
            self.torques.append(jointTorques)

        if t%10 == 0:
            grf = self.ball_reaction_forces()
            #print(grf.shape)
            self.grf.append(grf)

        if t%10 == 0:
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

    def is_lava(self):
        """ State of lava approaching the model. """
        #return (self.distance_y < (((self.TIME)/self.RUN_TIME)*2)-0.25)
        ball_rot = np.array(self.ball_rotations())
        dist_traveled = -ball_rot[0]*self.ball_radius # Distance in mm
        #print(dist_traveled)
        moving_limit = (((self.TIME)/self.RUN_TIME)*3*np.pi)-0.5*np.pi
        return dist_traveled < moving_limit
        #return (dist_traveled < (((self.TIME)/self.RUN_TIME)*2)-0.25)
    

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
                self.is_contact_ball(link) for link in self.link_id.keys() if 'Tarsus' not in link
                #self.is_contact(link) for link in [
                #    'LTibia', 'RTibia', 'LRadius', 'RRadius'
                #]
            ]
        )

    def is_velocity_limit(self):
        """ Check velocity limits. """
        return np.any(np.array(self.joint_velocities) > 1000)

    def is_flying(self):
        """Check if no leg of the model is in contact """
        contact_segments = [leg for leg in self.feet_links if self.is_contact_ball(leg)]
        num_legs = 0
        contact_legs = []
        for seg in contact_segments:
            if seg[1] not in contact_legs:
                contact_legs.append(seg[1])
                num_legs += 1
        #print(contact_legs)
        #print(num_legs < 2)
        return num_legs < 2
        #return not(
        #    np.any([self.is_contact_ball(leg) for leg in self.feet_links])
        #)

    def optimization_check(self):
        """ Check optimization status. """
        lava = self.is_lava()
        #bounding_box = self.is_in_not_bounds()
        flying = self.is_flying()
        velocity_cap = self.is_velocity_limit()
        touch = self.is_touch()
        if lava or velocity_cap or touch or flying:
            print(
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

        opti_active_muscle_gains = params[:6*N]
        opti_joint_phases = params[6*N:6*N+edges_joints]
        opti_antagonist_phases = params[6*N+edges_joints:6*N+edges_joints+edges_anta]
        opti_base_phases = params[6*N+edges_joints+edges_anta:]
        pylog.debug(
            "Opti active muscle gains {}".format(
                opti_active_muscle_gains
            )
        )
        pylog.debug("Opti joint phases {}".format(opti_joint_phases))
        pylog.debug("Opti antagonist phases {}".format(opti_antagonist_phases))
        pylog.debug("Opti base phases {}".format(opti_base_phases))

        #: update active muscle parameters
        symmetry_joints = filter(
            lambda x: x.split('_')[1][0] != 'R', self.actuated_joints
        )
        for j, joint in enumerate(symmetry_joints):
            self.active_muscles[joint].update_parameters(
                Parameters(*opti_active_muscle_gains[6*j:6*(j+1)])
            )
            self.active_muscles[joint.replace('L', 'R', 1)].update_parameters(
                Parameters(*opti_active_muscle_gains[6*j:6*(j+1)])
            )

        #: Update phases
        #: Edges to set phases for
        phase_edges = [
            ['Coxa', 'Femur'],
            ['Femur', 'Tibia'],
        ]
        
        for side in ('L', 'R'):
            n_leg = 0
            for pos in ('F', 'M', 'H'):
                for j1, ed in enumerate(phase_edges):
                    if ('Coxa' in ed[0]) and ((pos == 'M') or (pos == 'H')):
                        ed[0] = 'Coxa_roll'
                    if ('Coxa' in ed[0]) and (pos == 'F'):
                        ed[0] = 'Coxa'     
                    for j2, action in enumerate(('flexion', 'extension')):
                        node_1 = "joint_{}{}{}_{}".format(side, pos, ed[0], action)
                        node_2 = "joint_{}{}{}_{}".format(side, pos, ed[1], action)
                        parameters.get_parameter(
                            'phi_{}_to_{}'.format(node_1, node_2)
                        ).value = opti_joint_phases[2*j1 + j2]
                        parameters.get_parameter(
                            'phi_{}_to_{}'.format(node_2, node_1)
                        ).value = -1*opti_joint_phases[2*j1 + j2]
        
                if pos != 'F':
                    coxa_label = 'Coxa_roll'
                else:
                    coxa_label = 'Coxa'
                node_1 = "joint_{}{}{}_{}".format(side, pos, coxa_label, 'flexion')
                node_2 = "joint_{}{}{}_{}".format(side, pos, coxa_label, 'extension')
                parameters.get_parameter(
                    'phi_{}_to_{}'.format(node_1, node_2)
                ).value = opti_antagonist_phases[n_leg]
                parameters.get_parameter(
                    'phi_{}_to_{}'.format(node_2, node_1)
                ).value = -1*opti_antagonist_phases[n_leg]
                n_leg += 1

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
                parameters.get_parameter(
                    'phi_{}_to_{}'.format(node_1, node_2)
                ).value = opti_base_phases[2*j1 + j2]
                parameters.get_parameter(
                    'phi_{}_to_{}'.format(node_2, node_1)
                ).value = -1*opti_base_phases[2*j1 + j2]
        

def read_optimization_results(fun, var):
    """ Read optimization results. """
    return (np.loadtxt(fun), np.loadtxt(var))

def save_data(fly, filename):
    torques_dict = {}
    grf_dict = {}
    collisions_dict = {}
    data_torque = np.array(fly.torques).transpose()
    data_grf = np.array(fly.grf).transpose((1, 0, 2))
    #data_collisions = np.array(fly.collision_forces).transpose((1, 0, 2))
    currentDirectory = os.getcwd()

    for i, joint in enumerate(fly.joint_id.keys()):
        torques_dict[joint] = data_torque[i]

    for i, joint in enumerate(fly.GROUND_CONTACTS):
        grf_dict[joint] = data_grf[i]

    path_torque = os.path.join(currentDirectory,'Results','torques_'+filename)
    path_grf = os.path.join(currentDirectory,'Results','grf_'+filename)
    path_ball_rot = os.path.join(currentDirectory,'ball_data','ballRot_'+filename)

    with open(path_torque,'wb') as f:
        pickle.dump(torques_dict,f)

    with open(path_grf,'wb') as f:
        pickle.dump(grf_dict,f)

    with open(path_ball_rot,'wb') as f:
        pickle.dump(fly.ball_rot,f)

def main():
    """ Main """
    run_time = 5.
    time_step = 0.001

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
    
    sim_options = {
        "headless": False,
        "model": "../design/sdf/drosophila_100x_Limits_strict_offset.sdf",
        "model_offset": [0., 0., 1.12],
        "run_time": run_time,
        "pose": '../config/pose.yaml',
        "base_link": 'Thorax',
        "controller": '../config/locomotion_ball.graphml',
        "ground_contacts": ground_contact,
        'self_collisions':self_collision,
        "record": False,
        'camera_distance': 0.5,
        'track': False,
        'moviename': 'neuroOpt_walking_New_Opt.mp4',
        'slow_down': True,
        'sleep_time': 0.001,
        'rot_cam': False
        }
    container = Container(run_time/time_step)
    animal = DrosophilaSimulation(container, sim_options)

    #: read results
    
    #fun, var = read_optimization_results(
    #    "./sim_files/fun/FUN_last_good.ged3",
    #    "./sim_files/var/VAR_last_good.ged3"
    #)
    #fun, var = read_optimization_results(
    #    "./optimization_results/run_Drosophila_var_79_obj_2_pop_20_gen_200_18_01_16/FUN.199",
    #    "./optimization_results/run_Drosophila_var_79_obj_2_pop_20_gen_200_18_01_16/VAR.199"
    #)
    fun, var = read_optimization_results(
        "./FUN.ged3",
        "./VAR.ged3"
    )
    
    params = var[np.argmin(fun[:, 0])]
    animal.update_parameters(params)

    animal.run(optimization=False)
    animal.container.dump(overwrite=True)

    name_data = 'data_optimization.pkl'
    
    save_data(animal,name_data)

if __name__ == '__main__':
    main()
