from bullet_simulation_KM_noSelfCollisions import BulletSimulation
from random import random
import pybullet as p
import numpy as np
import sys
import math
import pickle
import os
from IPython import embed
import pybullet_data
import pandas as pd
import time
from pathlib import Path
import pkgutil
import glob
from NeuroMechFly.container import Container
from NeuroMechFly.sdf.units import SimulationUnitScaling
try:
    from df3dPostProcessing import df3dPostProcess
    df3dPP = True
except ImportError:
    df3dPP = False
    

class DrosophilaSimulation(BulletSimulation):
    
    def __init__(self, container, sim_options, units=SimulationUnitScaling(meters=1000,kilograms=1000)):
        super().__init__(container, units, **sim_options)
        self.torques=[]
        self.grf=[]
        self.collision_forces=[]
        self.ball_rot=[]
        self.lastDraw=[]
        pkg_path = Path(pkgutil.get_loader("NeuroMechFly").get_filename())
        data_path = os.path.join(pkg_path.parents[1],'data',self.behavior,'df3d')
        if df3dPP:
            self.angles = self.calculate_angles(data_path,overwrite_angles=False)
        else:
            angles_path = glob.glob(data_path + '/joint_angles*.pkl')[0]
            with open(angles_path, 'rb') as f:
                self.angles = pickle.load(f)
            
    def calculate_angles(self, data_path, overwrite_angles=False):
        
        experiment = glob.glob(data_path + '/pose_result*.pkl')[0]
        angles_path = glob.glob(data_path + '/joint_angles*.pkl')
        
        if angles_path and not overwrite_angles:
            with open(angles_path[0], 'rb') as f:
                angles = pickle.load(f)
        else:
            df3d = df3dPostProcess(experiment, calculate_3d=True)
            align = df3d.align_to_template(interpolate=True)
            angles = df3d.calculate_leg_angles(save_angles=True)
        
        return angles
        
    def controller_to_actuator(self,t):
        """
        Code that glues the controller the actuator in the system.
        If there are muscles then contoller actuates the muscles.
        If not then the controller directly actuates the joints
        """
        
        joints = [joint for joint in range(self.num_joints)]
        pose = [0]*self.num_joints

        pose[self.joint_id['joint_A3']] = np.deg2rad(-15)
        pose[self.joint_id['joint_A4']] = np.deg2rad(-15)
        pose[self.joint_id['joint_A5']] = np.deg2rad(-15)
        pose[self.joint_id['joint_A6']] = np.deg2rad(-15)

        pose[self.joint_id['joint_LAntenna']] = np.deg2rad(33)
        pose[self.joint_id['joint_RAntenna']] = np.deg2rad(-33)
   
        pose[self.joint_id['joint_Rostrum']] = np.deg2rad(90)
        pose[self.joint_id['joint_Haustellum']] = np.deg2rad(-60)

        pose[self.joint_id['joint_LWing_roll']] = np.deg2rad(90)
        pose[self.joint_id['joint_LWing_yaw']] = np.deg2rad(-17)
        pose[self.joint_id['joint_RWing_roll']] = np.deg2rad(-90)
        pose[self.joint_id['joint_RWing_yaw']] = np.deg2rad(17)

        pose[self.joint_id['joint_Head']] = np.deg2rad(10)

        ind = t
        
        #######Moving up#########
        init_lim = 25
        if ind<init_lim:
            pose[self.joint_id['prismatic_support_2']] = (1.01*self.MODEL_OFFSET[2]-ind*self.MODEL_OFFSET[2]/init_lim)*self.units.meters
        else:
            pose[self.joint_id['prismatic_support_2']] = 0
        #############################
        
        ####LEFT LEGS#######
        pose[self.joint_id['joint_LFCoxa_roll']] = self.angles['LF_leg']['roll'][ind]
        pose[self.joint_id['joint_LFCoxa_yaw']] = self.angles['LF_leg']['yaw'][ind]
        pose[self.joint_id['joint_LFCoxa']] = self.angles['LF_leg']['pitch'][ind]
        pose[self.joint_id['joint_LFFemur_roll']] = self.angles['LF_leg']['roll_tr'][ind]
        pose[self.joint_id['joint_LFFemur']] = self.angles['LF_leg']['th_fe'][ind]
        pose[self.joint_id['joint_LFTibia']] = self.angles['LF_leg']['th_ti'][ind]
        pose[self.joint_id['joint_LFTarsus1']] = self.angles['LF_leg']['th_ta'][ind]

        pose[self.joint_id['joint_LMCoxa_roll']] = self.angles['LM_leg']['roll'][ind]
        pose[self.joint_id['joint_LMCoxa_yaw']] = self.angles['LM_leg']['yaw'][ind]
        pose[self.joint_id['joint_LMCoxa']] = self.angles['LM_leg']['pitch'][ind]
        pose[self.joint_id['joint_LMFemur_roll']] = self.angles['LM_leg']['roll_tr'][ind]
        pose[self.joint_id['joint_LMFemur']] = self.angles['LM_leg']['th_fe'][ind]
        pose[self.joint_id['joint_LMTibia']] = self.angles['LM_leg']['th_ti'][ind]
        pose[self.joint_id['joint_LMTarsus1']] = self.angles['LM_leg']['th_ta'][ind]

        pose[self.joint_id['joint_LHCoxa_roll']] = self.angles['LH_leg']['roll'][ind]
        pose[self.joint_id['joint_LHCoxa_yaw']] = self.angles['LH_leg']['yaw'][ind]
        pose[self.joint_id['joint_LHCoxa']] = self.angles['LH_leg']['pitch'][ind]
        pose[self.joint_id['joint_LHFemur_roll']] = self.angles['LH_leg']['roll_tr'][ind]
        pose[self.joint_id['joint_LHFemur']] = self.angles['LH_leg']['th_fe'][ind]
        pose[self.joint_id['joint_LHTibia']] = self.angles['LH_leg']['th_ti'][ind]
        pose[self.joint_id['joint_LHTarsus1']] = self.angles['LH_leg']['th_ta'][ind]
        
        
        #####RIGHT LEGS######
        pose[self.joint_id['joint_RFCoxa_roll']] = self.angles['RF_leg']['roll'][ind]
        pose[self.joint_id['joint_RFCoxa_yaw']] = self.angles['RF_leg']['yaw'][ind]
        pose[self.joint_id['joint_RFCoxa']] = self.angles['RF_leg']['pitch'][ind]
        pose[self.joint_id['joint_RFFemur_roll']] = self.angles['RF_leg']['roll_tr'][ind]
        pose[self.joint_id['joint_RFFemur']] = self.angles['RF_leg']['th_fe'][ind]
        pose[self.joint_id['joint_RFTibia']] = self.angles['RF_leg']['th_ti'][ind]
        pose[self.joint_id['joint_RFTarsus1']] = self.angles['RF_leg']['th_ta'][ind]

        pose[self.joint_id['joint_RMCoxa_roll']] = self.angles['RM_leg']['roll'][ind]
        pose[self.joint_id['joint_RMCoxa_yaw']] = self.angles['RM_leg']['yaw'][ind]
        pose[self.joint_id['joint_RMCoxa']] = self.angles['RM_leg']['pitch'][ind]
        pose[self.joint_id['joint_RMFemur_roll']] = self.angles['RM_leg']['roll_tr'][ind]
        pose[self.joint_id['joint_RMFemur']] = self.angles['RM_leg']['th_fe'][ind]
        pose[self.joint_id['joint_RMTibia']] = self.angles['RM_leg']['th_ti'][ind]
        pose[self.joint_id['joint_RMTarsus1']] = self.angles['RM_leg']['th_ta'][ind]

        pose[self.joint_id['joint_RHCoxa_roll']] = self.angles['RH_leg']['roll'][ind]
        pose[self.joint_id['joint_RHCoxa_yaw']] = self.angles['RH_leg']['yaw'][ind]
        pose[self.joint_id['joint_RHCoxa']] = self.angles['RH_leg']['pitch'][ind]
        pose[self.joint_id['joint_RHFemur_roll']] = self.angles['RH_leg']['roll_tr'][ind]
        pose[self.joint_id['joint_RHFemur']] = self.angles['RH_leg']['th_fe'][ind]
        pose[self.joint_id['joint_RHTibia']] = self.angles['RH_leg']['th_ti'][ind]
        pose[self.joint_id['joint_RHTarsus1']] = self.angles['RH_leg']['th_ta'][ind]

        #p.setJointMotorControlArray(
        #            self.animal, joints,
        #            controlMode=p.POSITION_CONTROL,
        #            targetPositions=pose)#,
        #            #force=1e16)

        for joint in range(self.num_joints):
            #if joint!=19 and joint!=58:
                    p.setJointMotorControl2(
                    self.animal, joint,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=pose[joint],
                    #force=1e16,
                    positionGain=0.4)
        
        jointTorques = np.array(self.joint_torques())
        #print(jointTorques.shape)
        self.torques.append(jointTorques)

        grf = self.ball_reaction_forces()
        self.grf.append(grf)
        if t%10 == 0:            
            if self.draw_collisions:
                ind = np.where(np.array(grf).transpose()[0]>0)[0]
                draw=[]
                for i in ind:
                    link1 = self.GROUND_CONTACTS[i][:-1]
                    if link1 not in draw:
                        draw.append(link1)
                        p.changeVisualShape(self.animal, self.link_id[link1+'5'],rgbaColor=self.colorCollision)
                for link in self.lastDraw:
                    if link not in draw:
                        p.changeVisualShape(self.animal, self.link_id[link+'5'], rgbaColor=self.colorLegs)

                self.lastDraw = draw
                
        ball_rot = np.array(self.ball_rotations())
        ball_rot[:2] = ball_rot[:2]*self.ball_radius*10 # Distance in mm
        self.ball_rot.append(ball_rot)
        #print(ball_rot)


        #if t%10 == 0:
        #    coll_forces = self.selfCollision_reaction_forces(self.self_collisions)
        #    self.collision_forces.append(coll_forces)

        #print(self.antennae_pos())
        

        '''
        for joint in range(self.num_joints):
                    p.setJointMotorControl2(
                    self.animal, joint,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=np.sin(t*0.1))#,
                    #force=1e16)
        '''

    def feedback_to_controller(self):
        """
        Code that glues the sensors/feedback to controller in the system.
        """
        pass

    def update_parameters(self, params):
        """ Update parameters. """
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
    
    def selfCollision_reaction_forces(self,joints):
        """Get the ground reaction forces.  """
        return list(
            map(self._get_selfCollision_force, joints)
        )

    def ball_rotations(self):
        return tuple(
            state[0] for state in p.getJointStates(
                self.ball_id,
                np.arange(0, p.getNumJoints(self.ball_id))
            )
        )

    def antennae_pos(self):
        return tuple(
            state[0] for state in p.getJointStates(
                self.animal,[self.link_id['LAntenna'],self.link_id['RAntenna']])
        )
    
    
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

    #for i, joints in enumerate(fly.self_collisions):
    #    if joints[0] not in collisions_dict.keys():
    #        collisions_dict[joints[0]] = {}
    #    if joints[1] not in collisions_dict.keys():
    #        collisions_dict[joints[1]] = {}
    #    collisions_dict[joints[0]][joints[1]] = data_collisions[i]
    #    collisions_dict[joints[1]][joints[0]] = data_collisions[i]
        
    path_torque = os.path.join(currentDirectory,'results','torques_'+filename)
    path_grf = os.path.join(currentDirectory,'results','grf_'+filename)
    path_ball_rot = os.path.join(currentDirectory,'results','ballRot_'+filename)
    #path_collisions = os.path.join(currentDirectory,'results','collisions_'+filename)

    with open(path_torque,'wb') as f:
        pickle.dump(torques_dict,f)

    with open(path_grf,'wb') as f:
        pickle.dump(grf_dict,f)

    with open(path_ball_rot,'wb') as f:
        pickle.dump(fly.ball_rot,f)

    #with open(path_collisions,'wb') as f:
    #    pickle.dump(collisions_dict,f)


def main():
    """ Main """
    run_time = 8.97
    time_step = 0.001
    behavior = 'walking'
    
    side = ['L','R']
    pos = ['F','M','H']
    leg_segments = ['Tibia']+['Tarsus' + str(i) for i in range(1, 6)]
    left_front_leg = ['LF'+name for name in leg_segments]
    right_front_leg = ['RF'+name for name in leg_segments]
    body_segments = [s+b for s in side for b in ['Eye','Antenna']]

    ground_contact = [s+p+name for s in side for p in pos for name in leg_segments if name != 'Tibia']

    self_collision = []
    for link0 in left_front_leg:
        for link1 in right_front_leg:
            self_collision.append([link0,link1])
            
    for link0 in left_front_leg+right_front_leg:
        for link1 in body_segments:
            if link0[0] == link1[0]:
                self_collision.append([link0,link1])

    sim_options = {
        "headless": False,
        "model": "../../design/sdf/neuromechfly_noLimits.sdf",
        #"model_offset": [0., 0., 1.12],
        "model_offset": [0., 0., 11.2e-3],
        "run_time": run_time,
        #"pose": '../config/pose.yaml',
        "base_link": 'Thorax',
        #"controller": '../config/locomotion_trot.graphml',
        "ground_contacts": ground_contact,
        "self_collisions": self_collision,
        "draw_collisions": False,
        "record": False,
        'camera_distance': 3.5,
        'track': False,
        'moviename': 'videos/KM_1x_walking_interpolate.mp4',
        'moviefps': 50,
        'slow_down': True,
        'sleep_time': 0.001,
        'rot_cam': False,
        'behavior': behavior,
        }
    container = Container(run_time/time_step)
    animal = DrosophilaSimulation(container, sim_options)
    animal.run(optimization=False)

    name_data = 'data_ball_' + behavior + '.pkl'
    
    save_data(animal,name_data)

if __name__ == '__main__':
    main()
