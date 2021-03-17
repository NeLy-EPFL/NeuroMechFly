import numpy as np
import os
import pickle
from .utils.utils_alignment import align_3d
from .utils.utils_angles import calculate_angles
#from .utils.utils_plots import *

df3d_skeleton = ['RFCoxa',
                 'RFFemur',
                 'RFTibia',
                 'RFTarsus',
                 'RFClaw',
                 'RMCoxa',
                 'RMFemur',
                 'RMTibia',
                 'RMTarsus',
                 'RMClaw',
                 'RHCoxa',
                 'RHFemur',
                 'RHTibia',
                 'RHTarsus',
                 'RHClaw',
                 'RAntenna',
                 'RStripe1',
                 'RStripe2',
                 'RStripe3',
                 'LFCoxa',
                 'LFFemur',
                 'LFTibia',
                 'LFTarsus',
                 'LFClaw',
                 'LMCoxa',
                 'LMFemur',
                 'LMTibia',
                 'LMTarsus',
                 'LMClaw',
                 'LHCoxa',
                 'LHFemur',
                 'LHTibia',
                 'LHTarsus',
                 'LHClaw',
                 'LAntenna',
                 'LStripe1',
                 'LStripe2',
                 'LStripe3']

prism_skeleton_AG = ['LFCoxa',
                  'LFFemur',
                  'LFTibia',
                  'LFTarsus',
                  'LFClaw',
                  'LMCoxa',
                  'LMFemur',
                  'LMTibia',
                  'LMTarsus',
                  'LMClaw',
                  'LHCoxa',
                  'LHFemur',
                  'LHTibia',
                  'LHTarsus',
                  'LHClaw',
                  'RFCoxa',
                  'RFFemur',
                  'RFTibia',
                  'RFTarsus',
                  'RFClaw',
                  'RMCoxa',
                  'RMFemur',
                  'RMTibia',
                  'RMTarsus',
                  'RMClaw',
                  'RHCoxa',
                  'RHFemur',
                  'RHTibia',
                  'RHTarsus',
                  'RHClaw']

prism_skeleton = ['RFCoxa',
                  'RFFemur',
                  'RFTibia',
                  'RFTarsus',
                  'RFClaw',
                  'RMCoxa',
                  'RMFemur',
                  'RMTibia',
                  'RMTarsus',
                  'RMClaw',
                  'RHCoxa',
                  'RHFemur',
                  'RHTibia',
                  'RHTarsus',
                  'RHClaw',
                  'LFCoxa',
                  'LFFemur',
                  'LFTibia',
                  'LFTarsus',
                  'LFClaw',
                  'LMCoxa',
                  'LMFemur',
                  'LMTibia',
                  'LMTarsus',
                  'LMClaw',
                  'LHCoxa',
                  'LHFemur',
                  'LHTibia',
                  'LHTarsus',
                  'LHClaw',
                  'RAntenna',
                  'LAntenna',
                  'REye',
                  'LEye',
                  'RHaltere',
                  'LHaltere',
                  'RWing',
                  'LWing',
                  'Proboscis',
                  'Neck',
                  'Genitalia',
                  'Scutellum',
                  'A1A2',
                  'A3',
                  'A4',
                  'A5',
                  'A6']

template_nmf = {'RFCoxa':[0.345, -0.216, 0.327],
                'RFFemur':[0.345, -0.216, -0.025],
                'RFTibia':[0.345, -0.216, -0.731],
                'RFTarsus':[0.345, -0.216, -1.249],
                'RFClaw':[0.345, -0.216, -1.912],
                'LFCoxa':[0.345, 0.216, 0.327],
                'LFFemur':[0.345, 0.216, -0.025],
                'LFTibia':[0.345, 0.216, -0.731],
                'LFTarsus':[0.345, 0.216, -1.249],
                'LFClaw':[0.345, 0.216, -1.912],
                'RMCoxa':[0, -0.125, 0],
                'RMFemur':[0, -0.125, -0.182],
                'RMTibia':[0, -0.125, -0.965],
                'RMTarsus':[0, -0.125, -1.633],
                'RMClaw':[0, -0.125, -2.328],
                'LMCoxa':[0, 0.125, 0],
                'LMFemur':[0, 0.125, -0.182],
                'LMTibia':[0, 0.125, -0.965],
                'LMTarsus':[0, 0.125, -1.633],
                'LMClaw':[0, 0.125, -2.328],                      
                'RHCoxa':[-0.215, -0.087, -0.073],
                'RHFemur':[-0.215, -0.087, -0.272],
                'RHTibia':[-0.215, -0.087, -1.108],
                'RHTarsus':[-0.215, -0.087, -1.793],
                'RHClaw':[-0.215, -0.087, -2.588],                      
                'LHCoxa':[-0.215, 0.087, -0.073],
                'LHFemur':[-0.215, 0.087, -0.272],
                'LHTibia':[-0.215, 0.087, -1.108],
                'LHTarsus':[-0.215, 0.087, -1.793],
                'LHClaw':[-0.215, 0.087, -2.588]}

zero_pose_nmf = {'RF_leg':{'yaw':0,
                           'pitch':0,
                           'roll':0,
                           'roll_tr':0,
                           'th_fe':-np.pi,
                           'th_ti':np.pi,
                           'th_ta':-np.pi},
                 'RM_leg':{'yaw':0,
                           'pitch':0,
                           'roll':-np.pi/2,
                           'roll_tr':0,
                           'th_fe':-np.pi,
                           'th_ti':np.pi,
                           'th_ta':-np.pi},
                 'RH_leg':{'yaw':0,
                           'pitch':0,
                           'roll':-np.pi,
                           'roll_tr':0,
                           'th_fe':-np.pi,
                           'th_ti':np.pi,
                           'th_ta':-np.pi},
                 'LF_leg':{'yaw':0,
                           'pitch':0,
                           'roll':0,
                           'roll_tr':0,
                           'th_fe':-np.pi,
                           'th_ti':np.pi,
                           'th_ta':-np.pi},
                 'LM_leg':{'yaw':0,
                           'pitch':0,
                           'roll':np.pi/2,
                           'roll_tr':0,
                           'th_fe':-np.pi,
                           'th_ti':np.pi,
                           'th_ta':-np.pi},
                 'LH_leg':{'yaw':0,
                           'pitch':0,
                           'roll':np.pi,
                           'roll_tr':0,
                           'th_fe':-np.pi,
                           'th_ti':np.pi,
                           'th_ta':-np.pi}}

class df3dPostProcess:
    def __init__(self, exp_dir, multiple = False, file_name = '', skeleton='df3d', calculate_3d=False):
        self.exp_dir = exp_dir
        self.raw_data_3d = np.array([])
        self.raw_data_2d = np.array([])
        self.skeleton = skeleton
        self.template = template_nmf
        self.zero_pose = zero_pose_nmf
        self.raw_data_cams = {}
        self.load_data(exp_dir, calculate_3d, skeleton, multiple, file_name)
        self.data_3d_dict = load_data_to_dict(self.raw_data_3d, skeleton)
        self.data_2d_dict = load_data_to_dict(self.raw_data_2d, skeleton)
        self.aligned_model = {}

    def align_to_template(self,scale=True,interpolate=False,smoothing=True,original_time_step= 0.01,new_time_step=0.001,window_length=29):
        self.aligned_model = align_3d(self.data_3d_dict,self.skeleton,self.template,scale,interpolate, smoothing, original_time_step, new_time_step, window_length)

        return self.aligned_model

    def calculate_leg_angles(self, begin = 0, end = 0, get_roll_tr = True, save_angles=False, use_zero_pose=True, zero_pose=None):
        if not zero_pose:
            zero_pose = self.zero_pose
                
        leg_angles = calculate_angles(self.aligned_model, begin, end, get_roll_tr, zero_pose)

        if use_zero_pose:
            for leg, angles in leg_angles.items():
                for angle, vals in angles.items():
                    leg_angles[leg][angle] = list(np.array(vals) + zero_pose[leg][angle])
                
        if save_angles:
            path = self.exp_dir.replace('pose_result','joint_angles')
            with open(path, 'wb') as f:
                pickle.dump(leg_angles, f)
        return leg_angles

    
    def load_data(self, exp, calculate_3d, skeleton, multiple, file_name):
        if multiple:
            currentDirectory = os.getcwd()
            dataFile = os.path.join(currentDirectory,file_name)
            data = np.load(dataFile,allow_pickle=True)
            self.raw_data_3d  = data[exp]
        else:
            data = np.load(exp,allow_pickle=True)
            if skeleton == 'prism':
                data={'points3d':data}

            if calculate_3d:
                self.raw_data_3d = triangulate_2d(data, exp)
                
            for key, vals in data.items():
                if not isinstance(key,str):
                    self.raw_data_cams[key] = vals
                elif key == 'points3d':
                    if not calculate_3d:
                        self.raw_data_3d = vals
                elif key == 'points2d':
                    self.raw_data_2d = vals

def triangulate_2d(data, exp_dir):
    img_folder = exp_dir[:exp_dir.find('df3d')]
    out_folder = exp_dir[:exp_dir.find('pose_result')]
    num_images = data['points3d'].shape[0]
    
    from deepfly.CameraNetwork import CameraNetwork
    camNet = CameraNetwork(image_folder=img_folder, output_folder=out_folder, num_images=num_images)

    for cam_id in range(7): 
        camNet.cam_list[cam_id].set_intrinsic(data[cam_id]["intr"]) 
        camNet.cam_list[cam_id].set_R(data[cam_id]["R"]) 
        camNet.cam_list[cam_id].set_tvec(data[cam_id]["tvec"]) 
    camNet.triangulate() 

    return camNet.points3d_m

def load_data_to_dict(data, skeleton):
    final_dict ={}
    if len(data.shape) == 3:
        time_pts, body_parts, axes = data.shape
        num_cams = 1
    elif len(data.shape) == 4:
        num_cams, time_pts, body_parts, axes = data.shape
        cams_dict = {}
    else:
        return final_dict
    
    if skeleton == 'prism':
        tracked_joints = prism_skeleton
    elif skeleton == 'df3d':
        tracked_joints = df3d_skeleton
        
    if body_parts != len(tracked_joints):
        raise Exception("Check tracked joints definition")

    for cam in range(num_cams):
        exp_dict = {}
        if num_cams > 1:
            coords = data[cam]
        else:
            coords = data
        for i, name in enumerate(tracked_joints):
            if 'Coxa' in name or 'Femur' in name or 'Tibia' in name or 'Tarsus' in name or 'Claw' in name:
                body_part = name[0:2] + '_leg'
                landmark = name[2:]
            else:
                if 'Antenna' in name or 'Neck' in name or 'Proboscis' in name or 'Eye' in name: 
                    body_part = 'Head'
                elif 'Haltere' in name or 'Wing' in name or 'Scutellum' in name:
                    body_part = 'Thorax'
                elif 'Stripe' in name or 'Genitalia' in name or 'A1A2' in name or 'A3' in name or 'A4' in name or 'A5' in name or 'A6' in name:
                    body_part = 'Abdomen'
                landmark = name
                
            if body_part not in exp_dict.keys():
                exp_dict[body_part] = {}

            exp_dict[body_part][landmark] = coords[:,i,:]

        if num_cams>1:
            cams_dict[cam] = exp_dict

    if num_cams>1:
        final_dict = cams_dict
    else:
        final_dict = exp_dict
        
    return final_dict






