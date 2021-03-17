from df3dPostProcessing import df3dPostProcess
from df3dPostProcessing.utils import utils_plots
import numpy as np
import pickle

leg_name = 'allExtraDOF_grooming'

#experiment = '/home/nely/Desktop/animationSimfly/video2/180921_aDN_PR_Fly8_005_SG1_behData_images/images/df3d/pose_result__home_nely_Desktop_animationSimfly_video2_180921_aDN_PR_Fly8_005_SG1_behData_images_images.pkl'

experiment = '/home/nely/Desktop/animationSimfly/video2/180921_aDN_CsCh_Fly6_003_SG1_behData_images/images/df3d/pose_result__home_nely_Desktop_animationSimfly_video2_180921_aDN_CsCh_Fly6_003_SG1_behData_images_images.pkl'

df3d = df3dPostProcess(experiment) 
align = df3d.align_3d_data() 
angles = df3d.calculate_leg_angles()

errors = utils_plots.calculate_min_error(angles,align,extraDOF=['base','roll_tr','yaw_tr','roll_ti','yaw_ti','roll_ta','yaw_ta'])

filename= 'errors_' + leg_name + '.pkl'
with open(filename, 'wb') as handle: 
       pickle.dump(errors, handle)
