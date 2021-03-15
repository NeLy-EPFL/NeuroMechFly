""" use this file to generate template oscillation from the pkl file """
from df3dPostProcessing import df3dPostProcess
import pickle
#experiment = 'pose_result__data_paper_180918_MDN_PR_Fly1_005_SG1_behData_images.pkl'
#experiment = 'pose_result__data_paper_180919_MDN_PR_Fly5_004_SG1_behData_images.pkl'
#experiment = 'pose_result__data_paper_180921_aDN_PR_Fly8_005_SG1_behData_images.pkl'
#experiment = 'pose_result__data_paper_180919_MDN_CsCh_Fly6_001_SG1_behData_images.pkl'
experiment = '/home/nely/Desktop/animationSimfly/video2/180921_aDN_PR_Fly8_005_SG1_behData_images/images/df3d/pose_result__home_nely_Desktop_animationSimfly_video2_180921_aDN_PR_Fly8_005_SG1_behData_images_images.pkl'
#experiment = 'data/pose_result__home_nely_Desktop_animationSimfly_video2_180919_MDN_CsCh_Fly6_001_SG1_behData_images_images.pkl'

path = '/Users/ozdil/Desktop/GIT/NeuroMechFy1x/NeuroMechFly/scripts/SensitivityAnalysis/pose_result__home_nely_Desktop_DF3D_data_180921_aDN_CsCh_Fly6_003_SG1_behData_images_images.pkl'
df3d1 = df3dPostProcess(path)
align_processed = df3d1.align_to_template(interpolate=True,smoothing=True)
angles_processed = df3d1.calculate_leg_angles()

with open('grooming_joint_angles.pkl','wb') as f:
    pickle.dump(angles_processed,f)


