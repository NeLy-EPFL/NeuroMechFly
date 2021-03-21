""" use this file to generate template oscillation from the pkl file """
from df3dPostProcessing import df3dPostProcess

#experiment = 'pose_result__data_paper_180918_MDN_PR_Fly1_005_SG1_behData_images.pkl'
#experiment = 'pose_result__data_paper_180919_MDN_PR_Fly5_004_SG1_behData_images.pkl'
#experiment = 'pose_result__data_paper_180921_aDN_PR_Fly8_005_SG1_behData_images.pkl'
#experiment = 'pose_result__data_paper_180919_MDN_CsCh_Fly6_001_SG1_behData_images.pkl'
experiment = '/home/nely/Desktop/animationSimfly/video2/180921_aDN_PR_Fly8_005_SG1_behData_images/images/df3d/pose_result__home_nely_Desktop_animationSimfly_video2_180921_aDN_PR_Fly8_005_SG1_behData_images_images.pkl'
#experiment = 'data/pose_result__home_nely_Desktop_animationSimfly_video2_180919_MDN_CsCh_Fly6_001_SG1_behData_images_images.pkl'

df3d = df3dPostProcess(experiment)

align = df3d.align_3d_data()

angles = df3d.calculate_leg_angles()

