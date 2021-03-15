import numpy as np
import cv2 as cv
import os
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import pchip_interpolate

def align_3d(pos_3d_dict,skeleton,template,scale,interpolate,smoothing,ots,nts,window_length):
    if skeleton == 'prism':
        pos_3d_dict = get_flycentric(pos_3d_dict)
    fix = get_fix_coxae_pos(pos_3d_dict)
    align = align_3d_to_template(fix,skeleton,template,scale)

    if interpolate:
        align = interpolate_3d_data(align,smoothing,ots,nts,window_length)
    
    return align

def interpolate_3d_data(data,smoothing,ots,nts,window_length):
        """ This function interpolates the signal based on PCHIP and smoothes it using Hamming window. """

        interp_dict={}

        for leg, joints in data.items():
            interp_dict[leg]={}
            for joint, data in joints.items():
                interp_dict[leg][joint]={}
                for key, pos in data.items():
                    if key == 'raw_pos_aligned':
                        pos_t = pos.transpose()
                        interp_pos = []
                        tot_time = len(pos)*ots
                        original_x = np.arange(0,tot_time,ots)
                        new_x = np.arange(0,tot_time,nts)
                        
                        for i in range(len(pos_t)):
                            interpolated_signal = pchip_interpolate(original_x,pos_t[i],new_x)
                            if smoothing:
                                hamming_window = np.hamming(window_length)
                                interpolated_signal = np.convolve(hamming_window/hamming_window.sum(), interpolated_signal, mode='valid')
                            
                            interp_pos.append(interpolated_signal)
                        interp_dict[leg][joint][key] = np.array(interp_pos).transpose()
                    else:
                        interp_dict[leg][joint][key] = pos

        return interp_dict

def get_flycentric(data, pixel_size=[5.86e-3,5.86e-3,5.86e-3]):
    trans_dict = {}
    for segment, landmarks in data.items():
        trans_dict[segment] = {}
        for landmark, pos in landmarks.items():
            vals=[]
            for i, p in enumerate(pos):
                lm = data['LM_leg']['Coxa'][i]
                rm = data['RM_leg']['Coxa'][i]
                pos_zero = (lm + rm)/2
                new_pos = (p - pos_zero)*pixel_size
                vals.append(new_pos)
            trans_dict[segment][landmark] = np.array(vals)

    rot_dict = {}
    for segment, landmarks in trans_dict.items():
        rot_dict[segment] = {}
        mid_zero = (trans_dict['LM_leg']['Coxa'][0] + trans_dict['RM_leg']['Coxa'][0])/2
        hind_zero = (trans_dict['LH_leg']['Coxa'][0] + trans_dict['RH_leg']['Coxa'][0])/2
        th_zero = np.arctan2(mid_zero[1]-hind_zero[1],mid_zero[0]-hind_zero[0])
        for landmark, pos in landmarks.items():
            vals=[]
            for i, p in enumerate(pos):
                mid = (trans_dict['LM_leg']['Coxa'][i] + trans_dict['RM_leg']['Coxa'][i])/2
                hind = (trans_dict['LH_leg']['Coxa'][i] + trans_dict['RH_leg']['Coxa'][i])/2
                th_rot = np.arctan2(mid[1]-hind[1],mid[0]-hind[0])                
                rot_mat = R.from_euler('zyx', [-th_rot,0,0])
                rot_pos = rot_mat.apply(p)
                vals.append(rot_pos)
            rot_dict[segment][landmark] = np.array(vals)
                  
    return rot_dict
    
    
def get_fix_coxae_pos(pos_3d_dict):
    extended_dict = {}
    for segment, landmarks in pos_3d_dict.items():
        if 'leg' in segment:
            extended_dict[segment] = {}
            for landmark, pos in landmarks.items():
                extended_dict[segment][landmark]={}
                pos_t = pos.transpose()
                if 'Coxa' in landmark:
                    mean_x = np.mean(pos_t[0])
                    mean_y = np.mean(pos_t[1])
                    mean_z = np.mean(pos_t[2])
                    extended_dict[segment][landmark]['fixed_pos']=[mean_x,mean_y,mean_z]
                    
                extended_dict[segment][landmark]['raw_pos']=pos
                  
    return extended_dict

'''
def align_prism_to_template(fixed_dict,template,scale):
    lm = fixed_dict['LM_leg']['Coxa']['fixed_pos']
    rm = fixed_dict['RM_leg']['Coxa']['fixed_pos']
    lh = fixed_dict['LH_leg']['Coxa']['fixed_pos']
    rh = fixed_dict['RH_leg']['Coxa']['fixed_pos']

    th_left = np.arctan2(lm[2]-lh[2],lm[0]-lh[0])
    th_right = np.arctan2(rm[2]-rh[2],rm[0]-rh[0])
    th_align={'L': th_left, 'R':th_right}
    print([th*180/np.pi for th in th_align.values()])
    if scale:
        lengths = calculate_fix_lengths(fixed_dict, metric='raw_pos') 
    
    align_dict={}
    for leg, joints in fixed_dict.items():
        align_dict[leg]={}
        if scale:
            template_length = template[leg[:2]+'Coxa'][2]-template[leg[:2]+'Claw'][2]
            tot_length = lengths[leg]['Claw']['total_length']
            scale_factor = template_length/tot_length
        else:
            scale_factor = 1
        for joint, data in joints.items():
            align_dict[leg][joint]={}
            for metric, coords in data.items():
                if '_pos' in metric:
                    pos_zero = np.array(coords) - np.array(fixed_dict[leg]['Coxa']['fixed_pos'])
                    pos_zero_t = pos_zero.transpose()
                    #pos_zero_flip = np.array([pos_zero_t[0],-pos_zero_t[1],pos_zero_t[2]]).transpose()
                    rot_mat = R.from_euler('zyx', [th_align[leg[0]],0,0])
                    rot_pos = rot_mat.apply(pos_zero)*scale_factor
                    align_pos = rot_pos + template[leg[:2]+'Coxa']
                    align_dict[leg][joint][metric+'_aligned']=align_pos

    align_dict = calculate_fix_lengths(align_dict)
    return align_dict
'''
def align_3d_to_template(fixed_dict,skeleton,template,scale):
    lm = fixed_dict['LM_leg']['Coxa']['fixed_pos']
    rm = fixed_dict['RM_leg']['Coxa']['fixed_pos']
    lh = fixed_dict['LH_leg']['Coxa']['fixed_pos']
    rh = fixed_dict['RH_leg']['Coxa']['fixed_pos']

    if skeleton == 'df3d':
        th_left = np.arctan2(lm[2]-lh[2],lm[0]-lh[0])
        th_right = np.arctan2(rm[2]-rh[2],rm[0]-rh[0])
    if skeleton == 'prism':
        th_left = np.arctan2(lm[1]-lh[1],lm[0]-lh[0])
        th_right = np.arctan2(rm[1]-rh[1],rm[0]-rh[0])
        
    th_align={'L': th_left, 'R':th_right} 
    next_joints = {'Coxa':'Femur', 'Femur':'Tibia', 'Tibia':'Tarsus', 'Tarsus':'Claw'}
    prev_joints = {'Femur':'Coxa', 'Tibia':'Femur', 'Tarsus':'Tibia', 'Claw':'Tarsus'}
    if scale:
        lengths = calculate_fix_lengths(fixed_dict, next_joints, metric='raw_pos')
    
    align_dict={}
    for leg, joints in fixed_dict.items():
        align_dict[leg]={}
        for joint, data in joints.items():
            align_dict[leg][joint]={}
            
            if scale:
                if joint == 'Coxa':
                    template_length = template[leg[:2]+joint][2]-template[leg[:2]+'Claw'][2]
                    mean_length = lengths[leg]['Claw']['total_length']
                else:
                    template_length = template[leg[:2]+prev_joints[joint]][2]-template[leg[:2]+joint][2]
                    mean_length = lengths[leg][prev_joints[joint]]['mean_length']
                scale_factor = template_length/mean_length
            else:
                scale_factor = 1
            
            for metric, coords in data.items():
                if '_pos' in metric:
                    pos_zero = np.array(coords) - np.array(fixed_dict[leg]['Coxa']['fixed_pos'])
                    if skeleton == 'df3d':
                        pos_zero_t = pos_zero.transpose()
                        pos_zero = np.array([pos_zero_t[0],-pos_zero_t[1],pos_zero_t[2]]).transpose()
                        rot_mat = R.from_euler('zyx', [0,th_align[leg[0]],np.pi/2])
                    if skeleton == 'prism':
                        rot_mat = R.from_euler('zyx', [th_align[leg[0]],0,0])
                    rot_pos = rot_mat.apply(pos_zero)*scale_factor
                    align_pos = rot_pos + template[leg[:2]+'Coxa']
                    align_dict[leg][joint][metric+'_aligned']=align_pos

    align_dict = calculate_fix_lengths(align_dict, next_joints)
    return align_dict

def calculate_fix_lengths(data, next_joints, metric = 'raw_pos_aligned', data_used=0.8):
    for name, leg in data.items():
        lengths = []
        for segment, body_part in leg.items():
            if segment != 'Claw':
                dist = []                
                #if 'Coxa' in segment:
                #    next_segment = 'Femur'
                #    
                #if 'Femur' in segment:
                #    next_segment = 'Tibia'
                    
                #if 'Tibia' in segment:
                #    next_segment = 'Tarsus'
                    
                #if 'Tarsus' in segment:
                #    next_segment = 'Claw'
                ind = int(len(body_part[metric]) * (1-data_used)/2)
                for i, point in enumerate(body_part[metric]):
                    a = point
                    b = data[name][next_joints[segment]][metric][i] 
                    dist.append(np.linalg.norm(a-b))
                sort_dist = np.sort(dist)        
                body_part['mean_length']=np.mean(sort_dist[ind:-ind])
                lengths.append(np.mean(sort_dist[ind:-ind]))
            else:
                body_part['total_length']=np.sum(lengths)
    return data

#def align_data(exp_dict,skeleton='df3d'):
#    fix = fixed_lengths_and_base_point(exp_dict)
#    align = align_model(fix,skeleton)   
#    return align

'''    
def fixed_lengths_and_base_point(raw_dict):
    new_dict = {}
    for segment, landmarks in raw_dict.items():
        if 'leg' in segment:
            new_dict[segment] = {}
            for landmark, pos in landmarks.items():
                new_dict[segment][landmark]={}
                dist = []
                pos_t = pos.transpose()
                #pos_t[1] = -pos_t[1]
                #corr_pos = pos_t.transpose()
                corr_pos = pos
                if 'Coxa' in landmark:
                    mean_x = np.mean(pos_t[0])
                    mean_y = np.mean(pos_t[1])
                    mean_z = np.mean(pos_t[2])
                    new_dict[segment][landmark]['fixed_pos']=[mean_x,mean_y,mean_z]
                    for i, point in enumerate(corr_pos):
                        #key = [name for name in landmarks.keys() if 'Femur' in name]
                        a = point
                        b = raw_dict[segment]['Femur'][i] 
                        dist.append(np.linalg.norm(a-b))
                    
                if 'Femur' in landmark:
                    for i, point in enumerate(corr_pos):
                        a = point
                        b = raw_dict[segment]['Tibia'][i] 
                        dist.append(np.linalg.norm(a-b))

                if 'Tibia' in landmark:
                    for i, point in enumerate(corr_pos):
                        a = point
                        b = raw_dict[segment]['Tarsus'][i] 
                        dist.append(np.linalg.norm(a-b))

                if 'Tarsus' in landmark:
                    for i, point in enumerate(corr_pos):
                        a = point
                        b = raw_dict[segment]['Claw'][i] 
                        dist.append(np.linalg.norm(a-b))

                if 'Claw' in landmark:
                    new_dict[segment][landmark]['raw_pos']=pos
                    break
                    
                new_dict[segment][landmark]['raw_pos']=corr_pos
                
                new_dict[segment][landmark]['mean_length']=np.mean(dist)    
    return new_dict

def align_model(fixed_dict,skeleton):
    front_coxae = []
    middle_coxae = []
    hind_coxae = []
    coxae={}
    for leg, joints in fixed_dict.items():
        for joint, data in joints.items():
            if 'F_leg' in leg and 'Coxa' in joint:
                front_coxae.append(data['fixed_pos'])
            if 'M_leg' in leg and 'Coxa' in joint:
                middle_coxae.append(data['fixed_pos'])
            if 'H_leg' in leg and 'Coxa' in joint:
                hind_coxae.append(data['fixed_pos'])

    coxae['F_'] = np.array(front_coxae)
    coxae['M_'] = np.array(middle_coxae)
    coxae['H_'] = np.array(hind_coxae)

    alignment = {}
    for pos, coords in coxae.items():
        alignment[pos] = {}
        middle_point= [(point[0]+point[1])/2 for point in coords.transpose()]
        y_angle = np.arctan2(middle_point[2],middle_point[0])
        r_mp = R.from_euler('zyx', [0,y_angle,np.pi/2])
        new_mid_pnt = r_mp.apply(middle_point)
        #new_mid_pnt[2] *= -1
        zero_coords = coords - middle_point
        th_y = np.arctan2(zero_coords[0][0],zero_coords[0][2])
        r_roty = R.from_euler('zyx', [0,-th_y,0])
        new_coords = r_roty.apply(zero_coords)
        th_x = np.arctan2(new_coords[0][1],new_coords[0][2])        
        alignment[pos]['th_y'] = th_y
        alignment[pos]['th_x'] = th_x
        alignment[pos]['mid_pnt'] = middle_point
        alignment[pos]['offset'] = new_mid_pnt
                
    aligned_dict = {}
    for leg, joints in fixed_dict.items():
        aligned_dict[leg]={}
        theta_y = [angle['th_y'] for pos, angle in alignment.items() if pos in leg][0]
        theta_x = [angle['th_x'] for pos, angle in alignment.items() if pos in leg][0]
        mid_point = [point['mid_pnt'] for pos, point in alignment.items() if pos in leg][0]
        offset = [point['offset'] for pos, point in alignment.items() if pos in leg][0]        
        for joint, data in joints.items():
            aligned_dict[leg][joint]={}
            for metric, coords in data.items():
                if '_pos' in metric:
                    key = metric + '_aligned'
                    r = R.from_euler('zyx', [0,-theta_y,theta_x + np.pi/2])
                    zero_cent = coords - np.array(mid_point)
                    rot_coords = r.apply(zero_cent)
                    trans_coords = rot_coords + (offset - alignment['M_']['offset'])
                    align_coords = np.array([trans_coords.transpose()[0],trans_coords.transpose()[1],-trans_coords.transpose()[2]]).transpose()
                    aligned_dict[leg][joint][key] = align_coords
                    #if joint == 'Coxa':
                    #    aligned_dict[leg][joint]['offset'] = alignment['M_']['offset']*[1,1,-1]
                else:
                    aligned_dict[leg][joint][metric] = coords
       
    return aligned_dict

def rescale_using_2d_data(data_3d,data_2d,cams_info,exp_dir,pixelSize=[5.86e-3,5.86e-3],scale_procrustes = True,procrustes_factor={'LF':0.7,'LM':0.75,'LH':0.8,'RF':0.7,'RM':0.75,'RH':0.8}):
    """
    Rescale 3d data using 2d data
    """
    views = {}

    ##original: 0.8,0.25,0.4,0.2
    ##for walking: 0.75,0.1,0.15,0.0
    
    x_factor = 5.0
    y_factor = 0.33
    z_factor = -0.1

    for key, info in cams_info.items():
        r = R.from_dcm(info['R']) 
        th = r.as_euler('zyx', degrees=True)[1]
        if 90-th<15:
            views['R_points2d'] = data_2d[key-1]
            views['R_camID'] = key-1 
        elif 90-th>165:
            views['L_points2d'] = data_2d[key-1]
            views['L_camID'] = key-1

    for name, leg in data_3d.items():  
        for k, joints in leg.items():
            dist_px = views[name[:1]+'_points2d'][name][k][0]-np.mean(views[name[:1]+'_points2d'][name[:1]+'M_leg']['Coxa'],axis=0)
            #if scale_procrustes:
            #    y_dist = procrustes_factor*np.mean([dist_px[0],dist_px[1]])*pixelSize[0]
            #else:
            #    y_dist = 0#np.mean([dist_px[0],dist_px[1]])*pixelSize[0]
                
            dist_mm = dist_px *pixelSize
            
            #if k == 'Tarsus' and 'F' in name:
            #    y_offset = 0.1#0.5
            #elif k == 'Claw' and 'F' in name:
            #    y_offset = 0.15#0.55
            #elif k == 'Tibia' and 'F' in name:
            #    y_offset = 0.0#0.05

            if name[:1] == 'L':
                dist = np.array([dist_mm[0],0,-dist_mm[1]])
            if name[:1] == 'R':
                dist = np.array([-dist_mm[0],0,-dist_mm[1]])

            offset = joints['raw_pos_aligned'][0]-dist
            
            #x_vals = joints['raw_pos_aligned'].transpose()[0] - offset[0]
            #y_vals = joints['raw_pos_aligned'].transpose()[1] - y_offset
            #z_vals = joints['raw_pos_aligned'].transpose()[2] - offset[2]
            x_vals=[]
            y_vals=[]
            z_vals=[]
            #print(name,k,np.min(np.array(joints['raw_pos_aligned']).transpose()[1]),np.max(np.array(joints['raw_pos_aligned']).transpose()[1]))
            mean_x = np.mean(np.array(joints['raw_pos_aligned']).transpose()[0])
            diff_x = np.max(np.array(joints['raw_pos_aligned']).transpose()[0])-np.min(np.array(joints['raw_pos_aligned']).transpose()[0])
            for i, pnt in enumerate(joints['raw_pos_aligned']):
                if scale_procrustes:
                    if k == 'Coxa':
                        pnt[1] *= 0.5
                    else:
                        pnt[1] *= procrustes_factor[name[:2]]
                        
                if k!='Coxa' or k!='Femur':
                    x_new = pnt[0] - offset[0]-((pnt[0]-dist[0])/x_factor)
                else:
                    x_new = pnt[0] - offset[0]

                if abs(leg['Tarsus']['raw_pos_aligned'][i][1])<y_factor and (k=='Claw' or k=='Tarsus') and 'F' in name:
                    
                    if k=='Claw':
                        y_factor_mod = 1.5*y_factor
                    if k=='Tarsus':
                        y_factor_mod = 1.15*y_factor
                    if 'L' in name:
                        y_factor_mod *= -1
                    
                    y_new = pnt[1] - y_factor_mod
                else:
                    y_new = pnt[1]

                if k=='Claw' or k=='Tarsus':
                    z_new = pnt[2] - offset[2]*(z_factor-pnt[2]+ offset[2])/(z_factor-dist[2])
                else:
                    z_new = pnt[2] - offset[2]
                    
                x_vals.append(x_new)
                y_vals.append(y_new)
                z_vals.append(z_new)
                       
            joints['raw_pos_aligned'] = np.array([x_vals,y_vals,z_vals]).transpose()

            if k == 'Coxa':
                joints['fixed_pos_aligned'] = np.mean(joints['raw_pos_aligned'],axis=0)
    
    data_3d = recalculate_lengths(data_3d)
    
    return data_3d
'''

'''
def rescale_using_2d_data(data_3d,data_2d,cams_info,exp_dir,pixelSize=[5.86e-3,5.86e-3],scale_procrustes = True,procrustes_factor=0.75):
    """
    Rescale 3d data using 2d data
    """
    views = {}

    ##original: 0.8,0.25,0.4,0.2
    ##for walking: 0.75,0.1,0.15,0.0
    
    for key, info in cams_info.items():
        r = R.from_dcm(info['R']) 
        th = r.as_euler('zyx', degrees=True)[1]
        if 90-th<15:
            views['R_points2d'] = data_2d[key-1]
            views['R_camID'] = key-1 
        elif 90-th>165:
            views['L_points2d'] = data_2d[key-1]
            views['L_camID'] = key-1

    for name, leg in data_3d.items():  
        for k, joints in leg.items():
            dist_px = views[name[:1]+'_points2d'][name][k][0]-views[name[:1]+'_points2d'][name[:1]+'M_leg']['Coxa'][0]
            #if scale_procrustes:
            #    y_dist = procrustes_factor*np.mean([dist_px[0],dist_px[1]])*pixelSize[0]
            #else:
            #    y_dist = 0#np.mean([dist_px[0],dist_px[1]])*pixelSize[0]
                
            dist_mm = dist_px *pixelSize
            y_offset = 0
            
            if k == 'Tarsus' and 'F' in name:
                y_offset = 0.1#0.5
            elif k == 'Claw' and 'F' in name:
                y_offset = 0.15#0.55
            elif k == 'Tibia' and 'F' in name:
                y_offset = 0.0#0.05

            if name[:1] == 'L':
                dist = np.array([dist_mm[0],0,-dist_mm[1]])
                y_offset *=-1
            if name[:1] == 'R':
                dist = np.array([-dist_mm[0],0,-dist_mm[1]])

            offset = joints['raw_pos_aligned'][0]-dist
            
            #x_vals = joints['raw_pos_aligned'].transpose()[0] - offset[0]
            #y_vals = joints['raw_pos_aligned'].transpose()[1] - y_offset
            #z_vals = joints['raw_pos_aligned'].transpose()[2] - offset[2]
            x_vals=[]
            y_vals=[]
            z_vals=[]
            for pnt in joints['raw_pos_aligned']:
                x_new = pnt[0] - offset[0]
                y_new = pnt[1] - y_offset
                z_new = pnt[2] - offset[2]

                if scale_procrustes:
                    y_new *= procrustes_factor
                
                x_vals.append(x_new)
                y_vals.append(y_new)
                z_vals.append(z_new)
            
            #if scale_procrustes:
            #    y_vals *= procrustes_factor
                       
            joints['raw_pos_aligned'] = np.array([x_vals,y_vals,z_vals]).transpose()

            if k == 'Coxa':
                joints['fixed_pos_aligned'] = np.mean(joints['raw_pos_aligned'],axis=0)
    
    data_3d = recalculate_lengths(data_3d)
    
    return data_3d
'''
'''
def rescale_using_2d_data2(data_3d,data_2d,cams_info,exp_dir,pixelSize=[5.86e-3,5.86e-3],scale_procrustes = True,procrustes_factor=0.5):
    """
    Rescale 3d data using 2d data
    """
    right_view = {}
    left_view = {}
    #front_view = {}
    for key, info in cams_info.items():
        r = R.from_dcm(info['R']) 
        th = r.as_euler('zyx', degrees=True)[1]
        if 90-th<15:
            right_view['R_points2d'] = data_2d[key-1]
            right_view['cam_id'] = key-1 
        elif 90-th>165:
            left_view['L_points2d'] = data_2d[key-1]
            left_view['cam_id'] = key-1
        #elif abs(th)+1 < 10:
        #    front_view['F_points2d'] = data_2d[key-1]
        #    front_view['cam_id'] = key-1

    #draw_legs_from_2d(left_view, exp_dir,saveimgs=True)   

    for name, leg in data_3d.items():  
        for k, joints in leg.items():
            if scale_procrustes and k =='Coxa':
                joints['fixed_pos_aligned'][1] = joints['fixed_pos_aligned'][1]*procrustes_factor
                x_3d = joints['raw_pos_aligned'].transpose()[0] 
                y_3d = joints['raw_pos_aligned'].transpose()[1] * procrustes_factor
                z_3d = joints['raw_pos_aligned'].transpose()[2]
                scaled_data = np.array([x_3d,y_3d,z_3d]).transpose()
                joints['raw_pos_aligned'] = scaled_data
            if k == 'Femur':
                prev = 'Coxa'
            if k == 'Tibia':
                prev = 'Femur'
            if k == 'Tarsus':
                prev = 'Tibia'
            if k == 'Claw':
                prev = 'Tarsus'
            if k != 'Coxa':
                x_3d = joints['raw_pos_aligned'].transpose()[0]
                #if scale_procrustes:
                #    y_3d = joints['raw_pos_aligned'].transpose()[1]*0.5
                #else:
                y_3d = joints['raw_pos_aligned'].transpose()[1] 
                z_3d = joints['raw_pos_aligned'].transpose()[2]
                x_3d_amp = np.max(x_3d)-np.min(x_3d) 
                x_3d_zero = np.mean(data_3d[name][prev]['raw_pos_aligned'].transpose()[0])
                
                y_3d_amp = np.max(y_3d)-np.min(y_3d) 
                y_3d_zero = np.mean(data_3d[name][prev]['raw_pos_aligned'].transpose()[1])

                z_3d_amp = np.max(z_3d)-np.min(z_3d)
                z_3d_zero = np.mean(data_3d[name][prev]['raw_pos_aligned'].transpose()[2])

                if 'L' in name:
                    x_2d = left_view['L_points2d'][name][k].transpose()[0]
                    z_2d = left_view['L_points2d'][name][k].transpose()[1]
                    x_2d_amp = (np.max(x_2d)-np.min(x_2d)) * pixelSize[0]
                    z_2d_amp = (np.max(z_2d)-np.min(z_2d)) * pixelSize[1]
                if 'R' in name:
                    x_2d = right_view['R_points2d'][name][k].transpose()[0]
                    z_2d = right_view['R_points2d'][name][k].transpose()[1]
                    x_2d_amp = (np.max(x_2d)-np.min(x_2d)) * pixelSize[0]
                    z_2d_amp = (np.max(z_2d)-np.min(z_2d)) * pixelSize[1]

                #print(name, k, x_2d_amp / x_3d_amp, np.mean([x_2d_amp/x_3d_amp,z_2d_amp/z_3d_amp]), z_2d_amp / z_3d_amp)
                x_3d_scaled = []
                y_3d_scaled = []
                z_3d_scaled = []
                x_factor = x_2d_amp / x_3d_amp
                y_factor = np.mean([x_2d_amp/x_3d_amp,z_2d_amp/z_3d_amp])
                z_factor = z_2d_amp / z_3d_amp
                #print(name,x_factor,y_factor,z_factor)
                for i in range(len(x_3d)):
                    x_3d_scaled.append(x_3d_zero + (x_3d[i] - x_3d_zero) * x_factor)
                    if scale_procrustes:
                        y_3d_scaled.append((y_3d_zero + (y_3d[i] - y_3d_zero) * y_factor)*procrustes_factor)
                    else:
                        y_3d_scaled.append(y_3d_zero + (y_3d[i] - y_3d_zero) * y_factor)
                    z_3d_scaled.append(z_3d_zero + (z_3d[i] - z_3d_zero) * z_factor)
                
                scaled_data = np.array([x_3d_scaled,y_3d_scaled,z_3d_scaled]).transpose()

                joints['raw_pos_aligned'] = scaled_data
    
    data_3d = recalculate_lengths(data_3d)
    
    return data_3d
'''
