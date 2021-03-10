import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
from .sensitivity_analysis import calculate_forces
plt.style.use('ggplot')

legs = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']
joints = ['Coxa', 'Coxa_yaw', 'Coxa_roll', 'Femur', 'Femur_roll', 'Tibia', 'Tarsus1']


def plot_mu_sem(
    mu, 
    error, 
    conf=None, 
    plot_label='Mean', 
    x=None, 
    alpha=0.3, 
    color=None, 
    ax=None
):    
    """ Plots mean, confidence interval, and standard deviation (Author: JB)

    Args:
        mu (np.array): mean, shape [N_samples, N_lines] or [N_samples]
        error (np.array): error to be plotted, e.g. standard error of the mean, shape [N_samples, N_lines] or [N_samples]
        conf (int): confidence interval, if none, stderror is plotted instead of std
        plot_label (str, optional): the label for each line either a string if only one line or list of strings if multiple lines
        x (np.array, optional): shape [N_samples]. If not specified will be np.arange(mu.shape[0])
        alpha (float, optional): transparency of the shaded area. default 0.3
        color ([type], optional): pre-specify colour. if None, use Python default colour cycle
        ax ([type], optional): axis to be plotted on, otherwise the current is axis with plt.gca()
    """    
    if ax is None:
        ax = plt.gca()
    if x is None:
        x = np.arange(mu.shape[0])
    p = ax.plot(x, mu, lw=1, color=color, label=plot_label)
    if len(mu.shape) is 1:
        if conf is not None:
            ax.plot(x, mu - conf*error, alpha=alpha, \
                linewidth=1.5, linestyle = ':', color='black',\
                label="Confidence Interval {}%".format(conf))
            ax.plot(x, mu + conf*error, alpha=alpha, \
                linewidth=1.5, linestyle = ':', color='black')
        ax.fill_between(x, mu - error, mu + error, alpha=alpha, color=p[0].get_color())
    else:
        for i in np.arange(mu.shape[1]):
            if conf is not None:
                ax.plot(x, mu[:, i]  - conf*error[:, i], alpha=alpha,\
                        linewidth=1.5, linestyle = ':', color='black', \
                        label="Confidence Interval {}%".format(conf))
                ax.plot(x, mu[:, i]  + conf*error[:, i], alpha=alpha,\
                        linewidth=1.5, linestyle = ':', color='black')
            ax.fill_between(x, mu[:, i] - error[:, i], mu[:, i] + error[:, i],\
                        alpha=alpha, color=p[i].get_color())

def plot_kp_joint(
    *args, 
    show_vector=False,
    calc_force=False,
    full_name='joint_LMTibia', 
    gain_range= np.arange(0.1,1.1,0.2), 
    scaling_factor=1, 
    ax=None, 
    constant='Kv0.9',
    condition ='Kp0.4_Kv0.9',
    beg=2000, 
    intv=250, 
    ground_truth=None
):
    """Plot the joint info of one specific leg versus independent variable. 

    Args:
        *args (np.array): force to be plotted, i.e. grf, lateral friction, thorax
        multiple (bool, optional): plots vectors instead of norm. 
        data (dictionary, optional): dictionary to be plotted, i.e. joint torques
        full_name (str, optional): key name, 'joint_LMTibia'.
        gain_range (np.array, optional): range of gains to be plotted, i.e. np.arange(0.1,1.4,0.2).
        scaling_factor (int, optional): scale to change the units.
        ax ([type], optional): axis to be plotted on, otherwise the current is axis with plt.gca()
        beg (int, optional): beginning of the data to be plotted. the entire data is long
        intv (int, optional): int of the data to be plotted
        ground_truth (np.array, optional): ground truth for position or velocity
    """
    if ax is None:
        ax = plt.gca()
    if ground_truth is not None:
        ax.plot(np.array(ground_truth[beg:beg+intv])*scaling_factor, linewidth=2.5, color="red", label="Ground Truth")

    for k in gain_range:
        k_value = "_".join((constant, 'Kv'+str(round(k,1)) )) if 'Kp' in constant else "_".join(('Kp'+str(round(k,1)), constant))
        
        color = plt.cm.winter(np.linalg.norm(k))
        if condition==k_value: 
            color = 'red' 

        if not calc_force:
            ax.plot(np.array(args[0][k_value][full_name][beg:beg+intv])*scaling_factor, color=color, label=k_value)
        else:
            vector, norm = calculate_forces(full_name, k_value, *args)
            print(norm.shape)
            if show_vector:
                for i,axis in enumerate(['x','y','z']):
                    ax[i].plot(np.array(vector[i,beg:beg+intv])*scaling_factor,\
                                color=color, label=k_value)                
                    ax[i].set_ylabel(axis)
                plt.legend(bbox_to_anchor=(1.1, 0.5), loc = 'upper right')
            else:
                ax.plot(norm[beg:beg+intv]*scaling_factor, color=color, label=k_value)
                plt.legend(bbox_to_anchor=(1.1, 1), loc = 'upper right')

    plt.grid(True)

def calculate_stance_indices(
    grf_forces,     
    legs=['LF', 'RF'], 
    k_value='Kp0.7_Kv0.9', 
    start=1500, 
    stop=3500):
    """ Calculates stance indices 

    Args:
        grf_forces (dictionary): dictionary containing the ground reaction forces
        legs (list, optional): [description]. Defaults to ['LF', 'RF'].
        k_value (str, optional): [description]. Defaults to 'Kp0.7_Kv0.9'.
        start (int, optional): [description]. Defaults to 1500.
        stop (int, optional): [description]. Defaults to 3500.

    Returns:
        [type]: [description]
    """    
    stance_ind = {leg: list() for leg in legs}
    for leg in legs:
        _, norm = calculate_forces(leg,k_value,grf_forces)
        stance_ind[leg] = calculate_stance(norm[start:stop], 0, stop-start)

    return stance_ind

def plot_stance_force(
    data,
    grf_forces,
    show_stance=True,
    show_weight=True,
    ax=None,
    gain_range = np.arange(0.1,1.1,0.2),
    scaling_factor=1,
    constant='Kv0.9',
    legs=['LF', 'RF'], 
    k_constant='Kp0.4_Kv0.9', 
    start=1500,
    stop=3500
):
    """[summary]

    Args:
        data ([type]): [description]
        grf_forces ([type]): [description]
        show_stance (bool, optional): [description]. Defaults to True.
        show_weight (bool, optional): [description]. Defaults to True.
        stance_indices ([type], optional): [description]. Defaults to None.
        ax ([type], optional): [description]. Defaults to None.
        gain_range ([type], optional): [description]. Defaults to np.arange(0.1,1.1,0.2).
        scaling_factor (int, optional): [description]. Defaults to 1.
        constant (str, optional): [description]. Defaults to 'Kv0.9'.
        legs (list, optional): [description]. Defaults to ['LF', 'RF'].
        k_value (str, optional): [description]. Defaults to 'Kp0.4_Kv0.9'.
        start (int, optional): [description]. Defaults to 1500.
        stop (int, optional): [description]. Defaults to 3500.
    """

    stance_indices = calculate_stance_indices(grf_forces, legs, k_constant, start, stop)
    stance_ind_1 = stance_indices[legs[0]]
    stance_ind_2 = stance_indices[legs[1]]

    if ax is None:
        ax = plt.gca()

    for k in gain_range:
        k_value = "_".join((constant, 'Kv'+str(round(k,1)) )) if 'Kp' in constant else "_".join(('Kp'+str(round(k,1)), constant))

        color = plt.cm.winter(np.linalg.norm(k))
        norm_force = np.linalg.norm(data[k_value],axis=1)
        ax.plot(norm_force[start:stop]*scaling_factor,label=k_value, color=color)

    if show_weight:
        plt.axhline(y=10, linewidth=2, color='b', linestyle=':', label="Fly Weight")

    plot_handles, plot_labels = ax.get_legend_handles_labels()

    if show_stance:
        for ind1 in range(0, len(stance_ind_1),2):
            for ind2 in range(0, len(stance_ind_2),2):
                ax.axvspan(stance_ind_1[ind1],stance_ind_1[ind1+1],
                0, 1, facecolor='red', alpha=0.01, transform=ax.get_xaxis_transform())   
                ax.axvspan(stance_ind_2[ind2],stance_ind_2[ind2+1],
                0, 1, facecolor='yellow', alpha=0.01, transform=ax.get_xaxis_transform()) 

        red_patch = mpatches.Patch(color='red', alpha=0.3)
        yellow_patch = mpatches.Patch(color='yellow', alpha=0.3)
        plot_handles += [red_patch] + [yellow_patch]
        plot_labels += [leg+' Stance' for leg in legs]


    ax.legend(plot_handles,plot_labels,loc= 'upper right',bbox_to_anchor=(1.18, 1))
    ax.set(xlabel='Time (msec)', ylabel='Force (mN)')

def plot_grf_thorax(thorax_forces, grf_forces, k_value, ax=None, start= 1000, stop= 3000, scaling_factor=1, show_weight=True):
    """ Plots ground reaction forces of each leg and measured forces from the thorax.

    Args:
        thorax_forces (dictionary): dictionary contains thorax forces
        grf_forces (dictionary): dictionary contains GRF norms of each leg
        k_value (string): Kp and Kv value to be plotted. i.e., Kp0.8_Kv0.9
        ax (object): plot axis object
        start (int, optional): beginning of the data 
        stop (int, optional): end of the data
        scaling_factor (int, optional): scaling factor 
        show_weight (bool, optional): shows weight of the fly
    """    
    if ax is None:
        ax = plt.gca()
    norm_total = 0
    for ind, leg in enumerate(legs):
        _, norm = calculate_forces(leg,k_value,grf_forces)
        ax.plot(norm[start:stop]*1, label=leg)
        norm_total += norm
    ax.plot(norm_total[start:stop], label='Total GRFs')
    
    norm_force = np.linalg.norm(thorax_forces[k_value],axis=1)
    ax.plot(norm_force[start:stop]*scaling_factor, linewidth=2, label='Thorax')


    if show_weight:
        plt.axhline(y=10, linewidth=2, color='b', linestyle=':', label="Fly Weight")

    ax.legend(loc= 'upper right',bbox_to_anchor=(1.2, 1))
    ax.set(xlabel='Time (msec)', ylabel='Force (mN)')


def box_plot(title,x,y,data,ax=None):
    """ Plots a box plot for local sensitivity analysis. """
    if ax is None:
        ax = plt.gca()
    meanlineprops = dict(linestyle='--', linewidth=2.0, color='black')
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    ax= sns.boxplot(x=x, y=y,meanprops=meanlineprops,meanline=True,showmeans=True,data=data)
    ymin, ymax = ax.get_ylim()
    ax.set_yticks(np.arange(ymin,ymax,ymax/10)) 
    if ymax > 1000:
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    ax.set_title(title)

def violin_plot(title,x,y,data,ax=None):
    """ Plots a violin plot for local sensitivity analysis. """
    if ax is None:
        ax = plt.gca()
    meanlineprops = dict(linestyle='--', linewidth=2.0, color='black')
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    ax= sns.violinplot(x=x, y=y,meanprops=meanlineprops,meanline=True,showmeans=True,data=data)
    ymin, ymax = ax.get_ylim()
    ax.set_yticks(np.arange(ymin,ymax,ymax/10)) 
    if ymax > 1000:
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    ax.set_title(title)

def heatmap_plot(title, joint_data, ax=None):
    """ Plots a heatmap plot for global sensitivity analysis. """
    if ax is None:
        ax = plt.gca()

    ax = sns.heatmap(joint_data,annot=True, ax=ax)
    ax.set_title(title)    
    ax.invert_yaxis()


def plot_mse_multiple(axs, stat_results, gain):
    """ Plot Kp versus MSE of each joint. """
    for idx, leg in enumerate(stat_results):
        for i, joint in enumerate(joints):
        #fig.suptitle('Friction forces of {} leg'.format(leg))
            axs[idx].plot(gain, list(stat_results[leg][joint].values()), label=joint)#, color = color_list[i])
            axs[idx].set_ylabel('{}'.format(leg))
            axs[idx].grid()    
            ymin, ymax = axs[idx].get_ylim()
            axs[idx].set_yticks(np.arange(ymin,ymax,ymax/5)) 
            if ymax > 700:
                axs[idx].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
                axs[idx].grid()
    #plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)
    plt.legend(loc = 'upper right', bbox_to_anchor = (0,-0.2,1,1),
            bbox_transform = plt.gcf().transFigure )
    plt.gca().spines["top"].set_alpha(0.5)    
    plt.gca().spines["bottom"].set_alpha(0.5)
    plt.gca().spines["right"].set_alpha(0.5)    
    plt.gca().spines["left"].set_alpha(0.5) 
    plt.xticks(gain)
    plt.xlim([gain[0],gain[-1]])

def plot_grf_thorax(thorax_forces, grf_forces, k_value, ax=None, start= 1000, stop= 3000, scaling_factor=1, show_weight=True):
    if ax is None:
        ax = plt.gca()
    norm_total = 0
    for ind, leg in enumerate(legs):
        _, norm = calculate_forces(leg,k_value,grf_forces)
        ax.plot(norm[start:stop]*1, label=leg)
        norm_total += norm
    ax.plot(norm_total[start:stop], label='Total GRFs')
    
    norm_force = np.linalg.norm(thorax_forces[k_value],axis=1)
    ax.plot(norm_force[start:stop]*scaling_factor, linewidth=2, label='Thorax')


    if show_weight:
        plt.axhline(y=10, linewidth=2, color='b', linestyle=':', label="Fly Weight")

    ax.legend(loc= 'upper right',bbox_to_anchor=(1.2, 1))
    ax.set(xlabel='Time (msec)', ylabel='Force (mN)')
#Author: VLR
def calculate_stance(ground_contact, start, stop):
    """ Calculates the starting and end points of stance phases. """
    stance_ind = np.where(ground_contact>0)[0]
    if stance_ind.size!=0:
        stance_diff = np.diff(stance_ind)
        stance_lim = np.where(stance_diff>1)[0]
        stance=[stance_ind[0]-1]
        for ind in stance_lim:
            stance.append(stance_ind[ind]+1)
            stance.append(stance_ind[ind+1]-1)
        stance.append(stance_ind[-1])
        start_gait_list = np.where(np.array(stance) >= start)[0]
        if len(start_gait_list)>0:
            start_gait = start_gait_list[0]
        else:
            start_gait = start            
        stop_gait_list = np.where(np.array(stance) <= stop)[0]
        if len(stop_gait_list)>0:
            stop_gait = stop_gait_list[-1]+1
        else:
            stop_gait = start_gait
        stance_plot = stance[start_gait:stop_gait]
        if start_gait%2 != 0:
            stance_plot.insert(0,start)
        if len(stance_plot)%2 != 0:
            stance_plot.append(stop)
        return stance_plot



def plot_style():
    CB91_Blue = '#2CBDFE'
    CB91_Green = '#47DBCD'
    CB91_Pink = '#F3A0F2'
    CB91_Pomog = '#f37152'
    CB91_Violet = '#661D98'
    CB91_BrightGreen = '#97c655'
    CB91_Red = '#97200b'
    YELLOW =  '#FAEFAE'
    color_list = [CB91_Blue, CB91_Pomog, CB91_Green, CB91_BrightGreen, CB91_Violet, CB91_Red, CB91_Pink, YELLOW]
    return color_list