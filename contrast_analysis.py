#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Allen Institute Software License - This software license is the 2-clause BSD
# license plus a third clause that prohibits redistribution for commercial
# purposes without further permission.
#
# Copyright 2017. Allen Institute. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Redistributions for commercial purposes are not permitted without the
# Allen Institute's written permission.
# For purposes of this license, commercial purposes is the incorporation of the
# Allen Institute's software into anything for which you will charge fees or
# other compensation. Contact terms@alleninstitute.org for commercial licensing
# opportunities.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection


from contrast_single_cell_trace_figures import plot_rasters_at_peak, plot_example_trace_with_events, plot_single_cell_tuning_curves
from contrast_running import populate_curve_dict, plot_from_curve_dict
from contrast_utils import roi_duplications, load_dff_traces, grating_params, get_sessions, get_analysis_QCd, load_sweep_events, load_mean_sweep_events, load_mean_sweep_running, load_sweep_table, shorthand, dataset_params, get_cre_colors, get_peak_conditions, select_peak_contrast, sort_by_weighted_peak_direction, center_of_mass_for_list, center_direction_zero
from contrast_metrics import compute_error_curve, compute_mean_condition_responses, compute_SEM_condition_responses, chi_square_all_conditions, pool_sessions, percentile_to_NLL, calc_DSI, calc_OSI, test_SbC, calc_resultant, condition_response_running
from contrast_model_selection import LP_HP_model_selection, model_counts, get_best_model_idx
from contrast_GLM import model_GLM
from contrast_decoder import decode_direction_from_running
import SSN_flexible_size as ssn

SIG_THRESH = 0.01

def the_whole_trick(savepath):
    
    df = get_analysis_QCd(savepath)  

    plot_Figure_1(df,savepath)
    plot_Figure_2(df,savepath)
    plot_Figure_2_supplement_1(df,savepath)
    plot_Figure_2_supplement_2(df,savepath)
    plot_Figure_3(df,savepath)
    plot_Figure_4(df,savepath)
    plot_Figure_5(savepath)
                    
def plot_Figure_1(df,savepath):
    
    plot_single_cell_examples(df,savepath)
    
def plot_Figure_2(df,savepath):
    
   #  #2a
    plot_fraction_responsive(df,savepath)    
   #  #2b
   #  plot_all_waterfalls(df,savepath,scale='blank_subtracted_NLL')
    plot_all_waterfalls(df,savepath,scale='event')
    #make_direction_legend(savepath)
   #  #2c
    plot_direction_vector_sum_by_contrast(df,savepath)
   #  make_radial_plot_legend(savepath)
   #  #2d
    plot_SbC_stats(df,savepath)
   #  #2e
    model_by_cre_CI_plot(df,savepath)
   #  #2f
    plot_contrast_CoM(df,savepath,curve='cdf')
   #  #2g
    plot_OSI_distribution(df,savepath,curve='cdf')
   #  #2h
    plot_DSI_distribution(df,savepath,curve='cdf')
  
def plot_Figure_2_supplement_1(df,savepath):
    
    direction_bias_across_sessions(df,savepath)
    decode_direction_from_running(df,savepath)
    
def plot_Figure_2_supplement_2(df,savepath):
    
    fluorescence_suppression(df,savepath)
    
def plot_Figure_3(df,savepath):
    
    plot_tuning_split_by_run_state(df,savepath)
    
def plot_Figure_3_supplement_1(df,savepath):
    
    model_by_cre_CI_plot(df,savepath)
    
def plot_Figure_4(df,savepath):
    
    model_GLM(df,savepath)  
   
def plot_Figure_5(savepath):
    
    W = ssn.get_W()
    net = ssn.init_net()
    
    #5a-f
    ssn.run_main_condition(savepath,save_format='png')
    #5g-i
    ssn.VIP_SST_scale(W,net,savepath)
    #5j-l
    ssn.PV_SST_ratio(W,net,savepath)
    
def plot_fraction_responsive(df,savepath,verbose=True):
    
    colors = get_cre_colors()
    
    plt.figure(figsize=(6,4))
    ax = plt.subplot(111)
    ax.set_ylim(0.0,110)
    label_loc = []
    labels = []
    areas, cres = dataset_params()
    for i_a,area in enumerate(['VISp']):
        for i_c,cre in enumerate(cres):
            session_IDs = get_sessions(df,area,cre)
    
            if len(session_IDs) > 0:
    
                resp, blank, p_all = pool_sessions(session_IDs,area+'_'+cre,savepath) 
                x_pos = 8*i_a+i_c
                
                labels.append(shorthand(cre))
                label_loc.append(x_pos)
                ax.bar(x_pos,100.0*np.mean(p_all<0.01),color=colors[cre])
                
                if verbose:
                    num_cells = len(resp)
                    print(shorthand(cre) + ': ' + str(num_cells) + ' cells, ' + str(len(session_IDs)) + ' sessions')
                
    ax.set_xticks(label_loc)
    ax.set_xticklabels(labels,rotation=45,fontsize=10)
    ax.set_ylim(0,100)
    ax.set_xlim(-1,14)
    ax.set_ylabel('Percent of neurons responsive',fontsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(savepath+'responsiveness.svg',format='svg')
    plt.close() 
   
def plot_single_cell_examples(df,
                              savepath,
                              figure_format='svg'):

    examples = [
                  ('Vip',695640480,1,5),
                  ('Sst',785370795,2,1),
                  ('Cux2',724712005,26,7),
                  ('Rorb',692130296,24,4),
                ]
    
    for example in examples:
        cre = example[0]
        session_ID = example[1]
        cell_idx = example[2]
        example_trace = example[3]
        
        plot_path = savepath+'example_cells/'
        if not os.path.isdir(plot_path):
            os.mkdir(plot_path)
        
        sweep_table = load_sweep_table(savepath,session_ID)
        sweep_events = load_sweep_events(savepath, session_ID)
        mean_sweep_events = load_mean_sweep_events(savepath, session_ID)
        condition_responses, __ = compute_mean_condition_responses(sweep_table,mean_sweep_events)  
        
        plot_example_trace_with_events(session_ID,cre,cell_idx,example_trace,savepath,plot_path,figure_format)
        plot_single_cell_tuning_curves(session_ID,savepath,cre,cell_idx,plot_path,figure_format)
        plot_rasters_at_peak(sweep_table,sweep_events,condition_responses,cell_idx,session_ID,cre,plot_path,figure_format)
    
def fluorescence_suppression(df,savepath):
    
    directions,contrasts = grating_params()
    session_IDs = get_sessions(df,'VISp','Vip-IRES-Cre')
    low_contrast = np.zeros((63,120))
    high_contrast = np.zeros((63,120))
    curr = 0
    for session_ID in session_IDs:
        
        stim_table = load_sweep_table(savepath,session_ID)
        dff = load_dff_traces(savepath,session_ID)
        num_cells = np.shape(dff)[0]
        
        for nc in range(num_cells):
            
            low_contrast[curr] = get_mean_condition_dff(stim_table,dff[nc],0,0.05)
            high_contrast[curr] = get_mean_condition_dff(stim_table,dff[nc],0,0.8)
            curr+=1
            
    low_contrast = 100.0*low_contrast
    high_contrast = 100.0*high_contrast
    t = np.arange(-30,90) / 30.0       
    
    plt.figure(figsize=(6,4))
    ax1 = plt.subplot(121)
    
    y_max = 30
    pc = PatchCollection([Rectangle((0.0,0.0),2.0,y_max)], 
                         facecolor=[0.7,0.7,0.7], 
                         alpha=0.5,
                         edgecolor=[0.7,0.7,0.7])
    ax1.add_collection(pc)

    ax1.fill_between(t,
                     low_contrast.mean(axis=0) - low_contrast.std(axis=0)/np.sqrt(63.), 
                     low_contrast.mean(axis=0) + low_contrast.std(axis=0)/np.sqrt(63.))
    ax1.plot(t,low_contrast.mean(axis=0),'r',linewidth=2)
    
    ax1.set_ylim(0,y_max)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('dF/F (%)')
    ax1.set_title('Low contrast')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    
    ax2 = plt.subplot(122)
    
    pc = PatchCollection([Rectangle((0.0,0.0),2.0,y_max)], 
                         facecolor=[0.7,0.7,0.7], 
                         alpha=0.5,
                         edgecolor=[0.7,0.7,0.7])
    ax2.add_collection(pc)
    
    ax2.fill_between(t,
                     high_contrast.mean(axis=0) - high_contrast.std(axis=0)/np.sqrt(63.), 
                     high_contrast.mean(axis=0) + high_contrast.std(axis=0)/np.sqrt(63.))
    ax2.plot(t,high_contrast.mean(axis=0),'r',linewidth=2)
    
    ax2.set_ylim(0,y_max)
    ax2.set_xlabel('Time (s)')
    ax2.set_title('High contrast')
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    plt.savefig(savepath+'suppression_traces.svg',format='svg')
    plt.close()
    
def get_mean_condition_dff(stim_table,
                           cell_dff,
                           direction,
                           contrast,
                           pre_frames=30,
                           stim_frames=60,
                           post_frames=30):
    
    is_direction = stim_table['Ori'].values == direction
    is_contrast = stim_table['Contrast'].values == contrast
    is_condition = is_direction & is_contrast
    condition_sweeps = np.argwhere(is_condition)[:,0]
    
    num_frames = pre_frames+stim_frames+post_frames
    sweep_dff = np.zeros((num_frames,len(condition_sweeps)))
    for i_sweep,sweep in enumerate(condition_sweeps):
        start_frame = int(stim_table['Start'][sweep])
        end_frame = start_frame+stim_frames
        sweep_dff[:,i_sweep] = cell_dff[(start_frame-pre_frames):(end_frame+post_frames)]
        
    return sweep_dff.mean(axis=1)

def plot_all_waterfalls(df,savepath,scale='blank_subtracted_NLL'):
    
    areas, cres = dataset_params()
    for area in areas:
        for cre in cres:
            session_IDs = get_sessions(df,area,cre)

            if len(session_IDs)>0:

                #sort by direction using event magnitude
                direction_order = get_cell_order_direction_sorted(df,area,cre,savepath)
                
                #display response significance
                resp, blank, p_all = pool_sessions(session_IDs,
                                                          area+'_'+cre,
                                                          savepath,
                                                          scale=scale)
                if scale=='event':
                    #resp = (resp - blank.reshape(len(blank),1,1))/(resp + blank.reshape(len(blank),1,1))
                    norm_factor = (np.mean(resp,axis=(1,2),keepdims=True) + blank.reshape(len(blank),1,1))
                    resp = (resp - blank.reshape(len(blank),1,1))/norm_factor
                
                resp = center_direction_zero(resp)
                condition_responses = resp[p_all<SIG_THRESH]     
                
                dirXcon_mat = concatenate_contrasts(condition_responses)
                dirXcon_mat = dirXcon_mat[direction_order]
                dirXcon_mat = move_all_negative_to_bottom(dirXcon_mat)
                
                plot_full_waterfall(dirXcon_mat,cre,shorthand(area)+'_'+shorthand(cre)+'_full',scale,savepath)

def get_cell_order_direction_sorted(df,area,cre,savepath):
    
    session_IDs = get_sessions(df,area,cre)
    
    resp, blank, p_all = pool_sessions(session_IDs,
                                      area+'_'+cre,
                                      savepath,
                                      scale='event')
    
    resp = center_direction_zero(resp)          
    condition_responses = resp[p_all<SIG_THRESH]
    
    peak_dir, peak_con = get_peak_conditions(condition_responses)
    direction_mat = select_peak_contrast(condition_responses,peak_con)
    direction_order = sort_by_weighted_peak_direction(direction_mat)
    
    return direction_order

def concatenate_contrasts(condition_responses):

    (num_cells,num_dir,num_con) = np.shape(condition_responses)
    dirXcon_mat = np.zeros((num_cells,num_dir*num_con))
    for i_con in range(num_con):
        dirXcon_mat[:,(i_con*num_dir):((i_con+1)*num_dir)] = condition_responses[:,:,i_con]

    return dirXcon_mat

def move_all_negative_to_bottom(cells_by_condition):
    #assumes blank subtracted and already sorted by direction preference:
    
    (num_cells,num_conditions) = np.shape(cells_by_condition)
    
    all_neg = np.sum(cells_by_condition < 0.0,axis=1) == num_conditions
    
    all_neg_idx = np.argwhere(all_neg)[:,0]
    rest_idx = np.setxor1d(all_neg_idx,np.arange(num_cells))
    
    reordered_mat = cells_by_condition[rest_idx]
    reordered_mat = np.append(reordered_mat,cells_by_condition[all_neg_idx],axis=0)

    return reordered_mat

def plot_full_waterfall(cells_by_condition,
                       cre,
                       save_name,
                       scale,
                       savepath,
                       do_colorbar=False):
    
    if scale=='event':
        resp_max = 3.0
        resp_min = -3.0
    else:
        resp_max = 4.0
        resp_min = -4.0
    
    directions,contrasts = grating_params()
    num_contrasts = len(contrasts)
    num_directions = len(directions)
    
    (num_cells,num_conditions) = np.shape(cells_by_condition)
    
    cre_colors = get_cre_colors()
    
    plt.figure(figsize=(10,4))
    ax = plt.subplot(111)
    im = ax.imshow(cells_by_condition,vmin=resp_min,vmax=resp_max,interpolation='nearest',aspect='auto',cmap='RdBu_r')
    
    #dividing lines between contrasts
    for i_con in range(num_contrasts-1):
        ax.plot([(i_con+1)*num_directions-0.5,(i_con+1)*num_directions-0.5],[0,num_cells-1],'k',linewidth=2.0)
    
    layer_str = 'L2/3'
    if shorthand(cre)=='Sst' or shorthand(cre)=='Rorb':
        layer_str = 'L4'
    elif shorthand(cre)=='Rbp4':
        layer_str = 'L5'
    elif shorthand(cre)=='Ntsr1':
        layer_str = 'L6'
    
    ax.set_ylabel(layer_str+' '+shorthand(cre)+ ' neuron number',fontsize=14,color=cre_colors[cre],labelpad=-6)
    ax.set_xlabel('Contrast (%)',fontsize=14,labelpad=-5)
    
    ax.set_xticks(num_directions*np.arange(num_contrasts)+(num_directions/2)-0.5)
    ax.set_xticklabels([str(int(100*x)) for x in contrasts],fontsize=12)
    
    ax.set_yticks([0,num_cells-1])
    ax.set_yticklabels(['0',str(num_cells-1)],fontsize=12)
    
    if do_colorbar:
        if scale=='event':
            fraction_change_ticks = [-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0]
            
            cbar = plt.colorbar(im,ax=ax,ticks=fraction_change_ticks,orientation='horizontal')
            cbar.ax.set_xticklabels([str(round(x,1)) for x in fraction_change_ticks],fontsize=12) 
            cbar.set_label('Fraction Change',fontsize=16,rotation=0,labelpad=15.0)
        else:
            percentile_ticks = [0.0001,0.001,0.01,0.1,0.5,0.9,0.99,0.999,0.9999]
            NLL_ticks = percentile_to_NLL(percentile_ticks,num_shuffles=200000)
            
            cbar = plt.colorbar(im,ax=ax,ticks=NLL_ticks,orientation='horizontal')
            cbar.ax.set_xticklabels([str(100*x) for x in percentile_ticks],fontsize=12) 
            cbar.set_label('Response Percentile',fontsize=16,rotation=0,labelpad=15.0)
    
    plt.tight_layout()
    plt.savefig(savepath+save_name+'_'+scale+'.svg',format='svg')
    #plt.savefig(savepath+save_name+'_'+scale+'.png',dpi=300)
    plt.close()
    
def make_direction_legend(savepath):
    
    directions,contrasts = grating_params()
    num_conditions = len(directions)*len(contrasts)
    
    directions = [225,270,315,0,45,90,135,180]
    arrow_length = 0.4
    empty_im = np.zeros((num_conditions,num_conditions))
    plt.figure(figsize=(10,10))
    ax = plt.subplot(111)
    ax.imshow(empty_im,vmin=-1.0,vmax=1.0,interpolation='none',aspect='auto',cmap='RdBu_r')
    for i_con in range(len(contrasts)):
        for i_dir in range(len(directions)):
            x_center = i_dir+i_con*len(directions)
            y_center = np.shape(empty_im)[0]/2
            
            dx = arrow_length * np.cos(directions[i_dir]*np.pi/180.)
            dy = arrow_length * np.sin(directions[i_dir]*np.pi/180.)
            
            x = x_center - dx/2.0
            y = y_center - dy/2.0
            
            ax.arrow(x,y,dx,dy,head_width=0.2)
            
    plt.savefig(savepath+'arrow_legend.svg',format='svg')
    plt.close()

def plot_direction_vector_sum_by_contrast(df,savepath):
    
    areas, cres = dataset_params()
    directions,contrasts = grating_params()
    
    for area in areas:
        for cre in cres:
            session_IDs = get_sessions(df,area,cre)
            
            if len(session_IDs)>0:
            
                resp, blank, p_all = pool_sessions(session_IDs,area+'_'+cre,savepath,scale='event')     
                sig_resp = resp[p_all<SIG_THRESH]
                
                pref_dir_mat = calc_pref_direction_dist_by_contrast(sig_resp)  
                pref_dir_mat = pref_dir_mat / np.sum(pref_dir_mat,axis=0,keepdims=True)
                
                resultant_mag = []
                resultant_theta = []
                for i_con,contrast in enumerate(contrasts):
                    mag,theta = calc_vector_sum(pref_dir_mat[:,i_con])
                    resultant_mag.append(mag)
                    resultant_theta.append(theta)
                
                #bootstrap CI for distribution at 5% contrast
                num_cells = len(sig_resp)
                uniform_LB, uniform_UB = uniform_direction_vector_sum(num_cells)
                
                radial_direction_figure(np.zeros((len(directions),)),
                                        np.zeros((len(directions),)),
                                        resultant_mag,
                                        resultant_theta,
                                        uniform_LB,
                                        uniform_UB,
                                        cre,
                                        num_cells,
                                        shorthand(area)+'_'+shorthand(cre)+'_combined',
                                        savepath)       

def calc_vector_sum(fraction_prefer_directions):
    
    directions,contrasts = grating_params()
    
    x_coor = []
    y_coor = []
    for i_dir,direction in enumerate(directions):
        direction_magnitude = fraction_prefer_directions[i_dir]
        x_coor.append(direction_magnitude*np.cos(-np.pi*direction/180.0))
        y_coor.append(direction_magnitude*np.sin(-np.pi*direction/180.0))
    x_coor = np.array(x_coor)
    y_coor = np.array(y_coor)
    
    resultant_x = x_coor.sum()
    resultant_y = y_coor.sum()
    magnitude = np.sqrt(resultant_x**2 + resultant_y**2)
    
    if resultant_x==0.0:
        ratio = 1.0*np.sign(resultant_y)
    else:
        ratio = resultant_y/resultant_x
    theta = -np.arctan(ratio)
    
    return magnitude, theta

def radial_direction_figure(x_coor,
                            y_coor,
                            resultant_mag,
                            resultant_theta,
                            CI_LB,
                            CI_UB,
                            cre,
                            num_cells,
                            savename,
                            savepath,
                            max_radius=0.75):
    
    color = get_cre_colors()[cre]
    
    directions,contrasts = grating_params()
    
    unit_circle_x = np.linspace(-1.0,1.0,100)
    unit_circle_y = np.sqrt(1.0-unit_circle_x**2)
    
    plt.figure(figsize=(4,4))
    ax = plt.subplot(111)
    
    outer_CI = Circle((0,0),CI_UB/max_radius,facecolor=[0.7,0.7,0.7])
    inner_CI = Circle((0,0),CI_LB/max_radius,facecolor=[1.0,1.0,1.0])
    ax.add_patch(outer_CI)
    ax.add_patch(inner_CI)
    
    #spokes
    for i,direction in enumerate(directions):
        ax.plot([0,np.cos(np.pi*direction/180.0)],
                [0,np.sin(np.pi*direction/180.0)],
                'k--',linewidth=1.0)
    
    #outer ring
    ax.plot(unit_circle_x,unit_circle_y,'k',linewidth=2.0)
    ax.plot(unit_circle_x,-unit_circle_y,'k',linewidth=2.0)
    
    ax.plot(0.25*unit_circle_x/max_radius,0.25*unit_circle_y/max_radius,'--k',linewidth=1.0)
    ax.plot(0.25*unit_circle_x/max_radius,-0.25*unit_circle_y/max_radius,'--k',linewidth=1.0)
    
    ax.plot(0.5*unit_circle_x/max_radius,0.5*unit_circle_y/max_radius,'--k',linewidth=1.0)
    ax.plot(0.5*unit_circle_x/max_radius,-0.5*unit_circle_y/max_radius,'--k',linewidth=1.0)
    
    ax.plot(unit_circle_x,unit_circle_y,'k',linewidth=2.0)
    ax.plot(unit_circle_x,-unit_circle_y,'k',linewidth=2.0)
    
    #center
    ax.plot(unit_circle_x/200.0,unit_circle_y/200.0,'k',linewidth=2.0)
    ax.plot(unit_circle_x/200.0,-unit_circle_y/200.0,'k',linewidth=2.0)
    
    ax.plot(np.array(x_coor)/max_radius,np.array(y_coor)/max_radius,color=color,linewidth=2.0)
    
    contrast_colors = get_contrast_colors()
    for i,mag in enumerate(resultant_mag[::-1]):
        ax.arrow(0,0,
                 mag*np.cos(-resultant_theta[len(contrasts)-i-1])/(max_radius),
                 mag*np.sin(-resultant_theta[len(contrasts)-i-1])/(max_radius),
                 color=contrast_colors[i],
                 linewidth=2.5,
                 head_width=0.03,
                 alpha=1.0)

    #labels
    ax.text(0,1.02,'U',fontsize=12,horizontalalignment='center',verticalalignment='bottom')
    ax.text(0,-1.02,'D',fontsize=12,horizontalalignment='center',verticalalignment='top')
    ax.text(1.02,0,'T',fontsize=12,verticalalignment='center',horizontalalignment='left')
    ax.text(-1.02,0,'N',fontsize=12,verticalalignment='center',horizontalalignment='right')
    ax.text(-1,0.99,shorthand(cre),fontsize=16,horizontalalignment='left')
    ax.text(0.73,0.99,'(n='+str(num_cells)+')',fontsize=10,horizontalalignment='left')
    
    ax.text(.73,-.73,'45',fontsize=10,horizontalalignment='left',verticalalignment='top')
    ax.text(-.78,-.75,'135',fontsize=10,horizontalalignment='right',verticalalignment='top')
    ax.text(-.73,.73,'-135',fontsize=10,verticalalignment='bottom',horizontalalignment='right')
    ax.text(.73,.73,'-45',fontsize=10,verticalalignment='bottom',horizontalalignment='left')
    ax.text(.81,-.71,'$^\circ$',fontsize=18,horizontalalignment='left',verticalalignment='top')
    ax.text(-.69,-.73,'$^\circ$',fontsize=18,horizontalalignment='right',verticalalignment='top')
    ax.text(-.64,.69,'$^\circ$',fontsize=18,verticalalignment='bottom',horizontalalignment='right')
    ax.text(.85,.69,'$^\circ$',fontsize=18,verticalalignment='bottom',horizontalalignment='left')
        
    ax.set_xlim(-1.2,1.2)
    ax.set_ylim(-1.2,1.2)
    plt.axis('equal')
    plt.axis('off')
    plt.savefig(savepath+savename+'_radial_direction_tuning.svg',format='svg')
    plt.close()
    
def get_contrast_colors():
    
    directions,contrasts = grating_params()
    
    contrast_colors = []
    cmap = matplotlib.cm.get_cmap('bwr')
    for i in range(len(contrasts)):
        color_frac = i/float(len(contrasts)-1)
        contrast_color = cmap(color_frac)
        
        contrast_color = [1.0-color_frac,0.0,color_frac]
        contrast_colors.append(contrast_color)
        
    return contrast_colors

def make_radial_plot_legend(savepath,legend_savename='contrast_vector_legend.svg'):

    directions,contrasts = grating_params()
    
    contrast_colors = get_contrast_colors()
    
    plt.figure(figsize=(2,2))
    ax = plt.subplot(111)
    for i_con,contrast in enumerate(contrasts[::-1]):
        ax.arrow(0,
                 0.07+i_con/6.0,
                 0.2,
                 0,
                 color=contrast_colors[i_con],
                 linewidth=2.0)
        ax.text(0.3,
                0.07+i_con/6.0,
                str(int(100*contrast))+'% contrast',
                fontsize=10,
                verticalalignment='center',
                horizontalalignment='left')
    plt.axis('off')
    plt.savefig(savepath+legend_savename,format='svg')
    plt.close()

def plot_contrast_CoM(df,savepath,curve='cdf'):
    
    areas, cres = dataset_params()
    cre_colors = get_cre_colors()
    
    area = 'VISp'
    pooled_resp = []
    colors = []
    alphas = []
    cre_labels = []
    for cre in cres:
        session_IDs = get_sessions(df,area,cre)
        
        resp, blank, p_all = pool_sessions(session_IDs,area+'_'+cre,savepath,scale='event')
        pooled_resp.append(resp[p_all<SIG_THRESH])
        colors.append(cre_colors[cre])
        alphas.append(1.0)
        cre_labels.append(shorthand(cre))
        
    center_of_mass = center_of_mass_for_list(pooled_resp)
    
    contrasts = [5,10,20,40,60,80]
    
    plot_cdf(metric=center_of_mass,
             metric_labels=cre_labels,
             colors=colors,
             alphas=alphas,
             hist_range=(np.log(5.0),np.log(70.0)),
             hist_bins=200,
             x_label='Contrast (CoM)',
             x_ticks=np.log(contrasts),
             x_tick_labels=[str(x) for x in contrasts],
             save_name=shorthand(area)+'_contrast_'+curve,
             savepath=savepath,
             do_legend=True)

def plot_cdf(metric,
             metric_labels,
             colors,
             alphas,
             hist_range,
             hist_bins,
             x_label,
             x_ticks,
             x_tick_labels,
             save_name,
             savepath,
             do_legend=True):
    
    plt.figure(figsize=(6,4))
    ax = plt.subplot2grid((2,3),(0,0),rowspan=2,colspan=2)
    
    num_pdfs = len(metric)
    for i in range(num_pdfs):
        pdf_y, pdf_x = np.histogram(metric[i],range=hist_range,bins=hist_bins)
        pdf_y = 100.0*pdf_y /np.sum(pdf_y)
        cdf_y = pdf_y.copy()
        for i_bin in range(len(pdf_y)):
            cdf_y[i_bin] = np.sum(pdf_y[:(i_bin+1)])
        ax.plot(pdf_x[:-1]+(pdf_x[1]-pdf_x[0])/2.0,cdf_y,
                linewidth=3.0,color=colors[i],alpha=alphas[i])
    
    ax.set_ylabel('Percent of neurons',fontsize=14)
    ax.set_xlabel(x_label,fontsize=14)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels,fontsize=10)
    ax.set_yticks(np.arange(0,120,20))
    ax.set_yticklabels([str(x) for x in np.arange(0,120,20)],fontsize=10)
    ax.set_ylim(0,105)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if do_legend:
        for i_label,metric_label in enumerate(metric_labels):
            ax.text(x_ticks[0],100-i_label*8,metric_label,fontsize=10,color=colors[i_label])
    plt.savefig(savepath+save_name+'.svg',format='svg')
    plt.close()

def plot_SbC_stats(df,savepath):
    
    SbC_THRESH = 0.05
    
    cre_colors = get_cre_colors()
    directions, contrasts = grating_params()
    
    areas, cres = dataset_params()
    percent_SbC = []
    labels = []
    colors = []
    sample_size = []
    for area in areas:
        for cre in cres:
    
            session_IDs = get_sessions(df,area,cre)
            
            if len(session_IDs)>0:
                
                num_cells = 0
                num_SbC = 0
                for session_ID in session_IDs:
                    SbC_pval = test_SbC(session_ID,savepath)
                    num_cells += len(SbC_pval)
                    num_SbC += (SbC_pval<SbC_THRESH).sum()
                    
                labels.append(shorthand(cre))
                colors.append(cre_colors[cre])
                percent_SbC.append(100.0*num_SbC/num_cells)
                sample_size.append(num_cells)
           
    plt.figure(figsize=(6,4))
    ax = plt.subplot(111)
    for x,group in enumerate(labels):
        ax.bar(x,percent_SbC[x],color=colors[x])
        ax.text(x,max(percent_SbC[x],5)+1,
                '('+str(sample_size[x])+')',
                horizontalalignment='center',
                fontsize=8)
    ax.plot([-1,len(labels)],[100*SbC_THRESH,100*SbC_THRESH],'--k',linewidth=2.0)
    ax.set_ylim(0,30)
    ax.set_xlim(-1,14)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels,fontsize=10,rotation=45)
    ax.set_ylabel('Percent of neurons suppressed',fontsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(savepath+'SbC_stats.svg',format='svg')
    plt.close()

def plot_OSI_distribution(df,savepath,curve='cdf'):
    
    areas, cres = dataset_params()
    cre_colors = get_cre_colors()
    
    area = 'VISp'
    pooled_OSI = []
    colors = []
    alphas = []
    cre_labels = []
    for cre in cres:
        session_IDs = get_sessions(df,area,cre)
        
        resp, blank, p_all = pool_sessions(session_IDs,area+'_'+cre,savepath,scale='event')
        
        osi = calc_OSI(resp[p_all<SIG_THRESH])
        
        pooled_OSI.append(osi)
        colors.append(cre_colors[cre])
        alphas.append(1.0)
        cre_labels.append(shorthand(cre))
       
    xticks = [0.0,0.2,0.4,0.6,0.8,1.0]
    
    plot_cdf(metric=pooled_OSI,
             metric_labels=cre_labels,
             colors=colors,
             alphas=alphas,
             hist_range=(0.0,1.0),
             hist_bins=200,
             x_label='gOSI',
             x_ticks=xticks,
             x_tick_labels=[str(x) for x in xticks],
             save_name='V1_gOSI_'+curve,
             savepath=savepath,
             do_legend=False)

def plot_DSI_distribution(df,savepath,curve='cdf'):
    
    areas, cres = dataset_params()
    cre_colors = get_cre_colors()
    
    area = 'VISp'
    pooled_DSI = []
    colors = []
    alphas = []
    cre_labels = []
    for cre in cres:
        session_IDs = get_sessions(df,area,cre)
        
        resp, blank, p_all = pool_sessions(session_IDs,area+'_'+cre,savepath,scale='event')
        
        dsi = calc_DSI(resp[p_all<SIG_THRESH])
        
        pooled_DSI.append(dsi)
        colors.append(cre_colors[cre])
        alphas.append(1.0)
        cre_labels.append(shorthand(cre))
    
    xticks = [0.0,0.2,0.4,0.6,0.8,1.0]  
    
    plot_cdf(metric=pooled_DSI,
                 metric_labels=cre_labels,
                 colors=colors,
                 alphas=alphas,
                 hist_range=(0.0,1.0),
                 hist_bins=200,
                 x_label='DSI',
                 x_ticks=xticks,
                 x_tick_labels=[str(x) for x in xticks],
                 save_name='V1_DSI_'+curve,
                 savepath=savepath,
                 do_legend=False)

def plot_tuning_split_by_run_state(df,savepath):
    
    running_threshold = 1.0# cm/s
    directions,contrasts = grating_params()
    
    MIN_SESSIONS = 3
    MIN_CELLS = 3#per session
    
    areas, cres = dataset_params()
    for area in areas:
        for cre in cres[:4]:
    
            session_IDs = get_sessions(df,area,cre)
            num_sessions = len(session_IDs)
            
            if num_sessions >= MIN_SESSIONS:
            
                curve_dict = {}
                num_sessions_included = 0
                for i_session,session_ID in enumerate(session_IDs):
                    
                    sweep_table = load_sweep_table(savepath,session_ID)
                    mse = load_mean_sweep_events(savepath,session_ID)
                    condition_responses, blank_responses = compute_mean_condition_responses(sweep_table,mse)
                    
                    p_all = chi_square_all_conditions(sweep_table,mse,session_ID,savepath)
                    all_idx = np.argwhere(p_all<SIG_THRESH)[:,0]
                    
                    mean_sweep_running = load_mean_sweep_running(session_ID,savepath)
                    is_run = mean_sweep_running >= running_threshold
                    
                    run_responses, stat_responses, run_blank, stat_blank = condition_response_running(sweep_table,mse,is_run) 
                    
                    condition_responses = center_direction_zero(condition_responses)
                    run_responses = center_direction_zero(run_responses)
                    stat_responses = center_direction_zero(stat_responses)
                    
                    peak_dir, __ = get_peak_conditions(condition_responses)
                    
                    run_responses = scale_to_percent_per_second(run_responses)
                    stat_responses = scale_to_percent_per_second(stat_responses)
                    run_blank = scale_to_percent_per_second(run_blank)
                    stat_blank = scale_to_percent_per_second(stat_blank)
                    
                    if len(all_idx) >= MIN_CELLS:
                        curve_dict = populate_curve_dict(curve_dict,run_responses,run_blank,all_idx,'all_run',peak_dir)
                        curve_dict = populate_curve_dict(curve_dict,stat_responses,stat_blank,all_idx,'all_stat',peak_dir)
                        num_sessions_included += 1
                    
                if num_sessions_included >= MIN_SESSIONS:
                    plot_from_curve_dict(curve_dict,'all',area,cre,num_sessions_included,savepath)

def scale_to_percent_per_second(responses):
    frac_to_percent = 100.0
    frames_per_sec = 30.0
    return frac_to_percent*frames_per_sec*responses

def calc_pref_direction_dist_by_contrast(condition_responses):
    
    directions,contrasts = grating_params()
    
    pref_dir_mat = np.zeros((len(directions),len(contrasts)))
    for i_con,contrast in enumerate(contrasts):
        
        max_resps = np.max(condition_responses[:,:,i_con],axis=1)
        num_same_max = np.sum(condition_responses[:,:,i_con]==max_resps.reshape(len(condition_responses),1),axis=1)
        
        #multi peak cells: distribute across the directions with the same response magnitude
        multi_peak_cells = np.argwhere(num_same_max>1)[:,0]
        for nc in range(len(multi_peak_cells)):
            is_same_as_max = condition_responses[multi_peak_cells[nc],:,i_con]==max_resps[multi_peak_cells[nc]]
            cell_same_maxes = np.argwhere(is_same_as_max)[:,0]
            pref_dir_mat[cell_same_maxes,i_con]+= 1.0/len(cell_same_maxes)
        
        #one peak cells
        one_peak_cells = np.argwhere(num_same_max==1)[:,0]
        pref_dir_at_con = np.argmax(condition_responses[one_peak_cells,:,i_con],axis=1)
        for i_dir,direction in enumerate(directions):
            pref_dir_mat[i_dir,i_con] += np.sum(pref_dir_at_con==i_dir)

    return pref_dir_mat

def uniform_direction_vector_sum(num_cells,num_shuffles=1000,CI_range=0.95):
    #calculates the bounds of confidence interval for a null population with a uniform distribution
    # of direction preferences
    
    directions,contrasts = grating_params()
    
    # fam_a: prob of at least one type 1 error
    # (1-fam_a): prob of no type 1 errors, also = (1-a)^n
    # a = 1 - (1-fam_a)^(1/n)
    familywise_alpha = 1.0-CI_range
    one_contrast_alpha = 1.0 - (1.0 - familywise_alpha)**(1.0/len(contrasts))
    
    num_shuffles = int(100/one_contrast_alpha) #want at least 100 samples in tail
    
    #one-tailed 
    UB_idx = int(num_shuffles*(1.0-one_contrast_alpha))
    
    vector_sum = []
    for ns in range(num_shuffles):
        uniform_directions = directions[np.random.choice(len(directions),size=num_cells)]
        magnitude, __ = calc_resultant(uniform_directions)
        vector_sum.append(magnitude)
    vector_sum = np.array(vector_sum)
    
    sorted_shuffles = np.sort(vector_sum)
    CI_UB = sorted_shuffles[UB_idx]
    CI_LB = 0.0
    
    return CI_LB, CI_UB

def direction_bias_across_sessions(df,savepath):
    
    cre_colors = get_cre_colors()

    area = 'VISp'
    cre = 'Vip-IRES-Cre'

    session_IDs = get_sessions(df,area,cre)
    
    if len(session_IDs)>0:
        
        session_peak_dir = []
        for session_ID in session_IDs:
            sweep_table = load_sweep_table(savepath,session_ID)
            mse = load_mean_sweep_events(savepath,session_ID)
            session_resp, session_blank = compute_mean_condition_responses(sweep_table,mse)
            session_pvals = chi_square_all_conditions(sweep_table,mse,session_ID,savepath)
            d,c = get_peak_conditions(session_resp[session_pvals<SIG_THRESH])

            session_peak_dir.append(d)
        
        radial_session2session_plot(session_peak_dir,
                                      cre_colors[cre],
                                      cre,
                                      shorthand(area)+'_'+shorthand(cre),
                                      savepath
                                      )

def radial_session2session_plot(session_peak_dir,
                                color,
                                cre,
                                savename,
                                savepath,
                                max_radius=1.0,
                                save_format='svg'):
    
    directions,contrasts = grating_params()
    
    unit_circle_x = np.linspace(-1.0,1.0,100)
    unit_circle_y = np.sqrt(1.0-unit_circle_x**2)
    
    plt.figure(figsize=(4,4))
    ax = plt.subplot(111)
    
    #spokes
    for i,direction in enumerate(directions):
        ax.plot([0,np.cos(np.pi*direction/180.0)],
                [0,np.sin(np.pi*direction/180.0)],
                'k--',linewidth=1.0)
    
    #outer ring
    ax.plot(unit_circle_x,unit_circle_y,'k',linewidth=2.0)
    ax.plot(unit_circle_x,-unit_circle_y,'k',linewidth=2.0)
    
    ax.plot(0.25*unit_circle_x/max_radius,0.25*unit_circle_y/max_radius,'--k',linewidth=1.0)
    ax.plot(0.25*unit_circle_x/max_radius,-0.25*unit_circle_y/max_radius,'--k',linewidth=1.0)
    
    ax.plot(0.5*unit_circle_x/max_radius,0.5*unit_circle_y/max_radius,'--k',linewidth=1.0)
    ax.plot(0.5*unit_circle_x/max_radius,-0.5*unit_circle_y/max_radius,'--k',linewidth=1.0)
    
    ax.plot(0.75*unit_circle_x/max_radius,0.75*unit_circle_y/max_radius,'--k',linewidth=1.0)
    ax.plot(0.75*unit_circle_x/max_radius,-0.75*unit_circle_y/max_radius,'--k',linewidth=1.0)
    
    ax.plot(unit_circle_x,unit_circle_y,'k',linewidth=2.0)
    ax.plot(unit_circle_x,-unit_circle_y,'k',linewidth=2.0)
    
    #center
    ax.plot(unit_circle_x/200.0,unit_circle_y/200.0,'k',linewidth=2.0)
    ax.plot(unit_circle_x/200.0,-unit_circle_y/200.0,'k',linewidth=2.0)
    
    head = 0.0
    for ns,peak_dir in enumerate(session_peak_dir):
        magnitude, theta = calc_resultant(directions[peak_dir])
        ax.arrow(0,0,
                 magnitude*np.cos(-theta)/(max_radius),magnitude*np.sin(-theta)/(max_radius),
                 color=color,linewidth=3,head_length=head,length_includes_head=True)
    
    ax.text(0,1.02,'U',fontsize=12,horizontalalignment='center',verticalalignment='bottom')
    ax.text(0,-1.02,'D',fontsize=12,horizontalalignment='center',verticalalignment='top')
    ax.text(1.02,0,'T',fontsize=12,verticalalignment='center',horizontalalignment='left')
    ax.text(-1.02,0,'N',fontsize=12,verticalalignment='center',horizontalalignment='right')
    ax.text(-1,0.95,shorthand(cre),fontsize=16,horizontalalignment='left')
    
    ax.set_xlim(-1.2,1.2)
    ax.set_ylim(-1.2,1.2)
    plt.axis('equal')
    plt.axis('off')
    
    if save_format=='svg':
        plt.savefig(savepath+savename+'_radial_session2session.svg',format='svg')
    else:
        plt.savefig(savepath+savename+'_radial_session2session.png',dpi=300)
    plt.close()

def LP_HP_stats(df,plot_type,savepath):
    
    best_model_dict = {}
    
    areas, cres = dataset_params()
    for area in areas:
        for cre in cres:
            session_IDs = get_sessions(df,area,cre)
    
            if len(session_IDs) > 0:
    
                area_cre_best_model = np.array([])
                for session_ID in session_IDs:
                   
                    LP_c50, HP_c50, BP_rise_c50, BP_fall_c50, model_AIC = LP_HP_model_selection(session_ID,savepath)
                    p_all = chi_square_all_conditions(None,None,session_ID,savepath)
                    
                    best_model = np.argmin(model_AIC,axis=1)
                    best_model[p_all>=0.01] = -1
                    
                    area_cre_best_model = np.append(area_cre_best_model,best_model)
                    
                best_model_dict[area+'_'+cre] = area_cre_best_model
    
    model_by_cre_barplot(df,best_model_dict,savepath)

def model_by_cre_CI_plot(df,savepath,num_shuffles=10000,save_format='svg'):
    
    plt.figure(figsize=(6,4))
    ax = plt.subplot(111)
    
    labels = []
    
    all_shuffle_HP = []
    session_HP_frac = []
    session_LP_frac = []
    specimen_IDs_by_cre = []
    
    x_vals = np.arange(6)*0.6
    
    areas, cres = dataset_params()
    for i_a,area in enumerate(['VISp']):
        for i_c,cre in enumerate(cres):
            session_IDs = get_sessions(df,area,cre)
    
            if len(session_IDs) > 0:
                
                #shuffle fractions per session
                shuffle_LP = np.zeros((len(session_IDs),num_shuffles))
                shuffle_HP = np.zeros((len(session_IDs),num_shuffles))
                specimen_IDs = np.zeros((len(session_IDs),))
                HP_frac = np.zeros((len(session_IDs),))
                LP_frac = np.zeros((len(session_IDs),))
                for i_session, session_ID in enumerate(session_IDs):
                    LP_c50, HP_c50, BP_rise_c50, BP_fall_c50, model_AIC = LP_HP_model_selection(session_ID,savepath)
                    p_all = chi_square_all_conditions(None,None,session_ID,savepath)
                    
                    best_model = np.argmin(model_AIC,axis=1)
                    best_model[p_all>=0.01] = -1
                    
                    num_LP, num_HP, num_BP, num_NS, total = model_counts(best_model)
                
                    shuffle_LP[i_session,:] = resample_fraction(num_LP,total-num_NS,num_shuffles)
                    shuffle_HP[i_session,:] = resample_fraction(num_HP,total-num_NS,num_shuffles)
                    
                    specimen_ID = df['specimen_id'][df['ophys_session_id']==session_ID].values[0]
                    specimen_IDs[i_session] = specimen_ID
                    HP_frac[i_session] = num_HP/(1.0*(total-num_NS))
                    LP_frac[i_session] = num_LP/(1.0*(total-num_NS))
                
                session_HP_frac.append(HP_frac)
                session_LP_frac.append(LP_frac)
                specimen_IDs_by_cre.append(specimen_IDs)
                
                #shuffle sessions
                shuffle_LP = resample_sessions(shuffle_LP,specimen_IDs)
                shuffle_HP = resample_sessions(shuffle_HP,specimen_IDs)
                
                all_shuffle_HP.append(shuffle_HP)
                
                LP_mean, LP_LB_err, LP_UB_err = resample_CI(shuffle_LP)
                HP_mean, HP_LB_err, HP_UB_err = resample_CI(shuffle_HP)
                
                LP_err = np.zeros((2,1))
                LP_err[0] = LP_LB_err
                LP_err[1] = LP_UB_err
                
                HP_err = np.zeros((2,1))
                HP_err[0] = HP_LB_err
                HP_err[1] = HP_UB_err
                
                ax.plot(x_vals[i_c]-0.05,100*LP_mean,'ob')
                ax.plot(x_vals[i_c]-0.05,100*HP_mean,'or')
                ax.errorbar(x_vals[i_c]-0.05,100*LP_mean,yerr=100*LP_err,color='b',capsize=2.0)
                ax.errorbar(x_vals[i_c]-0.05,100*HP_mean,yerr=100*HP_err,color='r',capsize=2.0)
                    
                labels.append(shorthand(cre))
          
    for i_cre,cre in enumerate(cres):
        unique_mice = np.unique(specimen_IDs_by_cre[i_cre])
        HP_fracs = session_HP_frac[i_cre]
        LP_fracs = session_LP_frac[i_cre]
        mouse_percent_HP = np.zeros((len(unique_mice),))
        mouse_percent_LP = np.zeros((len(unique_mice),))
        for i_mouse,specimen_ID in enumerate(unique_mice):
            mouse_sessions = np.argwhere(specimen_IDs_by_cre[i_cre]==specimen_ID)[:,0]
            mouse_percent_HP[i_mouse] = np.mean(HP_fracs[mouse_sessions])
            mouse_percent_LP[i_mouse] = np.mean(LP_fracs[mouse_sessions])
        ax.plot(0.08*np.random.uniform(size=(len(unique_mice),))+0.1+x_vals[i_cre]*np.ones((len(unique_mice),)),100*mouse_percent_LP,'ob',markersize=2.0)
        ax.plot(0.08*np.random.uniform(size=(len(unique_mice),))+0.1+x_vals[i_cre]*np.ones((len(unique_mice),)),100*mouse_percent_HP,'or',markersize=2.0)
            
    pairwise_pvals = bootstrap_differences(all_shuffle_HP)
    print(pairwise_pvals)        
    
    add_comparison_bar(ax,x_vals[2],x_vals[3],102.5,comparison_bar_pval_text(pairwise_pvals[2,3],num_shuffles))
    add_comparison_bar(ax,x_vals[2],x_vals[4],100.5,comparison_bar_pval_text(pairwise_pvals[2,4],num_shuffles))
    add_comparison_bar(ax,x_vals[2],x_vals[5],98.5,comparison_bar_pval_text(pairwise_pvals[2,5],num_shuffles))
    add_comparison_bar(ax,x_vals[3],x_vals[5],94,comparison_bar_pval_text(pairwise_pvals[3,5],num_shuffles))
    add_comparison_bar(ax,x_vals[3],x_vals[4],89,comparison_bar_pval_text(pairwise_pvals[3,4],num_shuffles))
    add_comparison_bar(ax,x_vals[4],x_vals[5],89,comparison_bar_pval_text(pairwise_pvals[4,5],num_shuffles))
    
    ax.set_xticks(x_vals)
    ax.set_xticklabels(labels,fontsize=10,rotation=45)  
    ax.set_ylabel('Percent of responsive neurons',fontsize=14)
    ax.set_ylim(-1,103)
    ax.set_xlim(-0.3,5.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    if save_format=='svg':
        plt.savefig(savepath+'model_by_cre_CI_plot.svg',format='svg')
    else:
        plt.savefig(savepath+'model_by_cre_CI_plot.png',dpi=300)            
    
    plt.close()
    
def comparison_bar_pval_text(p_val,num_shuffles):
    rounded_p_val = round(p_val,int(np.log10(num_shuffles))-1)
    if rounded_p_val >= 0.05:
        return 'n.s.'
    elif rounded_p_val == 0.0:
        return 'p < ' + str(round(1.0/(num_shuffles/10),int(np.log10(num_shuffles))-1))
    return 'p = ' + str(rounded_p_val)
    
def add_comparison_bar(ax,x1,x2,y,text):
    
    str_len = len(text)
    ax.plot([x1+0.1,x2-0.1],[y,y],'k')
    ax.plot([x1+0.1,x1+0.1],[y-0.6,y],'k')
    ax.plot([x2-0.1,x2-0.1],[y-0.6,y],'k')
    ax.text((x1+x2)/2.0-0.05*str_len/2.0,y+0.8,text,fontsize=9)
    
def bootstrap_differences(all_shuffle_HP):
    
    num_cre = len(all_shuffle_HP)
    pairwise_pvals = np.ones((num_cre,num_cre))
    
    for i_cre in range(num_cre):
        for j_cre in range(num_cre):
            if i_cre < j_cre:
                pval = pairwise_diff(all_shuffle_HP[i_cre],all_shuffle_HP[j_cre])
                pairwise_pvals[i_cre,j_cre] = pval
                pairwise_pvals[j_cre,i_cre] = pval
    
    return pairwise_pvals

def pairwise_diff(shuffle_HP_1,shuffle_HP_2):
    
    diffs = shuffle_HP_1 - shuffle_HP_2
    pval = (diffs < 0).mean()
    if pval > 0.5:
        pval = 1.0 - pval
    return pval
    
def resample_CI(shuffle_samples,CI_range=0.90):
    
    num_shuffles = len(shuffle_samples)
    
    LB_idx = int(num_shuffles*(1.0-CI_range)/2.0)
    UB_idx = int(num_shuffles*(1.0-(1.0-CI_range)/2.0))
    sorted_shuffles = np.sort(shuffle_samples)
    
    CI_UB = sorted_shuffles[UB_idx]
    CI_LB = sorted_shuffles[LB_idx]
    
    CI_mean = shuffle_samples.mean()
    LB_err = CI_mean - CI_LB
    UB_err = CI_UB - CI_mean
    
    return CI_mean, LB_err, UB_err
           
def resample_sessions(shuffle_pass,specimen_IDs):
    (num_sessions,num_shuffles) = np.shape(shuffle_pass)
    unique_specimens = np.unique(specimen_IDs)
    num_mice = len(unique_specimens)
    
    # bootstrap by mouse
    shuffled_mice = np.random.choice(num_mice,size=(num_sessions,num_shuffles))
    
    # choose a session for each mouse sample
    shuffled_sessions = np.zeros((num_sessions,num_shuffles))
    for i_mouse, specimen_ID in enumerate(specimen_IDs):
        mouse_sessions = np.argwhere(specimen_IDs==specimen_ID)[:,0]
        num_mouse_sessions = len(mouse_sessions)
        if num_mouse_sessions==1:
            mouse_sessions = mouse_sessions*np.ones((num_sessions,num_shuffles))
        else:
            mouse_sessions = mouse_sessions[np.random.choice(num_mouse_sessions,size=(num_sessions,num_shuffles))]
    
        shuffled_sessions = np.where(shuffled_mice==i_mouse,mouse_sessions,shuffled_sessions)
    
    #replace session numbers with the resampled fractions
    shuffle_fractions = np.zeros((num_sessions,num_shuffles))
    for i_session in range(num_sessions):
        for i_shuffle in range(num_shuffles):
            sess = int(shuffled_sessions[i_session,i_shuffle])
            shuf = np.random.choice(num_shuffles)
            shuffle_fractions[i_session,i_shuffle] = shuffle_pass[sess,shuf]
    
    return shuffle_fractions.mean(axis=0)
                
def resample_fraction(num_pass,total,num_shuffles):
    shuffle_pass = np.random.choice(total,size=(total,num_shuffles),replace=True) < num_pass
    return shuffle_pass.mean(axis=0)

def model_by_cre_barplot(df,best_model_dict,savepath):
    
    plt.figure(figsize=(6,4))
    ax = plt.subplot(111)
    
    labels = []
    bar_loc = []
    
    areas, cres = dataset_params()
    for i_a,area in enumerate(['VISp']):
        for i_c,cre in enumerate(cres):
            session_IDs = get_sessions(df,area,cre)
    
            if len(session_IDs) > 0:
                num_LP, num_HP, num_BP, num_NS, total = model_counts(best_model_dict[area+'_'+cre])

                print(area + ' ' + cre + ' LP: ' + str(num_LP) + ' HP: ' + str(num_HP) + ' BP: ' + str(num_BP) + ' NS: ' + str(num_NS))

                x_pos = (8*i_a+i_c)
                
                labels.append(shorthand(cre))
                bar_loc.append(x_pos)
                
                perc_LP = 100.0*num_LP/(num_LP+num_BP+num_HP)
                perc_BP = 100.0*num_BP/(num_LP+num_BP+num_HP)
                perc_HP = 100.0*num_HP/(num_LP+num_BP+num_HP)
                
                ax.bar(x_pos-2,perc_LP,bottom=0.0,color='b')
                ax.bar(x_pos-2,perc_BP,bottom=perc_LP,color='r')
                ax.bar(x_pos-2,perc_HP,bottom=(perc_LP+perc_BP),color='g')
                
    ax.set_xticks(np.array(bar_loc)-2)
    ax.set_xticklabels(labels,fontsize=10,rotation=45)  
    ax.set_ylabel('Percent of responsive neurons',fontsize=14)
    ax.set_ylim(0.0,100.0)
    ax.set_xlim(-3,12)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(savepath+'model_by_cre_barplot.svg',format='svg')            
    plt.close()
    
def plot_run_split(session_IDs,area,cre,savename,savepath,running_threshold=1.0,MIN_CELLS=3,MIN_SESSIONS=3):

    curve_dict = {}
    num_sessions_included = 0
    for i_session,session_ID in enumerate(session_IDs):
        
        sweep_table = load_sweep_table(savepath,session_ID)
        mse = load_mean_sweep_events(savepath,session_ID)
        condition_responses, blank_responses = compute_mean_condition_responses(sweep_table,mse)
        
        p_all = chi_square_all_conditions(sweep_table,mse,session_ID,savepath)
        all_idx = np.argwhere(p_all<SIG_THRESH)[:,0]
        
        mean_sweep_running = load_mean_sweep_running(session_ID,savepath)
        is_run = mean_sweep_running >= running_threshold
        
        run_responses, stat_responses, run_blank, stat_blank = condition_response_running(sweep_table,mse,is_run) 
        
        condition_responses = center_direction_zero(condition_responses)
        run_responses = center_direction_zero(run_responses)
        stat_responses = center_direction_zero(stat_responses)
        
        peak_dir, __ = get_peak_conditions(condition_responses)
        
        if len(all_idx) >= MIN_CELLS:
            curve_dict = populate_curve_dict(curve_dict,run_responses,run_blank,all_idx,savename+'_run',peak_dir)
            curve_dict = populate_curve_dict(curve_dict,stat_responses,stat_blank,all_idx,savename+'_stat',peak_dir)
            num_sessions_included += 1
        
    if num_sessions_included >= MIN_SESSIONS:
        plot_from_curve_dict(curve_dict,savename,area,cre,num_sessions_included,savepath)

if __name__=='__main__':
    savepath = '/Users/danielm/Desktop/dandi_VIP/'
    the_whole_trick(savepath)
