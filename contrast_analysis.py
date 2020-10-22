#!/usr/bin/env python2
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
"""
Created on Mon Dec  9 13:05:54 2019

@author: dan
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from contrast_running import populate_curve_dict, plot_from_curve_dict
from contrast_utils import grating_params, get_sessions, get_analysis_QCd, load_mean_sweep_events, load_mean_sweep_running, load_sweep_table, shorthand, dataset_params, get_cre_colors, get_peak_conditions, select_peak_contrast, sort_by_weighted_peak_direction, center_of_mass_for_list, center_direction_zero
from contrast_metrics import compute_mean_condition_responses, compute_SEM_condition_responses, chi_square_all_conditions, pool_sessions, percentile_to_NLL, calc_DSI, calc_OSI, test_SbC, calc_resultant, condition_response_running

SIG_THRESH = 0.01

def the_whole_trick():
    
    savepath = '/Users/dan/Desktop/contrast/'
    df = get_analysis_QCd(savepath)  

   ## FIGURE 1
    #1a
    plot_single_cell_example(df,savepath,cre='Rorb-IRES2-Cre',example_cell=15)
    plot_single_cell_example(df,savepath,cre='Vip-IRES-Cre',example_cell=0)
    #1b
    plot_all_waterfalls(df,savepath,scale='blank_subtracted_NLL')
    make_direction_legend(savepath)
    #1c
    plot_direction_vector_sum_by_contrast(df,savepath)
    make_radial_plot_legend(savepath)
    #1d
    plot_contrast_CoM(df,savepath,curve='cdf')
    #1e
    plot_SbC_stats(df,savepath)
    #1f
    plot_OSI_distribution(df,savepath,curve='cdf')
    #1g
    plot_DSI_distribution(df,savepath,curve='cdf')
    
   ## FIGURE 2
    plot_tuning_split_by_run_state(df,savepath)

def plot_single_cell_example(df,savepath,cre,example_cell,example_session_idx=0):
    
    directions, contrasts = grating_params()
    
    session_IDs = get_sessions(df,'VISp',cre)
    session_ID = session_IDs[example_session_idx]
    
    mse = load_mean_sweep_events(savepath,session_ID)
    sweep_table = load_sweep_table(savepath,session_ID)

    condition_responses, __ = compute_mean_condition_responses(sweep_table,mse)
    condition_SEM, __ = compute_SEM_condition_responses(sweep_table,mse)
    p_all = chi_square_all_conditions(sweep_table,mse,session_ID,savepath)     
    
    sig_resp = condition_responses[p_all<SIG_THRESH]
    sig_SEM = condition_SEM[p_all<SIG_THRESH]    
    
    #shift zero to center:
    directions = [-135,-90,-45,0,45,90,135,180]
    sig_resp = center_direction_zero(sig_resp)
    sig_SEM = center_direction_zero(sig_SEM)
    
    #full direction by contrast response heatmap
    plt.figure(figsize=(6,4))
    ax = plt.subplot2grid((5,5),(0,0),rowspan=5,colspan=2)
    ax.imshow(sig_resp[example_cell],vmin=0.0,interpolation='nearest',aspect='auto',cmap='plasma')
    ax.set_ylabel('Direction (deg)',fontsize=14)
    ax.set_xlabel('Contrast (%)',fontsize=14)
    ax.set_xticks(np.arange(len(contrasts)))
    ax.set_xticklabels([str(int(100*x)) for x in contrasts],fontsize=10)
    ax.set_yticks(np.arange(len(directions)))
    ax.set_yticklabels([str(x) for x in directions],fontsize=10)  
    
    peak_dir_idx, peak_con_idx = get_peak_conditions(sig_resp)
    
    #contrast tuning at peak direction
    contrast_means = sig_resp[example_cell,peak_dir_idx[example_cell],:]
    contrast_SEMs = sig_SEM[example_cell,peak_dir_idx[example_cell],:]
    ax = plt.subplot2grid((5,5),(0,3),rowspan=2,colspan=2)
    ax.errorbar(np.log(contrasts),contrast_means,contrast_SEMs)
    ax.set_xticks(np.log(contrasts))
    ax.set_xticklabels([str(int(100*x)) for x in contrasts],fontsize=10)
    ax.set_xlabel('Contrast (%)',fontsize=14)
    ax.set_ylabel('Response',fontsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    #direction tuning at peak contrast
    direction_means = sig_resp[example_cell,:,peak_con_idx[example_cell]]
    direction_SEMs = sig_SEM[example_cell,:,peak_con_idx[example_cell]]
    ax = plt.subplot2grid((5,5),(3,3),rowspan=2,colspan=2)
    ax.errorbar(np.arange(len(directions)),direction_means,direction_SEMs)
    ax.set_xlim(-0.07,7.07)
    ax.set_xticks(np.arange(len(directions)))
    ax.set_xticklabels([str(x) for x in directions],fontsize=10)
    ax.set_xlabel('Direction (deg)',fontsize=14)
    ax.set_ylabel('Response',fontsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.tight_layout(w_pad=-5.5, h_pad=0.1)
    plt.savefig(savepath+shorthand(cre)+'_example_cell.svg',format='svg')
    plt.close()

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
    
    ax.set_ylabel(shorthand(cre)+ ' cell number',fontsize=14,color=cre_colors[cre],labelpad=-6)
    ax.set_xlabel('Contrast (%)',fontsize=14,labelpad=-5)
    
    ax.set_xticks(num_directions*np.arange(num_contrasts)+(num_directions/2)-0.5)
    ax.set_xticklabels([str(int(100*x)) for x in contrasts],fontsize=12)
    
    ax.set_yticks([0,num_cells-1])
    ax.set_yticklabels(['0',str(num_cells-1)],fontsize=12)
    
    if do_colorbar:
        
        percentile_ticks = [0.0001,0.001,0.01,0.1,0.5,0.9,0.99,0.999,0.9999]
        NLL_ticks = percentile_to_NLL(percentile_ticks,num_shuffles=200000)
        
        cbar = plt.colorbar(im,ax=ax,ticks=NLL_ticks,orientation='horizontal')
        cbar.ax.set_xticklabels([str(100*x) for x in percentile_ticks],fontsize=12) 
        cbar.set_label('Response Percentile',fontsize=16,rotation=0,labelpad=15.0)
    
    plt.tight_layout()
    plt.savefig(savepath+save_name+'_'+scale+'.svg',format='svg')
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
    
    outer_CI = Circle((0,0),CI_UB/max_radius,facecolor=[0.6,0.6,0.6])
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
                 linewidth=2.0,
                 head_width=0.03)

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
    cmap = matplotlib.cm.get_cmap('plasma')
    for i in range(len(contrasts)):
        color_frac = i/float(len(contrasts))
        contrast_color = cmap(color_frac)
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
    
    ax.set_ylabel('Percent of Cells',fontsize=14)
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
    ax.set_ylabel('% Suppressed by Contrast',fontsize=14)
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
        for cre in cres:
    
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
                    
                    if len(all_idx) >= MIN_CELLS:
                        curve_dict = populate_curve_dict(curve_dict,run_responses,run_blank,all_idx,'all_run',peak_dir)
                        curve_dict = populate_curve_dict(curve_dict,stat_responses,stat_blank,all_idx,'all_stat',peak_dir)
                        num_sessions_included += 1
                    
                if num_sessions_included >= MIN_SESSIONS:
                    plot_from_curve_dict(curve_dict,'all',area,cre,num_sessions_included,savepath)

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

def uniform_direction_vector_sum(num_cells,num_shuffles=1000,CI_range=0.9):
    #calculates the bounds of confidence interval for a null population with a uniform distribution
    # of direction preferences
    
    directions,contrasts = grating_params()
    
    vector_sum = []
    for ns in range(num_shuffles):
        uniform_directions = directions[np.random.choice(len(directions),size=num_cells)]
        magnitude, __ = calc_resultant(uniform_directions)
        vector_sum.append(magnitude)
    vector_sum = np.array(vector_sum)
    
    LB_idx = int(num_shuffles*(1.0-CI_range)/2.0)
    UB_idx = int(num_shuffles*(1.0-(1.0-CI_range)/2.0))
    sorted_shuffles = np.sort(vector_sum)
    
    CI_UB = sorted_shuffles[UB_idx]
    CI_LB = sorted_shuffles[LB_idx]
    
    return CI_LB, CI_UB

if __name__=='__main__':
    the_whole_trick()
