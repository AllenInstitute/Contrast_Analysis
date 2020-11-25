#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 16:39:54 2019

@author: dan
"""

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from contrast_utils import shorthand, grating_params, get_cre_colors, select_peak_direction, align_to_prefDir
from contrast_metrics import compute_error_curve

def plot_from_curve_dict(curve_dict,pass_type,area,cre,num_sessions,savepath):
    
    plot_pooled_mat(np.nanmean(get_from_curve_dict(curve_dict,'conXdir',pass_type+'_run',''),axis=0),area,cre,pass_type+'_run',savepath)
    plot_pooled_mat(np.nanmean(get_from_curve_dict(curve_dict,'conXdir',pass_type+'_stat',''),axis=0),area,cre,pass_type+'_stat',savepath)
   
    run_contrast_pooled = get_from_curve_dict(curve_dict,'contrast',pass_type+'_run','')
    stat_contrast_pooled = get_from_curve_dict(curve_dict,'contrast',pass_type+'_stat','')
    run_direction_pooled_low = get_from_curve_dict(curve_dict,'direction',pass_type+'_run','low')
    stat_direction_pooled_low = get_from_curve_dict(curve_dict,'direction',pass_type+'_stat','low')
    run_aligned_pooled_low = get_from_curve_dict(curve_dict,'aligned',pass_type+'_run','low')
    stat_aligned_pooled_low = get_from_curve_dict(curve_dict,'aligned',pass_type+'_stat','low')
    run_direction_pooled_high = get_from_curve_dict(curve_dict,'direction',pass_type+'_run','high')
    stat_direction_pooled_high = get_from_curve_dict(curve_dict,'direction',pass_type+'_stat','high')
    run_aligned_pooled_high = get_from_curve_dict(curve_dict,'aligned',pass_type+'_run','high')
    stat_aligned_pooled_high = get_from_curve_dict(curve_dict,'aligned',pass_type+'_stat','high')

    x_con = np.log([0.025,0.05,0.1,0.2,0.4,0.6,0.8])
    contrast_labels = ['blank','5','10','20','40','60','80']
    x_dir = [-45,0,45,90,135,180,225,270,315]
    
    direction_labels = ['blank ','-135','-90','-45','0','45','90','135','180']
    pref_labels = ['blank ','-135','-90','-45','0','45','90','135','180']


    inset_x_dir = [-45,45,135,225,315]
    inset_direction_labels = ['blank','-90','0','90','180']
    inset_x_pref = [-45,45,135,225,315]
    inset_pref_labels = ['blank','-90','0','90','180']
    
    make_errorbars_plot_running(run_contrast_pooled,
                                stat_contrast_pooled,
                                x_con,
                                contrast_labels,
                                'Contrast',
                                area,
                                cre,
                                num_sessions,
                                shorthand(area)+'_'+shorthand(cre)+'_'+pass_type,
                                'contrast',
                                savepath
                                )
                    
    make_errorbars_plot_running(run_direction_pooled_low,
                                stat_direction_pooled_low,
                                x_dir,
                                direction_labels,
                                'Direction',
                                area,
                                cre,
                                num_sessions,
                                shorthand(area)+'_'+shorthand(cre)+'_'+pass_type+'_low',
                                'direction',
                                savepath
                                )
    
    make_errorbars_plot_running(run_aligned_pooled_low,
                                stat_aligned_pooled_low,
                                x_dir,
                                pref_labels,
                                'Direction - Peak',
                                area,
                                cre,
                                num_sessions,
                                shorthand(area)+'_'+shorthand(cre)+'_'+pass_type+'_low',
                                'preferred_direction',
                                savepath
                                )
    
    make_errorbars_plot_running(run_direction_pooled_high,
                                stat_direction_pooled_high,
                                x_dir,
                                inset_direction_labels,
                                'Direction',
                                area,
                                cre,
                                num_sessions,
                                shorthand(area)+'_'+shorthand(cre)+'_'+pass_type+'_high',
                                'direction',
                                savepath,
                                as_inset=True,
                                inset_x_ticks=inset_x_dir
                                )
    
    make_errorbars_plot_running(run_aligned_pooled_high,
                                stat_aligned_pooled_high,
                                x_dir,
                                inset_pref_labels,
                                'Direction - Peak',
                                area,
                                cre,
                                num_sessions,
                                shorthand(area)+'_'+shorthand(cre)+'_'+pass_type+'_high',
                                'preferred_direction',
                                savepath,
                                as_inset=True,
                                inset_x_ticks=inset_x_pref
                                )
   
    plot_peak_response_distribution(run_aligned_pooled_low,
                                    stat_aligned_pooled_low,
                                    run_aligned_pooled_high,
                                    stat_aligned_pooled_high,
                                    area,
                                    cre,
                                    'aligned',
                                    savepath)
    
    plot_peak_response_distribution(run_direction_pooled_low,
                                    stat_direction_pooled_low,
                                    run_direction_pooled_high,
                                    stat_direction_pooled_high,
                                    area,
                                    cre,
                                    'direction',
                                    savepath)
    
def plot_pooled_mat(pooled_mat,area,cre,pass_str,savepath):
    
    max_resp=80.0
    cre_colors = get_cre_colors()
    x_tick_labels = ['-135','-90','-45','0','45','90','135','180']
    
    directions,contrasts = grating_params()
    
    plt.figure(figsize=(4.2,4))
    ax = plt.subplot(111)
    
    current_cmap = matplotlib.cm.get_cmap(name='RdBu_r')
    current_cmap.set_bad(color=[0.8,0.8,0.8])
    im = ax.imshow(pooled_mat.T,
                   vmin=-max_resp,
                   vmax=max_resp,
                   interpolation='nearest',
                   aspect='auto',
                   cmap='RdBu_r',
                   origin='lower')
    ax.set_xlabel('Direction (deg)',fontsize=14)
    ax.set_ylabel('Contrast (%)',fontsize=14)
    ax.set_yticks(np.arange(len(contrasts)))
    ax.set_yticklabels([str(int(100*x)) for x in contrasts],fontsize=10)
    ax.set_xticks(np.arange(len(directions)))
    ax.set_xticklabels(x_tick_labels,fontsize=10)
    ax.set_title(shorthand(cre) + ' population',fontsize=16,color=cre_colors[cre])
    cbar = plt.colorbar(im,
                        ax=ax,
                        ticks=[-max_resp,-max_resp/2.0,0.0,max_resp/2.0,max_resp])
    cbar.set_label('Event magnitude per second (%), blank subtracted', 
                   rotation=270,
                   labelpad=15.0)
    plt.savefig(savepath+shorthand(area)+'_'+shorthand(cre)+'_'+pass_str+'_summed_tuning.svg',format='svg')
    plt.close() 
    
def make_errorbars_plot_running(run_responses,
                                stat_responses,
                                x_values,
                                x_tick_labels,
                                x_label,
                                area,
                                cre,
                                num_sessions,
                                savename,
                                plot_type,
                                savepath,
                                as_inset=False,
                                inset_x_ticks=None
                                ):
    
    cre_colors = get_cre_colors()
    
    if as_inset:
        num_y_ticks = 4
        x_tick_loc = inset_x_ticks
        label_font_size = 24
        tick_font_size = 18
    else:
        num_y_ticks = 4
        x_tick_loc = x_values
        label_font_size = 15.4
        tick_font_size = 12
    
    min_y = 0.0
    max_y = 160#0.042
    
    y_ticks = np.linspace(min_y,150,num=num_y_ticks)  #np.linspace(min_y,0.04,num=num_y_ticks)    
    y_ticks = np.round(y_ticks,decimals=3)
        
    (num_cells,num_conditions) = np.shape(run_responses)
    
    run_means, run_errors = compute_error_curve(run_responses)
    stat_means, stat_errors = compute_error_curve(stat_responses)
    
    min_x = x_values[0] - 0.5 * (x_values[1] - x_values[0])
    max_x = x_values[-1] + 0.5 * (x_values[1] - x_values[0])
    
    plt.figure(figsize=(4.62,4.4))
    ax = plt.subplot(111)
    
    ax.plot([x_values[0],x_values[-1]],
            [run_means[0],run_means[0]],
            color=cre_colors[cre],
            linestyle='dotted',
            linewidth=2.0)
    
    ax.plot([x_values[0],x_values[-1]],
            [stat_means[0],stat_means[0]],
            color=cre_colors[cre],
            linestyle='dotted',
            linewidth=2.0,
            alpha=0.5)
    
    ax.errorbar([x_values[0]],
                [run_means[0]],
                yerr=run_errors[:,0].reshape(2,1),
                color=cre_colors[cre],
                linewidth=3,
                capsize=5, 
                elinewidth=2,
                markeredgewidth=2)
    
    ax.errorbar([x_values[0]],
                [stat_means[0]],
                yerr=stat_errors[:,0].reshape(2,1),
                color=cre_colors[cre],
                linewidth=3,
                capsize=5, 
                elinewidth=2,
                markeredgewidth=2,
                alpha=0.5)
    
    ax.errorbar(x_values[1:],
                run_means[1:],
                yerr=run_errors[:,1:],
                color=cre_colors[cre],
                linewidth=3,
                capsize=5, 
                elinewidth=2,
                markeredgewidth=2)
    
    ax.errorbar(x_values[1:],
                stat_means[1:],
                yerr=stat_errors[:,1:],
                color=cre_colors[cre],
                linewidth=3,
                capsize=5, 
                elinewidth=2,
                markeredgewidth=2,
                alpha=0.5)
    
    if x_label=='Contrast':
        x_label = 'Contrast (%)'
    elif x_label.find('Direction')!=-1:
        x_label = x_label+' (deg)'
    
    if not as_inset:
        ax.set_ylabel('Event magnitude per second (%)',fontsize=label_font_size)
    ax.set_xlabel(x_label,fontsize=label_font_size)
    ax.set_xticks(x_tick_loc)
    ax.set_xticklabels(x_tick_labels,fontsize=tick_font_size)
    ax.set_xlim(min_x,max_x)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(int(x)) for x in y_ticks],fontsize=tick_font_size)
    ax.set_ylim(min_y,max_y)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if plot_type == 'contrast':
        ax.text(x_values[4],0.9*max_y,'n = '+str(num_cells)+' ('+str(num_sessions)+')',fontsize=11,horizontalalignment='center')
        if shorthand(cre)=='Sst':
            ax.text(x_values[0],max_y*0.024/0.042,'run',fontsize=label_font_size,color=cre_colors[cre])
            ax.text(x_values[0],max_y*0.020/0.042,'stat',fontsize=label_font_size,color=cre_colors[cre],alpha=0.5)
        else:
            ax.text(x_values[-2],max_y*0.024/0.042,'run',fontsize=label_font_size,color=cre_colors[cre])
            ax.text(x_values[-2],max_y*0.020/0.042,'stat',fontsize=label_font_size,color=cre_colors[cre],alpha=0.5)
            
    ax.set_aspect('auto')
    plt.tight_layout()
    plt.savefig(savepath+savename+'_'+plot_type+'_errorbars.svg',format='svg')
    plt.close()     

def plot_peak_response_distribution(run_aligned_pooled_low,
                                    stat_aligned_pooled_low,
                                    run_aligned_pooled_high,
                                    stat_aligned_pooled_high,
                                    area,
                                    cre,
                                    savename,
                                    savepath):
    
    directions,contrasts = grating_params()
    
    plt.figure(figsize=(7,4))
    ax = plt.subplot(111)
    
    MAX_CELLS = 15000
    
    cre_colors = get_cre_colors()
    
    resp_dict = {}
    
    BLANK_IDX = 0
    resp_dict = add_group_to_dict(resp_dict,run_aligned_pooled_low,BLANK_IDX,'run blank')
    resp_dict = add_group_to_dict(resp_dict,stat_aligned_pooled_low,BLANK_IDX,'stat blank') 
    
    directions = [-135,-90,-45,0,45,90,135,180]
    contrasts = [0.05,0.8]
    for run_state in ['run','stat']:
        for i_con, contrast in enumerate(contrasts):
            
            if run_state=='run' and contrast==0.05:
                resps = run_aligned_pooled_low
            elif run_state=='run' and contrast==0.8:
                resps = run_aligned_pooled_high
            elif run_state=='stat' and contrast==0.05:
                resps = stat_aligned_pooled_low
            else:
                resps = stat_aligned_pooled_high
            
            for i_dir,direction in enumerate(directions):
                group_name = run_state + ' ' + str(direction) + ' ' + str(int(100*contrast)) + '%'
                resp_dict = add_group_to_dict(resp_dict,resps,1+i_dir,group_name) 
        
    plot_order = [('space1',''),
                  ('run blank',''),
                  ('stat blank',''),
                  ('space2','')]
    curr_space = 3 
    for run_state in ['run','stat']:
        for i_con, contrast in enumerate(contrasts):
            for i_dir,direction in enumerate(directions):
                plot_order.append((run_state + ' ' + str(direction) + ' ' + str(int(100*contrast)) + '%',''))
            plot_order.append(('space'+str(curr_space),''))
            curr_space += 1
    
    colors = ['#9f9f9f']#blanks
    for i in range(len(plot_order)):
        colors.append(cre_colors[cre])                    
    cre_palette = sns.color_palette(colors)
    
    resp_df = pd.DataFrame(np.zeros((MAX_CELLS,3)),columns=('Response to Preferred Direction','cell_type','cre'))
    curr_cell = 0
    labels = []
    x_pos = []
    dist = []
    dir_idx = 0
    for line,(group,cre_name) in enumerate(plot_order):
        if group.find('space')==-1:
            resp_mag = resp_dict[group]
            resp_mag = resp_mag[np.argwhere(np.isfinite(resp_mag))[:,0]]
            num_cells = len(resp_mag)
            resp_df['Response to Preferred Direction'][curr_cell:(curr_cell+num_cells)] = resp_mag
            resp_df['cre'][curr_cell:(curr_cell+num_cells)] = cre_name
            resp_df['cell_type'][curr_cell:(curr_cell+num_cells)] = group
            curr_cell += num_cells
            x_pos.append(line)
            dist.append(resp_mag)
            
            if group.find('blank')!=-1:
                if group.find('run')!=-1:
                    labels.append('run')
                else:
                    labels.append('stat')
            else:
                labels.append(str(directions[dir_idx]))
                dir_idx+=1
                if dir_idx==len(directions):
                    dir_idx=0
                
        else:
            resp_df['Response to Preferred Direction'][curr_cell] = np.NaN
            resp_df['cre'][curr_cell] = 'blank'
            resp_df['cell_type'][curr_cell] = group
            curr_cell+=1

    resp_df = resp_df.drop(index=np.arange(curr_cell,MAX_CELLS))
    
    ax = sns.swarmplot(x='cell_type',
                  y='Response to Preferred Direction',
                  hue='cre',
                  size=1.0,
                  palette=cre_palette,
                  data=resp_df)    
    
    ax.set_xticks(np.array(x_pos))
    ax.set_xticklabels(labels,fontsize=4.5,rotation=0)
    ax.legend_.remove()
    
    for i,d in enumerate(dist):
        plot_quartiles(ax,d,x_pos[i])
        
    ax.set_ylim(-20,400)
    ax.set_ylabel('Event magnitude per second (%)',fontsize=12)
    ax.set_xlabel('Blank     run 5% contrast        run 80% contrast       stat 5% contrast       stat 80% contrast ',fontsize=9)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig(savepath+shorthand(area)+'_'+shorthand(cre)+'_'+savename+'_cell_response_distribution.svg',format='svg')
    plt.close()

def plot_quartiles(ax,dist,x_pos,width=0.4):
    
    if len(dist) > 0:
    
        med = get_quartile(0.5,dist)
        low_quartile = get_quartile(0.25,dist)
        upper_quartile = get_quartile(0.75,dist)
        
        ax.plot([x_pos-width,x_pos+width],[low_quartile,low_quartile],'k',alpha=0.8,linewidth=1.0)
        ax.plot([x_pos-width,x_pos+width],[med,med],'r',alpha=0.8,linewidth=1.0)
        ax.plot([x_pos-width,x_pos+width],[upper_quartile,upper_quartile],'k',alpha=0.8,linewidth=1.0)
        
        ax.plot([x_pos-width,x_pos-width],[low_quartile,upper_quartile],'k',alpha=0.8,linewidth=0.6)
        ax.plot([x_pos+width,x_pos+width],[low_quartile,upper_quartile],'k',alpha=0.8,linewidth=0.6)
    
def get_quartile(frac,dist):
    sorted_dist = np.sort(dist)
    quartile = frac*len(dist)
    lb = int(np.floor(quartile))
    ub = int(np.ceil(quartile))
    return (sorted_dist[lb] + sorted_dist[ub])/2.0 

def add_group_to_dict(resp_dict,responses,condition_idx,group_name) :
    resp_dict[group_name] = responses[:,condition_idx]
    return resp_dict                             

def populate_curve_dict(curve_dict,responses,blank,cell_idx,curve_str,peak_dir):
    
    contrast_responses = select_peak_direction(responses,peak_dir)
    aligned_responses = align_to_prefDir(responses,peak_dir)
    
    curve_dict = add_to_curve_dict(curve_dict,'conXdir',curve_str,'',responses,blank,cell_idx)
    curve_dict = add_to_curve_dict(curve_dict,'contrast',curve_str,'',contrast_responses,blank,cell_idx)
    curve_dict = add_to_curve_dict(curve_dict,'direction',curve_str,'low',responses,blank,cell_idx)
    curve_dict = add_to_curve_dict(curve_dict,'aligned',curve_str,'low',aligned_responses,blank,cell_idx)
    curve_dict = add_to_curve_dict(curve_dict,'direction',curve_str,'high',responses,blank,cell_idx)
    curve_dict = add_to_curve_dict(curve_dict,'aligned',curve_str,'high',aligned_responses,blank,cell_idx)
    
    return curve_dict
    
def add_to_curve_dict(curve_dict,
                      curve_type,
                      curve_str,
                      contrast_regime,
                      condition_responses,
                      blank_responses,
                      cell_idx):
    
    curve_key = curve_str+'_'+curve_type+'_'+'_'+contrast_regime
    
    if curve_type=='contrast':
        tuning_curve = append_blank(condition_responses[cell_idx],blank_responses[cell_idx])
    elif curve_type=='conXdir':
        tuning_curve = condition_responses[cell_idx] - blank_responses[cell_idx].reshape(len(cell_idx),1,1)
    else:#direction or aligned-direction
        tuning_curve = get_direction_tuning(condition_responses,blank_responses,cell_idx,contrast_regime)
    
    if curve_key in curve_dict:
        curve_dict[curve_key] = np.append(curve_dict[curve_key],tuning_curve,axis=0)
    else:
        curve_dict[curve_key] = tuning_curve
    
    return curve_dict

def get_from_curve_dict(curve_dict,curve_type,curve_str,contrast_regime):
    return curve_dict[curve_str+'_'+curve_type+'_'+'_'+contrast_regime]

def get_direction_tuning(condition_responses,blank_responses,cell_idx,contrast_regime):
    
    if contrast_regime=='low':
        contrast_responses = condition_responses[:,:,:2]
    elif contrast_regime=='high':
        contrast_responses = condition_responses[:,:,4:]
        
    contrast_responses = np.nanmean(contrast_responses,axis=2)
    contrast_responses = append_blank(contrast_responses,blank_responses)    
    
    return contrast_responses[cell_idx]
         
def append_blank(responses,blank):
    
    (num_cells,num_conditions) = np.shape(responses)
    appended = np.zeros((num_cells,num_conditions+1))
    appended[:,0] = blank
    appended[:,1:] = responses
    
    return appended