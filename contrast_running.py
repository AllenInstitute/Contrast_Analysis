#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 16:39:54 2019

@author: dan
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

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
    
    direction_labels = ['blank','-135','-90','-45','0','45','90','135','180']
    pref_labels = ['blank','-135','-90','-45','0','45','90','135','180']


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
   
def plot_pooled_mat(pooled_mat,area,cre,pass_str,savepath):
    
    max_resp=0.02
    cre_colors = get_cre_colors()
    x_tick_labels = ['-135','-90','-45','0','45','90','135','180']
    
    directions,contrasts = grating_params()
    
    plt.figure(figsize=(4.2,4))
    ax = plt.subplot(111)
    
    current_cmap = matplotlib.cm.get_cmap(name='RdBu_r')
    current_cmap.set_bad(color=[0.8,0.8,0.8])
    im = ax.imshow(pooled_mat.T,vmin=-max_resp,vmax=max_resp,interpolation='nearest',aspect='auto',cmap='RdBu_r',origin='lower')
    ax.set_xlabel('Direction (deg)',fontsize=14)
    ax.set_ylabel('Contrast (%)',fontsize=14)
    ax.set_yticks(np.arange(len(contrasts)))
    ax.set_yticklabels([str(int(100*x)) for x in contrasts],fontsize=10)
    ax.set_xticks(np.arange(len(directions)))
    ax.set_xticklabels(x_tick_labels,fontsize=10)
    ax.set_title(shorthand(cre) + ' population',fontsize=16,color=cre_colors[cre])
    cbar = plt.colorbar(im,ax=ax,ticks=[-0.02,-0.01,0.0,0.01,0.02])
    cbar.set_label('Mean event magnitude, blank subtracted (a.u.)', rotation=270,labelpad=15.0)
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
        num_y_ticks = 3
        x_tick_loc = inset_x_ticks
        label_font_size = 22
        tick_font_size = 17
    else:
        num_y_ticks = 5
        x_tick_loc = x_values
        label_font_size = 14
        tick_font_size = 10
    
    min_y = 0.0
    max_y = 0.04
    
    y_ticks = np.linspace(min_y,max_y,num=num_y_ticks)    
    y_ticks = np.round(y_ticks,decimals=3)
        
    (num_cells,num_conditions) = np.shape(run_responses)
    
    run_means, run_errors = compute_error_curve(run_responses)
    stat_means, stat_errors = compute_error_curve(stat_responses)
    
    min_x = x_values[0] - 0.5 * (x_values[1] - x_values[0])
    max_x = x_values[-1] + 0.5 * (x_values[1] - x_values[0])
    
    plt.figure(figsize=(4.2,4))
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
        ax.set_ylabel('Mean event magnitude (a.u.)',fontsize=label_font_size)
    ax.set_xlabel(x_label,fontsize=label_font_size)
    ax.set_xticks(x_tick_loc)
    ax.set_xticklabels(x_tick_labels,fontsize=tick_font_size)
    ax.set_xlim(min_x,max_x)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(x) for x in y_ticks],fontsize=tick_font_size)
    ax.set_ylim(min_y,max_y)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if plot_type == 'contrast':
        ax.text(x_values[4],0.9*max_y,'n = '+str(num_cells)+' ('+str(num_sessions)+')',fontsize=10,horizontalalignment='center')
        if shorthand(cre)=='Sst':
            ax.text(x_values[0],0.024,'run',fontsize=14,color=cre_colors[cre])
            ax.text(x_values[0],0.020,'stat',fontsize=14,color=cre_colors[cre],alpha=0.5)
        else:
            ax.text(x_values[-2],0.024,'run',fontsize=14,color=cre_colors[cre])
            ax.text(x_values[-2],0.020,'stat',fontsize=14,color=cre_colors[cre],alpha=0.5)
            
    ax.set_aspect('auto')
    plt.tight_layout()
    plt.savefig(savepath+savename+'_'+plot_type+'_errorbars.svg',format='svg')
    plt.close()     

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
    
    if curve_dict.has_key(curve_key):
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