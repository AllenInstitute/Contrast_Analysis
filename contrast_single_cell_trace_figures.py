#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 16:03:10 2020

@author: danielm
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch

import contrast_utils as cu
import contrast_metrics as cm

def plot_single_cell_tuning_curves(session_ID,savepath,cre,example_cell,plot_path,figure_format):
    
    directions, contrasts = cu.grating_params()
    
    mse = 3000.0*cu.load_mean_sweep_events(savepath,session_ID)
    sweep_table = cu.load_sweep_table(savepath,session_ID)

    condition_responses, blank_responses = cm.compute_mean_condition_responses(sweep_table,mse)
    condition_SEM, __ = cm.compute_SEM_condition_responses(sweep_table,mse)
    
    #shift zero to center:
    directions = [-135,-90,-45,0,45,90,135,180]
    condition_resp = cu.center_direction_zero(condition_responses)
    condition_SEM = cu.center_direction_zero(condition_SEM)
    
    #full direction by contrast response heatmap
    plt.figure(figsize=(7,4))
    ax = plt.subplot2grid((5,5),(0,3),rowspan=5,colspan=2)
    im = ax.imshow(condition_resp[example_cell],vmin=0.0,interpolation='nearest',aspect='auto',cmap='plasma')
    ax.set_ylabel('Direction (deg)',fontsize=12)
    ax.set_xlabel('Contrast (%)',fontsize=12)
    ax.set_xticks(np.arange(len(contrasts)))
    ax.set_xticklabels([str(int(100*x)) for x in contrasts],fontsize=12)
    ax.set_yticks(np.arange(len(directions)))
    ax.set_yticklabels([str(x) for x in directions],fontsize=12)  
    cbar = plt.colorbar(im,ax=ax)
    cbar.set_label('Event magnitude per second (%)',fontsize=12)
    
    
    peak_dir_idx, peak_con_idx = cm.get_peak_conditions(condition_resp)
    
    #contrast tuning at peak direction
    contrast_means = condition_resp[example_cell,peak_dir_idx[example_cell],:]
    contrast_SEMs = condition_SEM[example_cell,peak_dir_idx[example_cell],:]
    
    y_max = 1.1*np.max(contrast_means+contrast_SEMs)
    
    ax = plt.subplot2grid((5,5),(0,0),rowspan=2,colspan=2)
    ax.errorbar(np.log(contrasts),contrast_means,contrast_SEMs,linewidth=0.7,color='b')
    ax.plot([np.log(contrasts[0]),np.log(contrasts[-1])],[blank_responses[example_cell],blank_responses[example_cell]],linewidth=0.7,linestyle='--',color='b')
    ax.set_xticks(np.log(contrasts))
    ax.set_xticklabels([str(int(100*x)) for x in contrasts],fontsize=12)
    ax.tick_params(axis='y',labelsize=12)
    ax.set_xlabel('Contrast (%)',fontsize=12)
    ax.set_ylabel('Event magnitude per second (%)  ',fontsize=12)
    ax.set_ylim([0,y_max])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title('@ '+str(directions[peak_dir_idx[example_cell]])+' degrees',fontsize=12)

    #direction tuning at peak contrast
    direction_means = condition_resp[example_cell,:,peak_con_idx[example_cell]]
    direction_SEMs = condition_SEM[example_cell,:,peak_con_idx[example_cell]]
    ax = plt.subplot2grid((5,5),(3,0),rowspan=2,colspan=2)
    ax.errorbar(np.arange(len(directions)),direction_means,direction_SEMs,linewidth=0.7,color='b')
    ax.plot([0,len(directions)-1],[blank_responses[example_cell],blank_responses[example_cell]],linestyle='--',color='b',linewidth=0.7)
    ax.set_xlim(-0.07,7.07)
    ax.set_xticks(np.arange(len(directions)))
    ax.set_xticklabels([str(x) for x in directions],fontsize=12)
    ax.tick_params(axis='y',labelsize=12)
    
    ax.set_xlabel('Direction (deg)',fontsize=12)
    #ax.set_ylabel('Response',fontsize=14)
    ax.set_ylim([0,y_max])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title('@ '+str(int(100*contrasts[peak_con_idx[example_cell]]))+'% contrast',fontsize=12)
    
    
    plt.tight_layout(w_pad=-5.5, h_pad=0.1)
    
    if figure_format=='svg':
        plt.savefig(plot_path+cre+'_'+str(session_ID)+'_cell_'+str(example_cell)+'_tuning_curves.svg',format='svg')
    else:
        plt.savefig(plot_path+cre+'_'+str(session_ID)+'_cell_'+str(example_cell)+'_tuning_curves.png',dpi=300)
    plt.close()
        
def plot_example_trace_with_events(session_ID,
                                   cre,
                                   cell_idx,
                                   example_trace,
                                   savepath,
                                   plot_path,
                                   figure_format,
                                   window_t=60.0,#seconds
                                   frames_per_sec=30.0,
                                   sweep_t=3.0,
                                   prestim_frames=30):
    
    sweep_table = cu.load_sweep_table(savepath,session_ID)
    sweep_events = cu.load_sweep_events(savepath,session_ID)
    traces = cu.load_dff_traces(savepath,session_ID)
    
    num_sweeps_to_plot = int(window_t / sweep_t)
    window_frames = int(window_t*frames_per_sec)
    
    sweep_frames = int(sweep_t*frames_per_sec)
    
    num_examples_to_plot = 10
    if example_trace is not None:
        first_sweep_list = [100+num_sweeps_to_plot*example_trace]
    else:
        first_sweep_list = np.arange(100,100+num_sweeps_to_plot*num_examples_to_plot,num_sweeps_to_plot)
    
    for i_example,first_sweep in enumerate(first_sweep_list):
    
        first_window_frame = int(sweep_table['Start'][first_sweep]) - prestim_frames
        
        trace_t = np.arange(0,window_frames) / frames_per_sec
        trace_dFF = traces[cell_idx,first_window_frame:(first_window_frame+window_frames)]
        
        trace_events = np.zeros((window_frames,))
        for i_sweep in range(num_sweeps_to_plot):
            this_sweep_events = sweep_events[i_sweep+first_sweep,cell_idx]
            trace_events[int(i_sweep*sweep_frames):int((i_sweep+1)*sweep_frames)] = this_sweep_events
        
        fig = plt.figure(figsize=(20,4))
        
        ax1 = plt.subplot2grid((3,1),(0,0),rowspan=2)
        ax1.plot(trace_t,100.0*trace_dFF,linewidth=0.5)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        
        ax1.plot([63,63],[0,50],'k',linewidth=2.0)
        ax1.plot([61,63],[0,0],'k',linewidth=2.0)
        ax1.text(64,25,'50%',fontsize=14)
        ax1.text(62,-12,'2s',fontsize=14)
        
        ax2 = plt.subplot2grid((3,1),(2,0),rowspan=1,sharex=ax1)
        
        #plot individual events so lines are completely vertical
        ax2.plot([trace_t[0],trace_t[-1]],[0,0],'k',linewidth=0.5)
        events = np.argwhere(trace_events>0.0)[:,0]
        for event_idx in events:
            ax2.plot([trace_t[event_idx],trace_t[event_idx]],[0,100.0*trace_events[event_idx]],'k',linewidth=0.5)
        
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
    
        ax2.plot([63,63],[0,25],'k',linewidth=2.0)
        ax2.text(64,12.5,'25%',fontsize=14)
    
        ax1.get_yaxis().set_visible(False)
        ax1.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        ax2.get_xaxis().set_visible(False)
        
        fig.subplots_adjust(hspace=0) 
    
        plt.tight_layout()
        
        if figure_format=='svg':
            plt.savefig(plot_path+cre+'_'+str(session_ID)+'_cell_'+str(cell_idx)+'_trace_with_events_'+str(i_example)+'.svg',format='svg')
        else:
            plt.savefig(plot_path+cre+'_'+str(session_ID)+'_cell_'+str(cell_idx)+'_trace_with_events_'+str(i_example)+'.png',dpi=300)
        
        plt.close()
        
def plot_rasters_at_peak(sweep_table,
                         sweep_events,
                         condition_responses,
                         cell_idx,
                         session_ID,
                         cre,
                         savepath,
                         figure_format,
                         max_sweeps=15):

    peak_dir,peak_con = cu.get_peak_conditions(condition_responses)

    directions, contrasts = cu.grating_params()
    
    fig = plt.figure(figsize=(15,5))
    
    cell_max = get_max_event_magnitude(sweep_events,cell_idx)
    
    direction_str = ['0','45','90','135','180','-135','-90','-45']
    dir_shift = 3
    
    ax = plt.subplot(1,10,10)
    plot_blank_raster(ax,sweep_table,sweep_events,cell_idx,cell_max)
    ax.set_xlabel('Time (s)',fontsize=16)
    ax.set_title('Blanks',fontsize=16)
    
    for i_con,contrast in enumerate(contrasts):
        ax = plt.subplot(2,10,2+i_con)
        
        plot_single_raster(ax,sweep_table,sweep_events,cell_idx,contrast,directions[peak_dir[cell_idx]],cell_max)
        
        ax.set_title(str(int(100*contrast))+'%',fontsize=16)
        ax.set_xlabel('Time (s)',fontsize=16)
        
        if i_con==0:
            ax.set_ylabel('Direction: '+direction_str[peak_dir[cell_idx]]+'$^\circ$',fontsize=16)
        
    for i_dir, direction in enumerate(directions):
        
        #shift 0-degrees to center
        plot_dir = np.mod(i_dir+dir_shift,len(directions))
        
        ax = plt.subplot(2,10,11+plot_dir)
        
        plot_single_raster(ax,sweep_table,sweep_events,cell_idx,contrasts[peak_con[cell_idx]],direction,cell_max)

        ax.set_title(' '+direction_str[i_dir]+'$^\circ$',fontsize=16)
        ax.set_xlabel('Time (s)',fontsize=16)
        
        if plot_dir==0:
            ax.set_ylabel('Contrast: '+str(int(100*contrasts[peak_con[cell_idx]]))+'%',fontsize=16)

    fig.subplots_adjust(hspace=0)
    plt.tight_layout()
    
    if figure_format=='svg':
        plt.savefig(savepath+cre+'_'+str(session_ID)+'_cell_'+str(cell_idx)+'_rasters_at_peak.svg',format='svg')
    else:
        plt.savefig(savepath+cre+'_'+str(session_ID)+'_cell_'+str(cell_idx)+'_rasters_at_peak.png',dpi=300)
        
    plt.close()      
        
def plot_rasters_across_contrasts(sweep_table,
                                  sweep_events,
                                  cell_idx,
                                  session_ID,
                                  cre,
                                  savepath):
    
    directions, contrasts = cu.grating_params()
    
    plt.figure(figsize=(16,16))
    
    cell_max = get_max_event_magnitude(sweep_events,cell_idx)
    
    direction_str = ['0','45','90','135','180','-135','-90','-45']
    dir_shift = 3
    
    for i_con,contrast in enumerate(contrasts):
        
        for i_dir, direction in enumerate(directions):
        
            #shift 0-degrees to center
            plot_dir = np.mod(i_dir+dir_shift,len(directions))
            
            ax = plt.subplot(len(directions),len(contrasts),1+len(contrasts)*plot_dir+i_con)
            
            plot_single_raster(ax,sweep_table,sweep_events,cell_idx,contrast,direction,cell_max)
                
            if plot_dir==7:
                ax.set_xlabel('Time from stimulus onset (s)')
            if plot_dir==0:
                ax.set_title(str(int(100*contrast))+'% Contrast',fontsize=14)
            
            if i_con==0:
                ax.set_ylabel(direction_str[i_dir],fontsize=14)
            
    plt.tight_layout()
    plt.savefig(savepath+cre+'_'+str(session_ID)+'_cell_'+str(cell_idx)+'_rasters.png',dpi=300)
    plt.close()
       
def plot_blank_raster(ax,
                    sweep_table,
                    sweep_events,
                    cell_idx,
                    cell_max,
                    max_sweeps=38):
    
    event_mag, event_t = get_blank_events(sweep_table,
                                              sweep_events,
                                              cell_idx=cell_idx)
                                
    num_sweeps = len(event_mag)
    for i_sweep in range(num_sweeps):
        for i_event, single_mag in enumerate(event_mag[i_sweep]):
            single_t = event_t[i_sweep][i_event]
            plot_single_event(ax,single_mag,single_t,i_sweep,cell_max)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_xlim(-1,2)
    ax.set_ylim(0,max_sweeps)
    
    ax.set_xticks([0,2])
    ax.set_xticklabels(['0','2'],fontsize=14)
    
    ax.set_yticks([0,10,20,30])
    ax.set_yticklabels(['0','10','20','30'],fontsize=14)

    rect = patch.Rectangle((0,0), 2, 30, facecolor=(0.8,0.8,0.8))
    ax.add_patch(rect)
    
def plot_single_raster(ax,
                       sweep_table,
                       sweep_events,
                       cell_idx,
                       contrast,
                       direction,
                       cell_max,
                       max_sweeps=15):
    
    event_mag, event_t = get_condition_events(sweep_table,
                                              sweep_events,
                                              cell_idx=cell_idx,
                                              contrast=contrast,
                                              direction=direction)
                                
    num_sweeps = len(event_mag)
    for i_sweep in range(num_sweeps):
        for i_event, single_mag in enumerate(event_mag[i_sweep]):
            single_t = event_t[i_sweep][i_event]
            plot_single_event(ax,single_mag,single_t,i_sweep,cell_max)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_xlim(-1,2)
    ax.set_ylim(0,max_sweeps)
    
    ax.set_xticks([0,2])
    ax.set_xticklabels(['0','2'],fontsize=14)
    
    ax.set_yticks([0,5,10,15])
    ax.set_yticklabels(['0','5','10','15'],fontsize=14)

    rect = patch.Rectangle((0,0), 2, max_sweeps, facecolor=(0.8,0.8,0.8))
    ax.add_patch(rect)
    
def plot_single_event(ax,
                      single_mag,
                      single_t,
                      i_sweep,
                      cell_max,
                      max_height=0.9):
    
    tick_height = max_height * single_mag / cell_max
    
    ax.plot([single_t,single_t],[i_sweep+0.5-tick_height/2.0,i_sweep+0.5+tick_height/2.0],'r',linewidth=0.4)
    
def plot_traces_across_contrasts(sweep_table,traces,cell_idx,session_ID,cre,savepath):
    
    directions, contrasts = cu.grating_params()
    
    plt.figure(figsize=(8,16))
    
    min_val = 0
    max_val = 0
    
    for i_con,contrast in enumerate(contrasts):
        
        for i_dir, direction in enumerate(directions):
        
            ax = plt.subplot(len(directions),len(contrasts),1+len(contrasts)*i_dir+i_con)
            
            ct, t = get_condition_traces(sweep_table,
                                         traces,
                                         cell_idx=cell_idx,
                                         contrast=contrast,
                                         direction=direction)
        
            #make raster
            (num_sweeps,num_frames) = ct.shape
            for i_sweep in range(num_sweeps):
                ax.plot(t,100.0*ct[i_sweep])
                
            if i_dir==7:
                ax.set_xlabel('Time from stimulus onset (s)')
            if i_dir==0:
                ax.set_title(str(int(100*contrast))+'% Contrast')
            
            if i_con==0:
                ax.set_ylabel(str(int(direction)))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            ax.set_xticks([0,2])
            ax.set_xticklabels(['0','2'])
            
            if min_val > ct.min():
                min_val = ct.min()
            if max_val < ct.max():
                max_val = ct.max()
    
        
    for i_con in range(len(contrasts)):
        for i_dir, direction in enumerate(directions):
            ax = plt.subplot(len(directions),len(contrasts),1+len(contrasts)*i_dir+i_con)
            ax.set_ylim([100*min_val, 100*max_val])
        
            rect = patch.Rectangle((0,100*min_val), 2, 100*(max_val-min_val), facecolor=(0.8,0.8,0.8))
            ax.add_patch(rect)
        
            
        
    plt.tight_layout()
    plt.savefig(savepath+cre+'_'+str(session_ID)+'_cell_'+str(cell_idx)+'_traces.png',dpi=300)
    plt.close()

def get_max_event_magnitude(sweep_events,cell_idx):
    
    cell_max = 0.0
    for i_sweep in range(len(sweep_events)):
        sweep_max = np.max(sweep_events[i_sweep,cell_idx])
        if sweep_max > cell_max:
            cell_max = sweep_max
    
    return cell_max

def get_blank_events(sweep_table,
                    sweep_events,
                    cell_idx=0,
                    prestim_frames = 30,
                    stim_frames = 60,
                    frames_per_sec = 30.0
                    ):
    
    is_blank = np.isnan(sweep_table['Ori'].values)
    blank_sweeps = np.argwhere(is_blank)[:,0]
    
    sweep_t = np.arange(-prestim_frames,stim_frames) / frames_per_sec
    
    event_mag = []
    event_t = []
    for i_sweep, sweep in enumerate(blank_sweeps):
        this_sweep = sweep_events[sweep,cell_idx]
        event_frames = np.where(this_sweep)[0]
        if len(event_frames)==0:
            event_mag.append([])
            event_t.append([])
        else:
            event_mag.append(this_sweep[event_frames])
            event_t.append(sweep_t[event_frames])
            
    return event_mag, event_t

def get_condition_events(sweep_table,
                         sweep_events,
                         cell_idx=0,
                         contrast=0.05,
                         direction=0,
                         prestim_frames = 30,
                         stim_frames = 60,
                         frames_per_sec = 30.0
                         ):
    
    is_direction = sweep_table['Ori'].values == direction
    is_contrast = sweep_table['Contrast'].values == contrast
    
    condition_sweeps = np.argwhere(is_direction & is_contrast)[:,0]
    
    sweep_t = np.arange(-prestim_frames,stim_frames) / frames_per_sec
    
    event_mag = []
    event_t = []
    for i_sweep, sweep in enumerate(condition_sweeps):
        this_sweep = sweep_events[sweep,cell_idx]
        event_frames = np.where(this_sweep)[0]
        if len(event_frames)==0:
            event_mag.append([])
            event_t.append([])
        else:
            event_mag.append(this_sweep[event_frames])
            event_t.append(sweep_t[event_frames])
            
    return event_mag, event_t

def get_condition_traces(sweep_table,
                         traces,
                         cell_idx=0,
                         contrast=0.05,
                         direction=0,
                         frames_per_sec = 30.0,
                         prestim_t = 1.0,#seconds
                         stim_t = 2.0,#seconds
                         poststim_t = 1.0#seconds
                         ):
    
    is_direction = sweep_table['Ori'].values == direction
    is_contrast = sweep_table['Contrast'].values == contrast
    
    condition_sweeps = np.argwhere(is_direction & is_contrast)[:,0]

    num_sweeps = len(condition_sweeps)
    
    prestim_frames = int(prestim_t * frames_per_sec)
    stim_frames = int(stim_t * frames_per_sec)
    poststim_frames = int(poststim_t * frames_per_sec)
    num_frames = prestim_frames + stim_frames + poststim_frames
    sweep_t = np.arange(-prestim_frames,stim_frames+poststim_frames) / frames_per_sec
    
    condition_traces = np.zeros((num_sweeps,num_frames))
    for i_sweep, sweep in enumerate(condition_sweeps):
        start = sweep_table['Start'][sweep]
        
        condition_traces[i_sweep] = traces[cell_idx,int(start-prestim_frames):int(start+stim_frames+poststim_frames)]

    return condition_traces, sweep_t