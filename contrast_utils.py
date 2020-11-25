#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 16:44:52 2019

@author: dan
"""

import os, sys
import numpy as np
import pandas as pd
import h5py

import simplejson

def center_of_mass_for_list(response_list):
    
    center_of_mass = []
    for i,condition_responses in enumerate(response_list):
        peak_dir, peak_con = get_peak_conditions(condition_responses)
        
        (num_cells,num_directions,num_contrasts) = np.shape(condition_responses)
        
        plot_mat = np.zeros((num_cells,num_contrasts))
        for nc in range(num_cells):
            plot_mat[nc] = condition_responses[nc,peak_dir[nc],:]
            
        plot_mat, CoM, __ = rectified_sort_by_center_of_mass(plot_mat)

        center_of_mass.append(CoM)
        
    return center_of_mass

def rectified_sort_by_center_of_mass(cells_by_condition):
    
    contrasts = np.array([5,10,20,40,60,80])
    
    rect_responses = np.where(cells_by_condition>0,cells_by_condition,0.0)
    rect_responses = rect_responses / np.sum(rect_responses,axis=1,keepdims=True)
    condition_weights = np.log(contrasts).reshape(1,6)
    center_of_mass = np.sum(rect_responses * condition_weights,axis=1)
    sort_idx = np.argsort(center_of_mass)
    
    return cells_by_condition[sort_idx], center_of_mass[sort_idx], sort_idx

def center_direction_zero(responses,center_pos=3):
    
    (num_cells,num_dir,num_con) = np.shape(responses)
    
    shifted_responses = responses.copy()
    for direction in range(num_dir):
        shifted_direction = np.mod(direction+center_pos+num_dir,num_dir)
        shifted_responses[:,shifted_direction] = responses[:,direction]
        
    return shifted_responses

def align_to_prefDir(responses,peak_dir):
    
    (num_cells,num_dir,num_con) = np.shape(responses)
    center_pos = 3
    aligned_responses = responses.copy()
    for preferred_dir in range(num_dir):
        cells_pref_dir = np.argwhere(peak_dir == preferred_dir)[:,0]
        aligned_to_dir = responses.copy()
        for actual_dir in range(num_dir):
            align_dir = np.mod(actual_dir-preferred_dir+num_dir+center_pos,num_dir)
            aligned_to_dir[:,align_dir] = responses[:,actual_dir]
        aligned_responses[cells_pref_dir] = aligned_to_dir[cells_pref_dir]
    
    return aligned_responses

def sort_by_weighted_peak_direction(condition_responses):
    
    (num_cells,num_directions) = np.shape(condition_responses)
    
    weighted_resp = np.zeros((num_cells,num_directions))
    relative_pos = np.zeros((num_cells,num_directions))
    for nc in range(num_cells):
        cell_resp = condition_responses[nc,:]
        for d in range(num_directions):
            left_idx = np.mod(d + (num_directions-1),num_directions)
            right_idx = np.mod(d + (num_directions+1),num_directions)
            weighted_resp[nc,d] = cell_resp[d] + (cell_resp[left_idx]+cell_resp[right_idx])/ np.sqrt(2.0) 
            relative_pos[nc,d] = ((cell_resp[right_idx]-cell_resp[left_idx])/np.sqrt(2.0))/ weighted_resp[nc,d]
    
    dir_idx = np.argmax(weighted_resp,axis=1)
    weighted_dir = np.zeros((num_cells,))
    rel_pos = np.zeros((num_cells,))
    for nc in range(num_cells):
        weighted_dir[nc] = dir_idx[nc]
        rel_pos[nc] = relative_pos[nc,dir_idx[nc]]
    
    sort_idx = np.array([],dtype=np.uint8)
    for d in range(num_directions):
        pref_this_dir = np.argwhere(weighted_dir == d)[:,0]
        if len(pref_this_dir)>0:
            dir_sort = np.argsort(rel_pos[pref_this_dir])
            sort_idx = np.append(sort_idx,pref_this_dir[dir_sort])
    
    return sort_idx

def select_peak_direction(condition_responses,peak_direction):
    
    (num_cells,num_directions,num_contrasts) = np.shape(condition_responses)
    
    contrast_responses = np.zeros((num_cells,num_contrasts))
    for nc in range(num_cells):
        contrast_responses[nc] = condition_responses[nc,peak_direction[nc],:]
        
    return contrast_responses

def select_peak_contrast(condition_responses,peak_contrast):
    
    (num_cells,num_directions,num_contrasts) = np.shape(condition_responses)
    
    direction_responses = np.zeros((num_cells,num_directions))
    for nc in range(num_cells):
        direction_responses[nc] = condition_responses[nc,:,peak_contrast[nc]]
        
    return direction_responses

def select_peak_condition(condition_responses,peak_direction,peak_contrast):
    
    (num_cells,num_directions,num_contrasts) = np.shape(condition_responses)
    
    peak_responses = np.zeros((num_cells,))
    for nc in range(num_cells):
        if (peak_direction[nc]==np.NaN) or (peak_contrast[nc]==np.NaN):
            peak_responses[nc] = np.NaN
        else:
            peak_responses[nc] = condition_responses[nc,peak_direction[nc],peak_contrast[nc]]
        
    return peak_responses

def get_peak_conditions(condition_responses):
    
    (num_cells,num_directions,num_contrasts) = np.shape(condition_responses)
    
    peak_direction = np.zeros((num_cells,),dtype=np.uint8)
    peak_contrast = np.zeros((num_cells,),dtype=np.uint8)
    for nc in range(num_cells):
        cell_max = np.nanmax(condition_responses[nc])
        is_max = condition_responses[nc] == cell_max
        
        if is_max.sum()==1:
            (direction,contrast) = np.argwhere(is_max)[0,:]
        else:
            print(str(is_max.sum())+' peaks')
            r = np.random.choice(is_max.sum())
            (direction,contrast) = np.argwhere(is_max)
            print(np.shape(direction))
            direction = direction[r]
            contrast = contrast[r]
        peak_direction[nc] = direction
        peak_contrast[nc] = contrast
        
    return peak_direction, peak_contrast

def load_sweep_table(dandiset_path,session_ID):
    nwb_path = get_nwb_path(dandiset_path,session_ID)
    nwb = h5py.File(nwb_path,mode='r')
    start_times = np.array(nwb['intervals/epochs/start_time'])
    stop_times = np.array(nwb['intervals/epochs/stop_time'])
    contrasts = np.array(nwb['intervals/epochs/contrast'])
    directions = np.array(nwb['intervals/epochs/direction'])
    SFs = np.array(nwb['intervals/epochs/spatial_frequency'])
    TFs = np.array(nwb['intervals/epochs/temporal_frequency'])
    
    df = pd.DataFrame(np.zeros((len(start_times),6)),columns=('Start','End','Contrast','Ori','SF','TF'))
    df['Start'] = start_times
    df['End'] = stop_times
    df['Contrast'] = contrasts
    df['Ori'] = directions
    df['SF'] = SFs
    df['TF'] = TFs
    return df

def load_mean_sweep_events(dandiset_path,session_ID):
    
    if os.path.isfile(dandiset_path+str(session_ID)+'_mean_sweep_events.npy'):
        mse = np.load(dandiset_path+str(session_ID)+'_mean_sweep_events.npy')
    else:
    
        nwb_path = get_nwb_path(dandiset_path,session_ID)
        nwb = h5py.File(nwb_path,mode='r')
        
        events = np.array(nwb['processing/brain_observatory_pipeline/l0_events/true_false_events/data'])
        sweep_table = load_sweep_table(dandiset_path,session_ID)
        
        num_neurons = np.shape(events)[0]
        num_sweeps = len(sweep_table)
        
        mse = np.zeros((num_sweeps,num_neurons))
        for sweep in range(num_sweeps):
            start_frame = int(sweep_table['Start'][sweep])
            end_frame = int(sweep_table['End'][sweep])
            mse[sweep] = np.mean(events[:,start_frame:end_frame],axis=1)
            
        np.save(dandiset_path+str(session_ID)+'_mean_sweep_events.npy',mse)
        
    duplicate_cells = roi_duplications(session_ID,dandiset_path)
    if duplicate_cells is not None:
        unique_cells = np.argwhere(~duplicate_cells)[:,0]
        mse = mse[:,unique_cells]
    
    return mse
    
def load_sweep_events(dandiset_path,session_ID,stim_frames=60,pre_frames=30):
    
    if os.path.isfile(dandiset_path+str(session_ID)+'_sweep_events.npy'):
        se = np.load(dandiset_path+str(session_ID)+'_sweep_events.npy')
    else:
    
        nwb_path = get_nwb_path(dandiset_path,session_ID)
        nwb = h5py.File(nwb_path,mode='r')
        
        events = np.array(nwb['processing/brain_observatory_pipeline/l0_events/true_false_events/data'])
        sweep_table = load_sweep_table(dandiset_path,session_ID)
        
        num_neurons = np.shape(events)[0]
        num_sweeps = len(sweep_table)
        num_sweep_frames = pre_frames+stim_frames
        
        se = np.zeros((num_sweeps,num_neurons,num_sweep_frames))
        for sweep in range(num_sweeps):
            start_frame = int(sweep_table['Start'][sweep])
            end_frame = start_frame+stim_frames
            se[sweep] = events[:,(start_frame-pre_frames):end_frame]
            
        np.save(dandiset_path+str(session_ID)+'_sweep_events.npy',se)
        
    duplicate_cells = roi_duplications(session_ID,dandiset_path)
    if duplicate_cells is not None:
        unique_cells = np.argwhere(~duplicate_cells)[:,0]
        se = se[:,unique_cells]
    
    return se

def load_mean_sweep_running(session_ID,dandiset_path):
    
    if os.path.isfile(dandiset_path+str(session_ID)+'_mean_sweep_running.npy'):
        msr = np.load(dandiset_path+str(session_ID)+'_mean_sweep_running.npy')
    else:
    
        nwb_path = get_nwb_path(dandiset_path,session_ID)
        nwb = h5py.File(nwb_path,mode='r')
        
        running = np.array(nwb['processing/brain_observatory_pipeline/RunningBehavior/running_speed/data'])
        sweep_table = load_sweep_table(dandiset_path,session_ID)
        
        num_sweeps = len(sweep_table)
        
        msr = np.zeros((num_sweeps,))
        for sweep in range(num_sweeps):
            start_frame = int(sweep_table['Start'][sweep])
            end_frame = int(sweep_table['End'][sweep])
            msr[sweep] = np.mean(running[start_frame:end_frame])
            
        np.save(dandiset_path+str(session_ID)+'_mean_sweep_running.npy',msr)
    
    return msr

def load_dff_traces(dandiset_path,session_ID):
    
    nwb_path = get_nwb_path(dandiset_path,session_ID)
    nwb = h5py.File(nwb_path,mode='r')
    
    dff = np.array(nwb['processing/brain_observatory_pipeline/Fluorescence/DfOverF/data'])
        
    duplicate_cells = roi_duplications(session_ID,dandiset_path)
    if duplicate_cells is not None:
        unique_cells = np.argwhere(~duplicate_cells)[:,0]
        dff = dff[unique_cells]
    
    return dff

def get_nwb_path(dandiset_path,session_ID):
    for sess in os.listdir(dandiset_path):
        session_path = os.path.join(dandiset_path,sess)
        if os.path.isdir(session_path):
            for f in os.listdir(session_path):
                if f.find(str(session_ID))!=-1 and f.find('.nwb')!=-1:
                    return os.path.join(session_path,f)
    return None

def get_sessions(df,area,cre):

    is_area = df['targeted_structure'] == area
    is_cre = df['genotype_name'] == cre
    session_ids = df['ophys_session_id'][is_area&is_cre].astype(np.int)

    return np.unique(session_ids)
    
def get_analysis_QCd(savepath):
    
    manifest = pd.read_csv(savepath+'targeted_manifest.csv')
    manifest = manifest.drop(columns=('Unnamed: 0'))
    manifest = manifest.drop_duplicates(subset=('ophys_session_id'))
    
    return manifest

def roi_duplications(session_ID,savepath):
    
    duplicate_cells = pd.read_csv(savepath+'duplicate_rois.csv')
    is_session = duplicate_cells['session_ID'].values == session_ID
    session_idx = np.argwhere(is_session)[:,0]
    if len(session_idx)>0:
        return duplicate_cells['is_duplicate'].values[session_idx]
    else:
        return None

#def roi_duplications(session_ID,savepath):
#    
#    if os.path.isfile(savepath+str(session_ID)+'_duplicate_cells.npy'):
#        print('Session ' + str(session_ID) + ' duplicate cells taken into account.')
#        return np.load(savepath+str(session_ID)+'_duplicate_cells.npy')
#    return None

def shorthand(name):
    
    sh = {}

    sh['Cux2-CreERT2'] = 'Cux2'
    sh['Rorb-IRES2-Cre'] = 'Rorb'
    sh['Rbp4-Cre_KL100'] = 'Rbp4'
    sh['Ntsr1-Cre_GN220'] = 'Ntsr1'
    sh['Sst-IRES-Cre'] = 'Sst'
    sh['Vip-IRES-Cre'] = 'Vip'
    sh['VISp'] = 'V1'
    
    return sh[name]

def dataset_params():
    
    areas = ['VISp']

    cres = ['Vip-IRES-Cre',
            'Sst-IRES-Cre',
            'Cux2-CreERT2',
            'Rorb-IRES2-Cre',
            'Rbp4-Cre_KL100',
            'Ntsr1-Cre_GN220']     
            
    return areas, cres

def grating_params():
    
    directions = np.arange(0,360,45)
    contrasts = np.array([0.05,0.1,0.2,0.4,0.6,0.8])
    
    return directions, contrasts

def get_cre_colors():
    '''returns dictionary of colors for specific Cre lines
        Returns
        -------
        cre color dictionary
    '''
    
    cre_colors = {}
    cre_colors['Cux2-CreERT2'] = '#a92e66'
    cre_colors['Rorb-IRES2-Cre'] = '#7841be'
    cre_colors['Rbp4-Cre_KL100'] = '#5cad53'
    cre_colors['Ntsr1-Cre_GN220'] = '#ff3b39'
    cre_colors['Sst-IRES-Cre'] = '#7B5217'
    cre_colors['Vip-IRES-Cre'] = '#b49139'
    
    return cre_colors