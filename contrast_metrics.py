#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 16:54:03 2019

@author: dan
"""

import os
import numpy as np

from contrast_utils import grating_params, load_sweep_table, load_mean_sweep_events, get_peak_conditions
import chisq_categorical as chi

def compute_error_curve(pooled_responses,use_CI=False):
    
    (num_cells,num_conditions) = np.shape(pooled_responses)
    
    means = np.nanmean(pooled_responses,axis=0)
 
    errors = np.zeros((2,num_conditions))
    if use_CI:
        CI_LB, CI_UB = compute_CI(pooled_responses,allow_nans=True)
        errors[0] = means - CI_LB
        errors[1] = CI_UB - means
    else:
        SEMs = np.nanstd(pooled_responses,axis=0) / np.sqrt(np.sum(np.isfinite(pooled_responses),axis=0))
        errors[:] = (SEMs).reshape(1,num_conditions) 
    
    return means, errors

def compute_CI(responses,interval=0.9,allow_nans=False):
    
    trials_outside_interval = 100
    frac_outside_interval = 1.0 - interval
    num_shuffles = int(trials_outside_interval/frac_outside_interval)
    
    (num_cells,num_conditions) = np.shape(responses)
    
    bootstrapped_means = np.zeros((num_shuffles,num_conditions))
    for ns in range(num_shuffles):
        shuffle_responses = responses[np.random.choice(num_cells,size=num_cells)]
        if allow_nans:
            bootstrapped_means[ns] = np.nanmean(shuffle_responses,axis=0)
        else:
            bootstrapped_means[ns] = np.mean(shuffle_responses,axis=0)
    
    tail_size = (1.0-interval)/2.0
    LB_idx = int(num_shuffles*tail_size)
    UB_idx = int(num_shuffles*(1.0-tail_size))
    
    lower_bounds = []
    upper_bounds = []
    for condition in range(num_conditions):
        sorted_means = np.sort(bootstrapped_means[:,condition])
        lower_bounds.append(sorted_means[LB_idx])
        upper_bounds.append(sorted_means[UB_idx])
    
    return np.array(lower_bounds), np.array(upper_bounds)

def calc_resultant(directions):
    
    x_comp = np.cos(np.pi*directions/180.0)
    y_comp = np.sin(np.pi*directions/180.0)
    
    resultant_magnitude = np.sqrt(x_comp.mean()**2+y_comp.mean()**2)
    resultant_theta = np.arctan(y_comp.mean()/x_comp.mean())
    
    return resultant_magnitude, resultant_theta

def calc_DSI(condition_responses):
    
    (num_cells,num_directions,num_contrasts) = np.shape(condition_responses)
    peak_dir, peak_con = get_peak_conditions(condition_responses)
    
    DSI = np.zeros((num_cells,))
    for nc in range(num_cells):
        cell_resp = condition_responses[nc,:,peak_con[nc]]
        pref_resp = cell_resp[peak_dir[nc]]
        null_resp = cell_resp[np.mod(peak_dir[nc]+4,8)]
        DSI[nc] = (pref_resp-null_resp) / (pref_resp+null_resp)
        
    return DSI

def calc_OSI(condition_responses):
    
    (num_cells,num_directions,num_contrasts) = np.shape(condition_responses)
    peak_dir, peak_con = get_peak_conditions(condition_responses)
    
    directions, contrasts = grating_params()
    radians_per_degree = np.pi/180.0
    x_comp = np.cos(2.0*directions*radians_per_degree)
    y_comp = np.sin(2.0*directions*radians_per_degree)
    
    OSI = np.zeros((num_cells,))
    for nc in range(num_cells):
        cell_resp = condition_responses[nc,:,peak_con[nc]]
        normalized_resp = cell_resp / cell_resp.sum()

        x_proj = normalized_resp * x_comp
        y_proj = normalized_resp * y_comp
    
        OSI[nc] = np.sqrt(x_proj.sum()**2 + y_proj.sum()**2)
        
    return OSI

def condition_response_running(sweep_table,mean_sweep_events,is_run):           
    
    MIN_SWEEPS = 4
    directions,contrasts = grating_params()
    (num_sweeps,num_cells) = np.shape(mean_sweep_events)
    
    run_resps = np.zeros((num_cells,len(directions),len(contrasts)))
    stat_resps = np.zeros((num_cells,len(directions),len(contrasts)))
    run_blank_resps = np.zeros((num_cells,))
    stat_blank_resps = np.zeros((num_cells,))
    
    is_blank = sweep_table['Contrast'].isnull().values
    run_blank_sweeps = np.argwhere(is_blank & is_run)[:,0]
    stat_blank_sweeps = np.argwhere(is_blank & ~is_run)[:,0]
    if (len(run_blank_sweeps)>=MIN_SWEEPS) and (len(stat_blank_sweeps)>=MIN_SWEEPS):
        run_blank_resps = mean_sweep_events[run_blank_sweeps].mean(axis=0)
        stat_blank_resps = mean_sweep_events[stat_blank_sweeps].mean(axis=0)
    else:
        run_blank_resps[:] = np.NaN
        stat_blank_resps[:] = np.NaN
        
    for i_dir,direction in enumerate(directions):
        is_direction = sweep_table['Ori'].values == direction
        for i_con,contrast in enumerate(contrasts):
            is_contrast = sweep_table['Contrast'].values == contrast
            
            run_sweeps = np.argwhere(is_direction & is_contrast & is_run)[:,0]
            stat_sweeps = np.argwhere(is_direction & is_contrast & ~is_run)[:,0]
            
            if (len(run_sweeps)>=MIN_SWEEPS) and (len(stat_sweeps)>=MIN_SWEEPS):
                run_resps[:,i_dir,i_con] = mean_sweep_events[run_sweeps].mean(axis=0)
                stat_resps[:,i_dir,i_con] = mean_sweep_events[stat_sweeps].mean(axis=0)
            else:
                run_resps[:,i_dir,i_con] = np.NaN
                stat_resps[:,i_dir,i_con] = np.NaN
                
    return run_resps, stat_resps, run_blank_resps, stat_blank_resps

def compute_mean_condition_responses(sweep_table,mean_sweep_events):
    
    (num_sweeps,num_cells) = np.shape(mean_sweep_events) 
    
    directions, contrasts = grating_params()
    
    condition_responses = np.zeros((num_cells,len(directions),len(contrasts)))
    for i_dir,direction in enumerate(directions):
        is_direction = sweep_table['Ori'] == direction
        for i_con,contrast in enumerate(contrasts):
            is_contrast = sweep_table['Contrast'] == contrast
            is_condition = (is_direction & is_contrast).values
            
            condition_responses[:,i_dir,i_con] = np.mean(mean_sweep_events[is_condition],axis=0)
            
    is_blank = np.isnan(sweep_table['Ori'].values)
    blank_sweep_responses = np.mean(mean_sweep_events[is_blank],axis=0)
            
    return condition_responses, blank_sweep_responses
 
def compute_SEM_condition_responses(sweep_table,mean_sweep_events):
    
    (num_sweeps,num_cells) = np.shape(mean_sweep_events) 
    
    directions, contrasts = grating_params()
    
    condition_responses = np.zeros((num_cells,len(directions),len(contrasts)))
    for i_dir,direction in enumerate(directions):
        is_direction = sweep_table['Ori'] == direction
        for i_con,contrast in enumerate(contrasts):
            is_contrast = sweep_table['Contrast'] == contrast
            is_condition = (is_direction & is_contrast).values
            
            condition_responses[:,i_dir,i_con] = np.std(mean_sweep_events[is_condition],axis=0)/np.sqrt(float(is_condition.sum()))
            
    is_blank = np.isnan(sweep_table['Ori'].values)
    blank_sweep_responses = np.std(mean_sweep_events[is_blank],axis=0)/np.sqrt(float(is_blank.sum()))
            
    return condition_responses, blank_sweep_responses 
 
def test_SbC(session_ID,savepath):
        
    max_contrast_idx = 5
    
    condition_responses, blank_responses = compute_blank_subtracted_NLL(session_ID,savepath)
    peak_dir = np.argmax(condition_responses[:,:,max_contrast_idx],axis=1)
    
    SbC_NLL = []
    for nc,direction in enumerate(peak_dir):
        SbC_NLL.append(condition_responses[nc,direction,max_contrast_idx])
    
    SbC_pvals = NLL_to_percentile(np.array(SbC_NLL))
        
    return SbC_pvals    

def compute_blank_subtracted_NLL(session_ID,savepath,num_shuffles=200000):
    
    if os.path.isfile(savepath+str(session_ID)+'_blank_subtracted_NLL.npy'):
        condition_NLL = np.load(savepath+str(session_ID)+'_blank_subtracted_NLL.npy')
        blank_NLL = np.load(savepath+str(session_ID)+'_blank_subtracted_blank_NLL.npy')
    else:
        sweep_table = load_sweep_table(savepath,session_ID)
        mean_sweep_events = load_mean_sweep_events(savepath,session_ID)
        
        (num_sweeps,num_cells) = np.shape(mean_sweep_events) 
        
        condition_responses, blank_sweep_responses = compute_mean_condition_responses(sweep_table,mean_sweep_events)
        condition_responses = np.swapaxes(condition_responses,0,2)
        condition_responses = np.swapaxes(condition_responses,0,1)
        
        directions, contrasts = grating_params()
        
        # different conditions can have different number of trials...
        trials_per_condition, num_blanks = compute_trials_per_condition(sweep_table)
        unique_trial_counts = np.unique(trials_per_condition.flatten())
        
        trial_count_mat = np.tile(trials_per_condition,reps=(num_cells,1,1))
        trial_count_mat = np.swapaxes(trial_count_mat,0,2)
        trial_count_mat = np.swapaxes(trial_count_mat,0,1)
        
        blank_shuffle_sweeps = np.random.choice(num_sweeps,size=(num_shuffles*num_blanks,))
        blank_shuffle_responses = mean_sweep_events[blank_shuffle_sweeps].reshape(num_shuffles,num_blanks,num_cells)
        blank_null_dist = blank_shuffle_responses.mean(axis=1)
        
        condition_NLL = np.zeros((len(directions),len(contrasts),num_cells))
        for trial_count in unique_trial_counts:
            
            #create null distribution and compute condition NLL
            shuffle_sweeps = np.random.choice(num_sweeps,size=(num_shuffles*trial_count,))
            shuffle_responses = mean_sweep_events[shuffle_sweeps].reshape(num_shuffles,trial_count,num_cells)
            
            null_diff_dist = shuffle_responses.mean(axis=1) - blank_null_dist
            actual_diffs = condition_responses.reshape(len(directions),len(contrasts),1,num_cells) - blank_sweep_responses.reshape(1,1,1,num_cells)
            resp_above_null = null_diff_dist.reshape(1,1,num_shuffles,num_cells) < actual_diffs
            percentile = resp_above_null.mean(axis=2)
            NLL = percentile_to_NLL(percentile,num_shuffles)
        
            has_count = trial_count_mat == trial_count
            condition_NLL = np.where(has_count,NLL,condition_NLL)
            
        #repeat for blank sweeps
        blank_null_dist_2 = blank_null_dist[np.random.choice(num_shuffles,size=num_shuffles),:]
        blank_null_diff_dist = blank_null_dist_2 - blank_null_dist
        actual_diffs = 0.0
        resp_above_null = blank_null_diff_dist < actual_diffs
        percentile = resp_above_null.mean(axis=0)
        blank_NLL = percentile_to_NLL(percentile,num_shuffles)
        
        np.save(savepath+str(session_ID)+'_blank_subtracted_NLL.npy',condition_NLL)
        np.save(savepath+str(session_ID)+'_blank_subtracted_blank_NLL.npy',blank_NLL)
        
    condition_NLL = np.swapaxes(condition_NLL,0,2)
    condition_NLL = np.swapaxes(condition_NLL,1,2)
        
    return condition_NLL, blank_NLL
    
def percentile_to_NLL(percentile,num_shuffles):
    
    percentile = np.where(percentile==0.0,1.0/num_shuffles,percentile)
    percentile = np.where(percentile==1.0,1.0-1.0/num_shuffles,percentile)
    NLL = np.where(percentile<0.5,
                   np.log10(percentile)-np.log10(0.5),
                   -np.log10(1.0-percentile)+np.log10(0.5))
    
    return NLL

def NLL_to_percentile(NLL):
    
    percentile = np.where(NLL<0.0,
                          10.0**(NLL+np.log10(0.5)),
                          1.0-10.0**(np.log10(0.5)-NLL))
    
    return percentile

def compute_trials_per_condition(sweep_table):
    
    directions, contrasts = grating_params()
    trials_per_condition = np.zeros((len(directions),len(contrasts)),dtype=np.int)
    for i_dir,direction in enumerate(directions):
        is_direction = sweep_table['Ori'] == direction
        for i_con,contrast in enumerate(contrasts):
            is_contrast = sweep_table['Contrast'] == contrast
            is_condition = (is_direction & is_contrast).values
            trials_per_condition[i_dir,i_con] = is_condition.sum()

    num_blanks = np.isnan(sweep_table['Ori'].values).sum()

    return trials_per_condition, num_blanks
    
def pool_sessions(session_IDs,pool_name,savepath,scale='blank_subtracted_NLL'):
    
    if os.path.isfile(savepath+pool_name+'_condition_responses_'+scale+'.npy'):
    
        pooled_condition_responses = np.load(savepath+pool_name+'_condition_responses_'+scale+'.npy')
        pooled_blank_responses = np.load(savepath+pool_name+'_blank_responses_'+scale+'.npy')
        p_vals_all = np.load(savepath+pool_name+'_chisq_all.npy')
    
    else:
        
        print('Pooling sessions for ' + pool_name)
        
        directions, contrasts = grating_params()
        
        MAX_CELLS = 5000
        pooled_condition_responses = np.zeros((MAX_CELLS,len(directions),len(contrasts)))
        pooled_blank_responses = np.zeros((MAX_CELLS,))
        p_vals_all = np.zeros((MAX_CELLS,))
        curr_cell = 0
        for session_ID in session_IDs:
            
            print(str(session_ID))
            
            mse = load_mean_sweep_events(savepath,session_ID)
            sweep_table = load_sweep_table(savepath,session_ID)
    
            if scale=='event':
                condition_responses, blank_responses = compute_mean_condition_responses(sweep_table,mse)
            elif scale=='blank_subtracted_NLL':
                condition_responses, blank_responses = compute_blank_subtracted_NLL(session_ID,savepath)
                
            p_all = chi_square_all_conditions(sweep_table,mse,session_ID,savepath)

            session_cells = len(p_all)
            
            pooled_condition_responses[curr_cell:(curr_cell+session_cells)] = condition_responses
            pooled_blank_responses[curr_cell:(curr_cell+session_cells)] = blank_responses
            p_vals_all[curr_cell:(curr_cell+session_cells)] = p_all
            curr_cell += session_cells
            
        pooled_condition_responses = pooled_condition_responses[:curr_cell]
        pooled_blank_responses = pooled_blank_responses[:curr_cell]
        p_vals_all = p_vals_all[:curr_cell]
        
        np.save(savepath+pool_name+'_condition_responses_'+scale+'.npy',pooled_condition_responses)
        np.save(savepath+pool_name+'_blank_responses_'+scale+'.npy',pooled_blank_responses)
        np.save(savepath+pool_name+'_chisq_all.npy',p_vals_all)
    
        print('Done.')
    
    return pooled_condition_responses, pooled_blank_responses, p_vals_all

def chi_square_all_conditions(sweep_table,mean_sweep_events,session_ID,savepath):
    
    if os.path.isfile(savepath+str(session_ID)+'_chisq_all.npy'):
        p_vals = np.load(savepath+str(session_ID)+'_chisq_all.npy')
    else:
        p_vals = chi.chisq_from_stim_table(sweep_table,
                                           ['Ori','Contrast'],
                                           mean_sweep_events)
        
        np.save(savepath+str(session_ID)+'_chisq_all.npy',p_vals)
    
    return p_vals