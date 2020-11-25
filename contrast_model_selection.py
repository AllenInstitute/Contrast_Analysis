#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 16:10:07 2019

@author: dan
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import least_squares

from contrast_utils import grating_params, load_sweep_table, load_mean_sweep_events, get_peak_conditions
from contrast_metrics import chi_square_all_conditions, compute_mean_condition_responses

SIG_THRESH = 0.01

def get_best_model_idx(session_ID,savepath):
    
    if not os.path.isfile(savepath+str(session_ID)+'_model_AIC.npy'):
        LP_HP_model_selection(session_ID,savepath)
    
    model_AIC = np.load(savepath+str(session_ID)+'_model_AIC.npy')
    best_model = select_best_model(model_AIC)
    
    LP_idx = np.argwhere(best_model==0)[:,0]
    HP_idx = np.argwhere(best_model==1)[:,0]
    BP_idx = np.argwhere(best_model==2)[:,0]
    
    return LP_idx, HP_idx, BP_idx
      
def select_best_model(model_AIC):
    # 0 is LP, 1 is HP, 2 is BP
    best_model = np.argmin(model_AIC,axis=1)
    non_significant = np.argwhere(np.isnan(model_AIC))[:,0]
    best_model[non_significant] = -1
    return best_model

def model_counts(best_model):
    num_LP = (best_model==0).sum()
    num_HP = (best_model==1).sum()
    num_BP = (best_model==2).sum()
    num_NS = (best_model==-1).sum()
    total = num_LP + num_HP + num_BP + num_NS
    return num_LP, num_HP, num_BP, num_NS, total
   
def LP_HP_model_selection(session_ID,savepath,do_plot=False):
    
    if os.path.isfile(savepath+str(session_ID)+'_model_AIC.npy'):
        model_AIC = np.load(savepath+str(session_ID)+'_model_AIC.npy')
        low_pass_params = np.load(savepath+str(session_ID)+'_LP_params.npy')
        high_pass_params = np.load(savepath+str(session_ID)+'_HP_params.npy')
        band_pass_params = np.load(savepath+str(session_ID)+'_BP_params.npy')
    else:
    
        directions, __ = grating_params()
        
        sweep_table = load_sweep_table(savepath,session_ID)
        mean_sweep_events = load_mean_sweep_events(savepath,session_ID)
       
        (num_sweeps,num_cells) = np.shape(mean_sweep_events)
        
        condition_responses, __ = compute_mean_condition_responses(sweep_table,mean_sweep_events)
        
        p_all = chi_square_all_conditions(sweep_table,mean_sweep_events,session_ID,savepath)
        sig_cells = p_all < SIG_THRESH
        
        peak_dir_idx, __ = get_peak_conditions(condition_responses)
        peak_directions = directions[peak_dir_idx]
        
        high_pass_params = np.zeros((num_cells,3))
        high_pass_params[:] = np.NaN
        low_pass_params = np.zeros((num_cells,3))
        low_pass_params[:] = np.NaN
        band_pass_params = np.zeros((num_cells,4))
        band_pass_params[:] = np.NaN
        model_AIC = np.zeros((num_cells,3))
        model_AIC[:] = np.NaN
        for i_dir,direction in enumerate(directions):
            cells_pref_dir = (peak_directions == direction) & sig_cells
            
            not_60_contrast = sweep_table['Contrast'].values != 0.6
            is_direction = (sweep_table['Ori'] == direction).values
            sweeps_with_dir = np.argwhere(is_direction & not_60_contrast)[:,0]
            sweep_contrasts = 100 * sweep_table['Contrast'][sweeps_with_dir].values
            
            cell_idx = np.argwhere(cells_pref_dir)[:,0]
            for cell in cell_idx:
                
                sweep_responses = mean_sweep_events[sweeps_with_dir,cell]
                
                lp_params, lp_aic = select_over_initial_conditions(sweep_responses,sweep_contrasts,'LP')
                hp_params, hp_aic = select_over_initial_conditions(sweep_responses,sweep_contrasts,'HP')
                bp_params, bp_aic = select_over_initial_conditions(sweep_responses,sweep_contrasts,'BP')

                low_pass_params[cell,:] = lp_params
                model_AIC[cell,0] = lp_aic
                high_pass_params[cell,:] = hp_params
                model_AIC[cell,1] = hp_aic
                band_pass_params[cell,:] = bp_params
                model_AIC[cell,2] = bp_aic
                
                if do_plot:
                    
                    x_sample = np.linspace(np.log(5.0),np.log(80.0))
                    plt.figure()
                    plt.plot(np.log(sweep_contrasts),sweep_responses,'ko')
                    plt.plot(x_sample,high_pass(x_sample,high_pass_params[cell,0],high_pass_params[cell,1],high_pass_params[cell,2]),'r')
                    plt.plot(x_sample,low_pass(x_sample,low_pass_params[cell,0],low_pass_params[cell,1],low_pass_params[cell,2]),'b')
                    plt.plot(x_sample,band_pass(x_sample,band_pass_params[cell,0],band_pass_params[cell,1],band_pass_params[cell,2],band_pass_params[cell,3]),'g')
                    plt.show()
                    
        np.save(savepath+str(session_ID)+'_model_AIC.npy',model_AIC)
        np.save(savepath+str(session_ID)+'_LP_params.npy',low_pass_params)
        np.save(savepath+str(session_ID)+'_HP_params.npy',high_pass_params)
        np.save(savepath+str(session_ID)+'_BP_params.npy',band_pass_params)

    LP_c50 = low_pass_params[:,0]
    HP_c50 = high_pass_params[:,0]
    BP_rise_c50 = band_pass_params[:,0]
    BP_fall_c50 = band_pass_params[:,1]

    return LP_c50, HP_c50, BP_rise_c50, BP_fall_c50, model_AIC  

def select_over_initial_conditions(sweep_responses,sweep_contrasts,model):
    
    min_height = np.mean(sweep_responses)
    max_height = np.max(sweep_responses)
    min_c50 = 8.0
    max_c50 = 50.0
    bandwidth_min = np.log(2.0) #units of log contrast 
    bandwidth_max = np.log(8.0) # 3 doublings
    
    if model=='BP':
        initial_guess = [10.0,np.log(2.0),0.0,(min_height+max_height)/2.0]
        param_bounds = ([min_c50,bandwidth_min,0.0,min_height],[max_c50,bandwidth_max,min_height,max_height])
    
    elif model=='LP':# LP or HP
        initial_guess = [10.0,0.0,(min_height+max_height)/2.0]
        param_bounds = ([min_c50,0.0,min_height],[max_c50,min_height,max_height])
    else:
        initial_guess = [10.0,0.0,(min_height+max_height)/2.0]
        param_bounds = ([min_c50,0.0,min_height],[max_c50,min_height,max_height])
      
    init_c50 = np.exp(np.linspace(np.log(param_bounds[0][0]),np.log(param_bounds[1][0]),5))[1:]
        
    best_AIC = np.inf
    best_params = []
    for i,c50 in enumerate(init_c50):
        
        initial_guess[0] = c50
        
        if model=='LP':
            p, a = fit_LP(sweep_responses,sweep_contrasts,initial_guess,param_bounds)
        elif model=='HP':
            p, a = fit_HP(sweep_responses,sweep_contrasts,initial_guess,param_bounds)
        else:
            p, a = fit_BP(sweep_responses,sweep_contrasts,initial_guess,param_bounds)
            
        if a<best_AIC:
            best_AIC = a
            best_params = p
    
    return best_params, best_AIC

def fit_LP(sweep_responses,sweep_contrasts,initial_guess,param_bounds):
    
    res_robust = least_squares(low_pass_residuals,
                               initial_guess, 
                               bounds=param_bounds, 
                               loss='linear', 
                               f_scale=get_inlier_scale(sweep_responses), 
                               args=(np.log(sweep_contrasts), sweep_responses))
    
    params = res_robust.x
    model_AIC = low_AIC(sweep_contrasts,sweep_responses,params)

    return params, model_AIC

def fit_HP(sweep_responses,sweep_contrasts,initial_guess,param_bounds):
    
    res_robust = least_squares(high_pass_residuals,
                               initial_guess, 
                               bounds=param_bounds, 
                               loss='linear', 
                               f_scale=get_inlier_scale(sweep_responses), 
                               args=(np.log(sweep_contrasts), sweep_responses))
    
    params = res_robust.x
    model_AIC = high_AIC(sweep_contrasts,sweep_responses,params)

    return params, model_AIC

def fit_BP(sweep_responses,sweep_contrasts,initial_guess,param_bounds):
    
    res_robust = least_squares(band_pass_residuals,
                               initial_guess, 
                               bounds=param_bounds, 
                               loss='linear', 
                               f_scale=get_inlier_scale(sweep_responses), 
                               args=(np.log(sweep_contrasts), sweep_responses))
    
    params = res_robust.x
    model_AIC = band_AIC(sweep_contrasts,sweep_responses,params)

    return params, model_AIC

def get_inlier_scale(sweep_responses,min_scale=0.005):
    robust_std = np.std(sweep_responses[int(len(sweep_responses)/5):int(4*len(sweep_responses)/5)])
    return max(min_scale,robust_std)

def high_AIC(sweep_contrasts,obs_resp,params):
    contrasts = [5,10,20,40,80]
    pred_resp = high_pass(np.log(contrasts),params[0],params[1],params[2])
    log_likelihood = bootstrap_likelihood(sweep_contrasts,obs_resp,pred_resp)
    return AIC(log_likelihood,3)

def low_AIC(sweep_contrasts,obs_resp,params):
    contrasts = [5,10,20,40,80]
    pred_resp = low_pass(np.log(contrasts),params[0],params[1],params[2])
    log_likelihood = bootstrap_likelihood(sweep_contrasts,obs_resp,pred_resp)
    return AIC(log_likelihood,3)

def band_AIC(sweep_contrasts,obs_resp,params):
    contrasts = [5,10,20,40,80]
    pred_resp = band_pass(np.log(contrasts),params[0],params[1],params[2],params[3])
    log_likelihood = bootstrap_likelihood(sweep_contrasts,obs_resp,pred_resp)
    return AIC(log_likelihood,4)  
  
def AIC(log_likelihood,num_params):
    return 2.0*(num_params-log_likelihood)
     
def bootstrap_likelihood(sweep_contrasts,obs_resp,pred_resp):
    
    contrasts = [5,10,20,40,80]
    
    log_likelihood = 0.0
    #sum over contrasts
    for i_con,contrast in enumerate(contrasts):
        sweeps_at_contrast = sweep_contrasts == contrast
        contrast_resp = obs_resp[sweeps_at_contrast]
    
        p_val = bootstrap_chi(pred_resp[i_con],contrast_resp,obs_resp)
    
        log_likelihood += np.log(p_val)
        
    return log_likelihood

def bootstrap_chi(prediction,sample,population,num_shuffles=1000):
    
    sample_size = len(sample)
    population_size = len(population)
    shuffled_samples = population[np.random.choice(population_size,
                                                       size=(sample_size,num_shuffles))]
    
    pop_var = np.var(population)
    
    sample_chi = np.sum((sample-prediction)**2)/pop_var
    shuffle_chi = np.sum((shuffled_samples-prediction)**2,axis=0)/pop_var
    
    p_val = np.mean(shuffle_chi>sample_chi)
    
    if p_val==0.0:
        p_val = 1.0/num_shuffles
    elif p_val==1.0:
        p_val = 1.0-1.0/num_shuffles
    
    return p_val
        
def sigmoid(x,bias,slope=10.0):
    return np.exp(slope*(x-bias))/(1.0+np.exp(slope*(x-bias))) 

def high_pass(log_contrasts,c50,base,height):
    return base+height*sigmoid(log_contrasts,np.log(c50))

def low_pass(log_contrasts,c50,base,height):
    return base+height*(1.0-sigmoid(log_contrasts,np.log(c50)))

def band_pass(log_contrasts,rise_c50,bandwidth,base,height):
    return base+height*sigmoid(log_contrasts,np.log(rise_c50)) * (1.0-sigmoid(log_contrasts,np.log(rise_c50)+bandwidth))
    
def high_pass_residuals(params,log_contrasts,responses):
    c50 = params[0]
    base = params[1]
    height = params[2]
    predictions = base+height*sigmoid(log_contrasts,np.log(c50))
    return predictions - responses 

def low_pass_residuals(params,log_contrasts,responses):
    c50 = params[0]
    base = params[1]
    height = params[2]
    predictions = base+height*(1.0-sigmoid(log_contrasts,np.log(c50)))
    return predictions - responses 

def band_pass_residuals(params,log_contrasts,responses):
    rise_c50 = params[0]
    bandwidth = params[1]
    base = params[2]
    height = params[3]
    predictions = base+height*sigmoid(log_contrasts,np.log(rise_c50)) * (1.0-sigmoid(log_contrasts,np.log(rise_c50)+bandwidth))
    return predictions - responses 
