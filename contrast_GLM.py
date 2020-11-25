#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 19:43:27 2020

@author: dan
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from contrast_utils import get_cre_colors, center_direction_zero, shorthand, get_sessions, grating_params, load_mean_sweep_events, load_mean_sweep_running, load_sweep_table 
from contrast_metrics import chi_square_all_conditions
from contrast_running import plot_pooled_mat

def model_GLM(df,savepath):
    
    for area in ['VISp']:
        for cre in ['Vip-IRES-Cre','Sst-IRES-Cre','Cux2-CreERT2']:
            
            session_IDs = get_sessions(df,area,cre)
            savename = shorthand(area)+'_'+shorthand(cre)
            if len(session_IDs) > 0:
                
                lamb = K_session_cross_validation(session_IDs,savename,savepath)
                
                print(shorthand(cre)+ ' lambda used: ' + str(lamb))
                
                X, y = construct_pooled_Xy(session_IDs,savepath)

                glm = sm.GLM(y,X,family=sm.families.Poisson())
                
                res = glm.fit_regularized(method='elastic_net',
                                          alpha=lamb,
                                          maxiter=200,
                                          L1_wt=1.0, # 1.0:all L1, 0.0: all L2
                                          refit=True
                                          )
                
                model_params = np.array(res.params)
                
                y_hat = np.exp(np.sum(model_params.reshape(1,len(model_params))*X,axis=1))
                plot_y(X,y_hat,area,cre,'predicted',savepath)
                
                plot_param_CI(res,area,cre,savepath)
                
                plot_param_heatmaps(model_params,cre,shorthand(area)+ '_' + shorthand(cre),savepath)
        
def plot_y(X,y,area,cre,savename,savepath):

    (num_conditions, num_params) = np.shape(X)
    
    directions,contrasts = grating_params()
    
    stat_resp, run_resp, stat_blank_resp, run_blank_resp = extract_tuning_curves(y)

    stat_resp = center_direction_zero(stat_resp.reshape(1,len(directions),len(contrasts)))[0]
    run_resp = center_direction_zero(run_resp.reshape(1,len(directions),len(contrasts)))[0]

    plot_pooled_mat(run_resp-run_blank_resp,area,cre,'GLM_run_'+savename,savepath)
    plot_pooled_mat(stat_resp-stat_blank_resp,area,cre,'GLM_stat_'+savename,savepath)
    
def extract_tuning_curves(y):

    condition_rows, blank_rows = get_condition_rows()
    
    stat_tuning = y[condition_rows[0]]
    run_tuning = y[condition_rows[1]]
    stat_blank = y[blank_rows[0]]
    run_blank = y[blank_rows[1]]
    
    return stat_tuning, run_tuning, stat_blank, run_blank

def get_condition_rows():
    
    directions,contrasts = grating_params()
    
    condition_rows = np.zeros((2,len(directions),len(contrasts)),dtype=np.int)
    blank_rows = np.zeros((2,),dtype=np.int)
    i_condition = 0
    for run_state in [0,1]:
        
        #blank sweeps   
        blank_rows[run_state] = i_condition
        i_condition+=1       
        
        for i_dir,direction in enumerate(directions):
            for i_con,contrast in enumerate(contrasts):
                condition_rows[run_state,i_dir,i_con] = i_condition
                i_condition+=1
        
    return condition_rows, blank_rows

def plot_param_CI(res,area,cre,savepath,PARAM_TOL=1E-2,save_format='svg'):
    
    model_params = np.array(res.params)
    terms = unpack_params(model_params)
    
    CI = np.array(res.conf_int(alpha=0.05))
    CI_lb = unpack_params(CI[:,0])
    CI_ub = unpack_params(CI[:,1])
    
    plot_order = ['blank',
                  'run',
                  'dir',
                  'con',
                  'dirXrun',
                  'conXrun',
                  'dirXcon',
                  'dirXconXrun']
        
    directions,contrasts = grating_params()
    directions = [-135,-90,-45,0,45,90,135,180]
    
    x_labels = {}
    x_labels['blank'] = ['blank']
    x_labels['blankXrun'] = ['blank X run']
    x_labels['run'] = ['run']
    x_labels['dir'] = [str(x) for x in directions]
    x_labels['con'] = [str(int(100*x)) for x in contrasts]
    x_labels['dirXrun'] = [str(x) for x in directions]
    x_labels['conXrun'] = [str(int(100*x)) for x in contrasts]
        
    plt.figure(figsize=(20,4.5))
    
    savename = shorthand(area)+ '_' + shorthand(cre)
    cre_colors = get_cre_colors()
    
    ax = plt.subplot(111)
    
    curr_x = 0
    
    x_ticks = []
    x_ticklabels = []
    for i,param_name in enumerate(plot_order):
    
        param_means = terms[param_name]
        param_CI_lb = CI_lb[param_name]
        param_CI_ub = CI_ub[param_name]
        
        #center directions on zero
        if param_name=='dirXcon' or param_name=='dirXconXrun' or param_name=='dir' or param_name=='dirXrun':
            param_means = center_dir_on_zero(param_means)
            param_CI_lb = center_dir_on_zero(param_CI_lb)
            param_CI_ub = center_dir_on_zero(param_CI_ub)
        
        #handle parameters that are not 1D arrays
        if type(param_means)==np.float64:
            param_means = np.array([param_means])
            param_CI_lb = np.array([param_CI_lb])
            param_CI_ub = np.array([param_CI_ub])
        elif param_name=='dirXcon' or param_name=='dirXconXrun':
            param_means = param_means.flatten()
            param_CI_lb = param_CI_lb.flatten()
            param_CI_ub = param_CI_ub.flatten()
            
        param_errs = CI_to_errorbars(param_means,
                                     param_CI_lb,
                                     param_CI_ub)
        
        num_params = np.shape(param_errs)[1]
        
        
        #for dirXcon terms, only plot non-zero values
        if param_name=='dirXcon' or param_name=='dirXconXrun':
            non_zero_idx = np.argwhere((param_CI_ub<-PARAM_TOL) | (param_CI_lb>PARAM_TOL))[:,0]
            num_params = len(non_zero_idx)
            cond_tick_labels = []
            if num_params > 0:
                param_means = param_means[non_zero_idx]
                param_errs = param_errs[:,non_zero_idx]
                
                for i_cond,idx in enumerate(non_zero_idx):
                    i_dir = int(idx / 6)
                    i_con = int(idx % 6)
                    cond_tick_labels.append(str(int(100*contrasts[i_con])) + '%,' + str(int(directions[i_dir])))
                
            # pad params to make all plots equal size
            ticks_to_plot = 6
            num_to_pad = ticks_to_plot - num_params
            for i_pad in range(num_to_pad):
                cond_tick_labels.append('')
                
            x_labels[param_name] = cond_tick_labels
            x_values = np.arange(curr_x,curr_x+2*num_params,2) #double spacing
        else:
            ticks_to_plot = num_params
            x_values = np.arange(curr_x,curr_x+num_params)
                
        if num_params > 0:
            
            ax.errorbar(x_values,
                        param_means,
                        yerr=param_errs,
                        fmt='o',
                        color=cre_colors[cre],
                        linewidth=3,
                        capsize=5, 
                        elinewidth=2,
                        markeredgewidth=2)
            
            for i_x,x in enumerate(x_values):
                x_ticks.append(x)
                x_ticklabels.append(x_labels[param_name][i_x])
        
        curr_x += ticks_to_plot + 1 
    
    ax.plot([-1,curr_x],[0,0],'k',linewidth=1.0)
    
    ax.set_ylabel('Weight',fontsize=14)

    ax.set_xlim([-1,curr_x])
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels)
    
    y_max = 2.5
    y_min = -1.5
    y_ticks = [-1,0,1,2]
    
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(int(y)) for y in y_ticks])
    ax.set_ylim([y_min,y_max])
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    if save_format == 'svg':
        plt.savefig(savepath+savename+'_param_CI.svg',format='svg')
    else:
        plt.savefig(savepath+savename+'_param_CI.png',dpi=300)
        
    plt.close()
    
def center_dir_on_zero(orig):
    new = orig.copy()
    new[:3] = orig[5:]
    new[3:] = orig[:5]
    return new
    
def CI_to_errorbars(means,CI_lb,CI_ub):
    num_vars = len(means)
    errs = np.zeros((2,num_vars))
    errs[0] = means - CI_lb
    errs[1] = CI_ub - means
    return errs

def plot_param_heatmaps(model_params,cre,savename,savepath,save_format='svg'):

    terms = unpack_params(model_params)

    w_max = 2.0
    
    directions,contrasts = grating_params()
    num_dir = len(directions)
    num_con = len(contrasts)
    
    title_fontsize = 10
    tick_fontsize = 10
    
    dir_ticks = np.arange(num_dir)
    dir_ticklabels = ['','-90','','0','','90','','180']
    
    plt.figure(figsize=(20,6))
    ax1 = plt.subplot(194)
    ax1.imshow(np.tile(center_dir_on_zero(terms['dir']),reps=(num_con,1)),interpolation='none',origin='lower',cmap='RdBu_r',vmin=-w_max,vmax=w_max)
    ax1.set_xticks(dir_ticks)
    ax1.set_xticklabels(dir_ticklabels,fontsize=tick_fontsize)
    ax1.set_yticks([])
    ax1.set_title('Direction',fontsize=title_fontsize)
    
    ax2 = plt.subplot(195)
    ax2.imshow(np.tile(terms['con'],reps=(num_dir,1)).T,interpolation='none',origin='lower',cmap='RdBu_r',vmin=-w_max,vmax=w_max)
    ax2.set_yticks(np.arange(num_con))
    ax2.set_yticklabels([str(int(100*x)) for x in contrasts],fontsize=tick_fontsize)
    ax2.set_xticks([])
    ax2.set_title('Contrast',fontsize=title_fontsize)
    
    ax3 = plt.subplot(198)
    ax3.imshow(center_dir_on_zero(terms['dirXcon']).T,interpolation='none',cmap='RdBu_r',origin='lower',vmin=-w_max,vmax=w_max)
    ax3.set_xticks(dir_ticks)
    ax3.set_xticklabels(dir_ticklabels,fontsize=tick_fontsize)
    ax3.set_yticks(np.arange(num_con))
    ax3.set_yticklabels([str(int(100*x)) for x in contrasts],fontsize=tick_fontsize)
    ax3.set_title('Direction X Contrast',fontsize=title_fontsize)
    
    ax4 = plt.subplot(197)
    ax4.imshow(np.tile(terms['conXrun'],reps=(num_dir,1)).T,interpolation='none',origin='lower',cmap='RdBu_r',vmin=-w_max,vmax=w_max)
    ax4.set_xticks([])
    ax4.set_yticks(np.arange(num_con))
    ax4.set_yticklabels([str(int(100*x)) for x in contrasts],fontsize=tick_fontsize)
    ax4.set_title('Run X Contrast',fontsize=title_fontsize)
    
    ax5 = plt.subplot(196)
    ax5.imshow(np.tile(center_dir_on_zero(terms['dirXrun']),reps=(num_con,1)),interpolation='none',origin='lower',cmap='RdBu_r',vmin=-w_max,vmax=w_max)
    ax5.set_xticks(dir_ticks)
    ax5.set_xticklabels(dir_ticklabels,fontsize=tick_fontsize)
    ax5.set_title('Run X Direction',fontsize=title_fontsize)
    ax5.set_yticks([])
    
    ax6 = plt.subplot(193)
    ax6.imshow(np.tile(terms['run'],reps=(num_con,num_dir)),interpolation='none',origin='lower',cmap='RdBu_r',vmin=-w_max,vmax=w_max)
    ax6.set_xticks([])
    ax6.set_title('Run',fontsize=title_fontsize)
    ax6.set_yticks([])
    
    ax7 = plt.subplot(199)
    ax7.imshow(center_dir_on_zero(terms['dirXconXrun']).T,interpolation='none',cmap='RdBu_r',origin='lower',vmin=-w_max,vmax=w_max)
    ax7.set_xticks(dir_ticks)
    ax7.set_xticklabels(dir_ticklabels,fontsize=tick_fontsize)
    ax7.set_yticks(np.arange(num_con))
    ax7.set_yticklabels([str(int(100*x)) for x in contrasts],fontsize=tick_fontsize)
    ax7.set_title('Run X Direction X Contrast',fontsize=title_fontsize)
    
    ax8 = plt.subplot(192)
    blanks = np.tile(terms['blank'],reps=(num_con,num_dir))
    ax8.imshow(blanks,interpolation='none',origin='lower',cmap='RdBu_r',vmin=-w_max,vmax=w_max)
    ax8.set_xticks([])
    ax8.set_title('Blank',fontsize=title_fontsize)
    ax8.set_yticks([])
    
    ax9 = plt.subplot(191)
    im = ax9.imshow(np.tile(terms['const'],reps=(num_con,num_dir)),interpolation='none',origin='lower',cmap='RdBu_r',vmin=-w_max,vmax=w_max)
    ax9.set_xticks([])
    ax9.set_title('Constant',fontsize=title_fontsize)
    ax9.set_yticks([])
    
    cbar = plt.colorbar(im,ax=ax9,ticks=[-2,-1,0,1,2])
          
    if save_format == 'svg':
        plt.savefig(savepath+savename+'_GLM.svg',format='svg')
    else:
        plt.savefig(savepath+savename+'_GLM.png',dpi=300)
    
    plt.close()     
           
def unpack_params(model_params):

    directions,contrasts = grating_params()
    
    curr = 0
    
    terms = {}
    
    terms['blank'] = model_params[curr]
    curr+=1
    
    # terms without run interaction
    dir_terms = []
    for i_dir,this_direction in enumerate(directions):
        dir_terms.append(model_params[curr])
        curr+=1
    terms['dir'] = np.array(dir_terms)
    
    con_terms = []
    for i_con,this_contrast in enumerate(contrasts):
        con_terms.append(model_params[curr])
        curr+=1
    terms['con'] = np.array(con_terms)
        
    dirXcon_terms = []
    for i_dir,this_direction in enumerate(directions):
        for i_con,this_contrast in enumerate(contrasts):
            dirXcon_terms.append(model_params[curr])
            curr+=1
    terms['dirXcon'] = np.array(dirXcon_terms).reshape(len(directions),len(contrasts))
    
    terms['blankXrun'] = model_params[curr]
    curr+=1
    
    dirXrun_terms = []
    for i_dir,this_direction in enumerate(directions):
        dirXrun_terms.append(model_params[curr])
        curr+=1
    terms['dirXrun'] = np.array(dirXrun_terms)
    
    conXrun_terms = []
    for i_con,this_contrast in enumerate(contrasts):
        conXrun_terms.append(model_params[curr])
        curr+=1
    terms['conXrun'] = np.array(conXrun_terms)
    
    dirXconXrun_terms = []
    for i_dir,this_direction in enumerate(directions):
        for i_con,this_contrast in enumerate(contrasts):
            dirXconXrun_terms.append(model_params[curr])
            curr+=1
    terms['dirXconXrun'] = np.array(dirXconXrun_terms).reshape(len(directions),len(contrasts))
 
    terms['run'] = model_params[curr]
    curr+=1
    
    terms['const'] = model_params[curr]
    
    return terms

def K_session_cross_validation(session_IDs,
                               savename,
                               savepath,
                               LAMBDA_SAMPLES=20):
    
    #cross-validation is to determine which parameters make a significant
    #     contribution to model performance.
    #use L1 penalty and find lambda through cross-validation
    #then the non-zero coefficients are part of the model. Train the final model
    # on all of the data and record the model coefficients.
    
    # GLM: exp( C + blank + run + dir + con + dirXcon + dirXrun + conXrun + dirXconXrun )
    
    #first pass with wide range of lambdas
    if os.path.isfile(savepath+savename+'_validation_performance.npy'):
        validation_performance = np.load(savepath+savename+'_validation_performance.npy')
        lambdas = np.load(savepath+savename+'_validation_lambdas.npy')
    else:
        lambdas = 10.0 ** np.linspace(-6.0,6.0,num=LAMBDA_SAMPLES)
        validation_performance = Kfold_CV(session_IDs,lambdas,savepath)
        np.save(savepath+savename+'_validation_lambdas.npy',lambdas)
        np.save(savepath+savename+'_validation_performance.npy',validation_performance)
    
    Kfold_performance = validation_performance.mean(axis=1)      
    
    # do a second pass near the minimum:
    if os.path.isfile(savepath+savename+'_validation_performance_fine.npy'):
        validation_performance_fine = np.load(savepath+savename+'_validation_performance_fine.npy')
        lambdas_fine = np.load(savepath+savename+'_validation_lambdas_fine.npy')
    else:
        
        lambda_min_idx = np.argmin(Kfold_performance)
        lambda_lb = lambdas[lambda_min_idx-1]
        lambda_ub = lambdas[lambda_min_idx+2]
        
        lambdas_fine = np.linspace(lambda_lb,lambda_ub,num=20)
        validation_performance_fine = Kfold_CV(session_IDs,lambdas_fine,savepath)
        np.save(savepath+savename+'_validation_lambdas_fine.npy',lambdas_fine)
        np.save(savepath+savename+'_validation_performance_fine.npy',validation_performance_fine)
    
    # combine coarse and fine runs
    lambdas = np.append(lambdas,lambdas_fine)
    validation_performance = np.append(validation_performance,
                                       validation_performance_fine,
                                       axis=0)
    sort_idx = np.argsort(lambdas)
    lambdas = lambdas[sort_idx]
    validation_performance = validation_performance[sort_idx]
    
    #plot results
    Kfold_performance = validation_performance.mean(axis=1) 
    
    plt.figure(figsize=(8,8))
    ax = plt.subplot(111)
    ax.plot(np.log(lambdas),Kfold_performance,'k',linewidth=2.0)
    ax.set_xticks(np.log(lambdas))
    ax.set_xticklabels([str(x) for x in lambdas])
    plt.savefig(savepath+savename+'_validation_performance.png',dpi=300)
    plt.close()
    
    return lambdas[np.argmin(Kfold_performance)]
    
def Kfold_CV(session_IDs,lambdas,savepath):
    
    #K-fold CV: train on K-1 sessions, test on held-out session.
    num_folds = len(session_IDs)
    validation_performance = np.zeros((len(lambdas),num_folds))
    for i_fold,test_session_ID in enumerate(session_IDs):
        
        test_sessions = [test_session_ID]
        (X_test,y_test) = construct_pooled_Xy(test_sessions,savepath)
        
        train_sessions = np.setxor1d(session_IDs,test_sessions)
        (X_train,y_train) = construct_pooled_Xy(train_sessions,savepath)
        
        for i_lambda, lamb in enumerate(lambdas):
            
            glm = sm.GLM(y_train,X_train,family=sm.families.Poisson())
                
            res = glm.fit_regularized(method='elastic_net',
                                      alpha=lamb,
                                      maxiter=200,
                                      L1_wt=1.0 # 1.0:all L1, 0.0: all L2
                                      )
            
            y_predicted = glm.predict(params=res.params,exog=X_test)
            
            validation_performance[i_lambda,i_fold] = np.mean((y_test-y_predicted)**2)
    
            print('lambda: ' + str(lamb) + ' perf: ' + str(validation_performance[i_lambda,i_fold]))     
    
    return validation_performance
    
def construct_pooled_Xy(session_IDs,
                        savepath,
                        RUN_THRESH=1.0):
    
    y = None
    for session_ID in session_IDs:
        
        sweep_table = load_sweep_table(savepath,session_ID)
        mean_sweep_events = load_mean_sweep_events(savepath,session_ID)
        
        pvals = chi_square_all_conditions(sweep_table,mean_sweep_events,session_ID,savepath)
        sig_cells = pvals < 0.01
        mean_sweep_events = mean_sweep_events[:,sig_cells]   
        
        (num_sweeps,session_cells) = np.shape(mean_sweep_events)
        
        mean_sweep_running = load_mean_sweep_running(session_ID,savepath)         
        is_run = mean_sweep_running >= RUN_THRESH
        
        X, session_y = construct_session_Xy(sweep_table,is_run,mean_sweep_events)
        
        if y is not None:
            y = np.append(y,session_y,axis=1)      
        else:
            y = session_y
    
    y = np.nanmean(y,axis=1)
    
    conditions_sampled = np.argwhere(np.isfinite(y))[:,0]
    
    y *= 3000
    
    return X[conditions_sampled], y[conditions_sampled]

def concat_neurons(X,y):
    
    (num_conditions,num_neurons) = np.shape(y) 
    
    X_concat = None
    y_concat = None
    for i_neuron in range(num_neurons):
        conditions_sampled = np.argwhere(np.isfinite(y[:,i_neuron]))[:,0]
        
        if i_neuron==0:
            X_concat = X[conditions_sampled]
            y_concat = y[conditions_sampled,0]
        else:
            X_concat = np.append(X_concat,X[conditions_sampled],axis=0)
            y_concat = np.append(y_concat,y[conditions_sampled,i_neuron],axis=0)
        
    return X_concat, y_concat

def construct_session_Xy(sweep_table,
                         is_run,
                         mean_sweep_events):
    
    directions,contrasts = grating_params()

    num_dir = len(directions)
    num_con = len(contrasts)
    (num_sweeps,num_cells) = np.shape(mean_sweep_events)
    
    num_vars = 2 * (1 + num_dir + num_con + num_dir*num_con) + 2
    num_conditions = 2 * (1 + num_dir * num_con)
    
    X = np.zeros((num_conditions,num_vars),dtype=np.bool) #[0-7 dir,
                                                          # 8-13 con,
                                                          # 14-61 dirXcon,
                                                          # 62 blank,
                                                          # 63-70 dirXrun,
                                                          # 71-76 conXrun,
                                                          # 77-124 dirXconXrun,
                                                          # 125 blankXrun,
                                                          # 126 run,
                                                          # 127 const]
    y = np.zeros((num_conditions,num_cells)) 

    i_condition = 0
    for run_state in [False,True]:
        
        blank_resp = condition_y_separate(None,None,run_state,sweep_table,is_run,mean_sweep_events)
        X[i_condition] = condition_vars(None,None,run_state)
        y[i_condition] = blank_resp
        i_condition+=1
        
        for i_dir,direction in enumerate(directions):
            for i_con,contrast in enumerate(contrasts):
                this_resp = condition_y_separate(direction,contrast,run_state,sweep_table,is_run,mean_sweep_events)
                
                # only include cells that we have a reliable measure of blank response
                #this_resp = np.where(np.isfinite(blank_resp),this_resp,np.NaN)
                
                X[i_condition] = condition_vars(direction,contrast,run_state)
                y[i_condition] = this_resp
                i_condition+=1
        
    return X.astype(np.float), y

def condition_vars(direction,
                   contrast,
                   run_state):
    
    directions,contrasts = grating_params()
    
    var_bool = []
    
    # terms without run interaction
    var_bool.append(contrast is None)#blank 
    for i_dir,this_direction in enumerate(directions):
        var_bool.append(direction==this_direction)     
    for i_con,this_contrast in enumerate(contrasts):
        var_bool.append(contrast==this_contrast)     
    for i_dir,this_direction in enumerate(directions):
        for i_con,this_contrast in enumerate(contrasts):
            var_bool.append(direction==this_direction and contrast==this_contrast)
    
    # terms with run interaction
    var_bool.append(contrast is None and run_state)#blankXrun
    for i_dir,this_direction in enumerate(directions):
        var_bool.append(direction==this_direction and run_state)     
    for i_con,this_contrast in enumerate(contrasts):
        var_bool.append(contrast==this_contrast and run_state)     
    for i_dir,this_direction in enumerate(directions):
        for i_con,this_contrast in enumerate(contrasts):
            var_bool.append(direction==this_direction and contrast==this_contrast and run_state)
        
    #run var
    var_bool.append(run_state)
    
    #const var
    var_bool.append(True)
    
    return np.array(var_bool)

def condition_y_separate(direction,
                        contrast,
                        run_state,
                        sweep_table,
                        is_run,
                        mean_sweep_events,
                        MIN_SWEEPS=4):
    
    if contrast is None:
        is_blank = sweep_table['Contrast'].isnull().values
        run_sweeps = is_blank & is_run
        stat_sweeps = is_blank & ~is_run
    else:
        is_direction = sweep_table['Ori'].values == direction
        is_contrast = sweep_table['Contrast'].values == contrast
        run_sweeps = is_direction & is_contrast & is_run
        stat_sweeps = is_direction & is_contrast & ~is_run
        
    if run_sweeps.sum()>=MIN_SWEEPS and run_state:
        return mean_sweep_events[run_sweeps].mean(axis=0) 
    elif stat_sweeps.sum()>=MIN_SWEEPS and not run_state:
        return mean_sweep_events[stat_sweeps].mean(axis=0)
    else:     
        (num_sweeps,num_cells) = np.shape(mean_sweep_events)
        all_NaNs = np.zeros((num_cells,))
        all_NaNs[:] = np.NaN
        return all_NaNs

def condition_y(direction,
                contrast,
                run_state,
                sweep_table,
                is_run,
                mean_sweep_events,
                MIN_SWEEPS=4):
    
    if contrast is None:
        is_blank = sweep_table['Contrast'].isnull().values
        run_sweeps = is_blank & is_run
        stat_sweeps = is_blank & ~is_run
    else:
        is_direction = sweep_table['Ori'].values == direction
        is_contrast = sweep_table['Contrast'].values == contrast
        run_sweeps = is_direction & is_contrast & is_run
        stat_sweeps = is_direction & is_contrast & ~is_run
        
    if run_sweeps.sum()>=MIN_SWEEPS and stat_sweeps.sum()>=MIN_SWEEPS:
        if run_state:
            return mean_sweep_events[run_sweeps].mean(axis=0) 
        else:
            return mean_sweep_events[stat_sweeps].mean(axis=0)
          
    (num_sweeps,num_cells) = np.shape(mean_sweep_events)
    all_NaNs = np.zeros((num_cells,))
    all_NaNs[:] = np.NaN
    return all_NaNs