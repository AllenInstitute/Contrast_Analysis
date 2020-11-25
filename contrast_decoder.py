#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 21:50:28 2020

@author: danielm
"""
import os
import numpy as np
from matplotlib import pyplot as plt

from sklearn.svm import SVC
from sklearn import model_selection

from contrast_utils import shorthand, grating_params, dataset_params, get_sessions, get_cre_colors, load_sweep_table, load_mean_sweep_running

def decode_direction_from_running(df,savepath,save_format='svg'):
    
    directions,contrasts = grating_params()
    
    running_dict = {}
    
    areas, cres = dataset_params()
    for area in ['VISp']:
        for cre in cres:
            
            celltype = shorthand(area)+ ' ' + shorthand(cre)
            
            session_IDs = get_sessions(df,area,cre)
            num_sessions = len(session_IDs)
            
            if num_sessions > 0:
                
                savename = shorthand(area)+'_'+shorthand(cre)+'_running_direction_decoder.npy'
                if os.path.isfile(savepath+savename):
                    #decoder_performance = np.load(savepath+savename)
                    running_performance = np.load(savepath+shorthand(area)+'_'+shorthand(cre)+'_running_direction_decoder.npy')
                else:
                    #decoder_performance = []
                    running_performance = []
                    for i_session,session_ID in enumerate(session_IDs):
                        
                        #mean_sweep_events = load_mean_sweep_events(savepath,session_ID)
                        mean_sweep_running = load_mean_sweep_running(session_ID,savepath)
                        
                        sweep_table = load_sweep_table(savepath,session_ID)
                        
                        #(num_sweeps,num_cells) =  np.shape(mean_sweep_events)
                        
                        is_blank = sweep_table['Ori'].isnull().values
                        blank_sweeps = np.argwhere(is_blank)[:,0]
                        sweep_directions = sweep_table['Ori'].values
                        
                        sweep_categories = sweep_directions.copy()
                        sweep_categories[blank_sweeps] = 360
                        sweep_categories = sweep_categories.astype(np.int) / 45
                        
                        is_low = sweep_table['Contrast'].values < 0.2
                        sweeps_included = np.argwhere(is_low)[:,0]
                        
                        sweep_categories = sweep_categories[sweeps_included]
                        #mean_sweep_events = mean_sweep_events[sweeps_included]
                        mean_sweep_running = mean_sweep_running[sweeps_included]
                        
                        #decode front-to-back motion
#                        is_front_to_back = (sweep_categories==0) |  (sweep_categories==7)
#                        front_to_back_sweeps = np.argwhere(is_front_to_back)[:,0]
#                        rest_sweeps = np.argwhere(~is_front_to_back)[:,0]
#                        sweep_categories[front_to_back_sweeps] = 0
#                        sweep_categories[rest_sweeps] = 1
                        
                        running_performance.append(decode_direction(mean_sweep_running.reshape(len(sweeps_included),1),sweep_categories))
                        #for nc in range(num_cells):
                        #decoder_performance.append(decode_direction(mean_sweep_events,sweep_categories))
                    #decoder_performance = np.array(decoder_performance)
                    running_performance = np.array(running_performance)
                    #np.save(savepath+savename,decoder_performance)
                    np.save(savepath+shorthand(area)+'_'+shorthand(cre)+'_running_direction_decoder.npy',running_performance)
                #print celltype + ': ' + str(np.mean(decoder_performance))  
                print(celltype + ': ' + str(np.mean(running_performance)))
                running_dict[shorthand(cre)] = running_performance 
         
    cre_colors = get_cre_colors()
    
    plt.figure(figsize=(6,4))
    ax = plt.subplot(111)
    ax.plot([-1,6],[12.5,12.5],'k--')
    label_loc = []
    labels = []
    for i,cre in enumerate(cres):
        session_performance = running_dict[shorthand(cre)]
        ax.plot(i*np.ones((len(session_performance),)),100.0*session_performance,'.',markersize=4.0,color=cre_colors[cre])
        ax.plot([i-0.4,i+0.4],[100.0*session_performance.mean(),100.0*session_performance.mean()],color=cre_colors[cre],linewidth=3)
        label_loc.append(i)
        labels.append(shorthand(cre))
    ax.set_xticks(label_loc)
    ax.set_xticklabels(labels,rotation=45,fontsize=10)
    ax.set_ylim(0,25)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim(-1,14)
    #ax.text(3,20,'Predict direction from running',fontsize=14,horizontalalignment='center')
    ax.set_ylabel('Decoding performance (%)',fontsize=14)
    
    if save_format=='svg':
        plt.savefig(savepath+'running_decoder.svg',format='svg')
    else:
        plt.savefig(savepath+'running_decoder.png',dpi=300)
        
    plt.close()
    
def decode_direction(X,y,num_subsamples=20):
    
    classifier = SVC()#KNN()
    
    test_performance = np.array([])
    for subsample in range(num_subsamples):
        X_balanced, y_balanced = balance_categories(X,y)
        scores = model_selection.cross_validate(classifier,
                                                X_balanced,
                                                y=y_balanced)
        
        test_performance = np.append(test_performance,np.mean(scores['test_score']))
    
    return test_performance.mean()

def balance_categories(X,y):
    
    categories = np.unique(y)
    
    category_count = []
    for category in categories:
        category_count.append(np.sum(y==category))
    min_sweeps = np.array(category_count).min()
        
    balanced_sweeps = np.array([])
    for category in categories:
        category_idx = np.argwhere(y==category)[:,0]
        sample = np.random.permutation(category_idx)[:min_sweeps]
        balanced_sweeps = np.append(balanced_sweeps,sample)
    balanced_sweeps = balanced_sweeps.astype(np.int)
    
    return X[balanced_sweeps], y[balanced_sweeps]