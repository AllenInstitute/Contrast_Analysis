#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 16:27:05 2019

@author: dan
"""
import os
import numpy as np
import matplotlib.pyplot as plt

def white_board():
    
    savepath = '/Users/danielm/Desktop/SSN/'
    save_format = 'svg'
    
    run_main_condition(savepath,save_format='png')
    
    W = get_W()
    net = init_net()
    VIP_SST_scale(W,net,savepath)
    PV_SST_ratio(W,net,savepath)

def run_main_condition(savepath,save_format='svg'):
    
    W = get_W()
    
    net = init_net()
    stim_oris = [0,30,60,90,-60,-30]

    Max_input = 80.0
    steps_per_input_strength = 10
    N_input_steps = int(steps_per_input_strength*Max_input)
    input_strengths = np.linspace(0., Max_input,N_input_steps+1)
    
    rates,currents = single_run(net,
                            W,
                            input_strengths,
                            stimulus_oris=stim_oris)
    
    plot_connection_distributions(net,W,savepath,save_format=save_format)
    
    for i_stim,stim_ori in enumerate(stim_oris):
        plot_rates(rates[i_stim],
                    input_strengths,
                    'pyr_extra_0deg_at_'+str(int(stim_ori)),
                    savepath,
                    save_format=save_format)
        plot_gain(rates[i_stim],
                    input_strengths,
                    'pyr_extra_0deg_at_'+str(int(stim_ori)),
                    savepath,
                    save_format=save_format)
        plot_E_currents(currents[i_stim],
                        input_strengths,
                        'pyr_extra_0deg_at_'+str(int(stim_ori)),
                        savepath,
                        save_format=save_format)
        plot_fraction_currents(currents[i_stim],
                            input_strengths,
                            'pyr_extra_0deg_at_'+str(int(stim_ori)),
                            savepath,
                            save_format=save_format)

    plot_summed_tuning(rates,
                        input_strengths,
                        savepath,
                        directions_to_sample=stim_oris,
                        save_format=save_format)

def VIP_SST_scale(W,net,savepath,steps_per_input=10):
    
    Max_input = 80.0
    N_input_steps= int(steps_per_input*Max_input)
    input_strengths = np.linspace(0., Max_input, N_input_steps+1)
    
    VIP_SST_scale = np.round(np.linspace(0.0,1.5,num=16),decimals=3)
    
    if os.path.isfile(savepath+'VIP_SST_scale_pyr_rates.npy'):
        pyr_rates_mat = np.load(savepath+'VIP_SST_scale_pyr_rates.npy')
        pv_rates_mat = np.load(savepath+'VIP_SST_scale_pv_rates.npy')
        sst_rates_mat = np.load(savepath+'VIP_SST_scale_sst_rates.npy')
        vip_rates_mat = np.load(savepath+'VIP_SST_scale_vip_rates.npy')
        
    else:
    
        pyr_rates_mat = np.zeros((len(input_strengths),len(VIP_SST_scale)))
        pv_rates_mat = np.zeros((len(input_strengths),len(VIP_SST_scale)))
        sst_rates_mat = np.zeros((len(input_strengths),len(VIP_SST_scale)))
        vip_rates_mat = np.zeros((len(input_strengths),len(VIP_SST_scale)))
        
        for i_scale,scale in enumerate(VIP_SST_scale):
            
            print(str(scale))
                    
            this_W = get_W(vip_scale=scale)
        
            rates,currents = single_run(net,
                                        this_W,
                                        input_strengths,
                                        stimulus_oris=[0])
            Pyr_rates, PV_rates, Sst_rates, Vip_rates = extract_rates(rates[0])
            
            pyr_rates_mat[:,i_scale] = Pyr_rates
            pv_rates_mat[:,i_scale] = PV_rates
            sst_rates_mat[:,i_scale] = Sst_rates
            vip_rates_mat[:,i_scale] = Vip_rates
            
        np.save(savepath+'VIP_SST_scale_pyr_rates.npy',pyr_rates_mat)
        np.save(savepath+'VIP_SST_scale_pv_rates.npy',pv_rates_mat)
        np.save(savepath+'VIP_SST_scale_sst_rates.npy',sst_rates_mat)
        np.save(savepath+'VIP_SST_scale_vip_rates.npy',vip_rates_mat)
        
    plot_rates_matrix(pyr_rates_mat,
                                   input_strengths,
                                   VIP_SST_scale,
                                   np.round(np.linspace(0.0,1.0,6),1),
                                   'Pyr',
                                   'Wvip->sst',
                                   '',
                                   'VIP_SST_scale',
                                   savepath,
                                   y_tick_vals = np.arange(0,100,20),
                                   save_format='png',
                                   max_val=30.)
    plot_rates_matrix(pv_rates_mat,
                                   input_strengths,
                                   VIP_SST_scale,
                                   np.round(np.linspace(0.0,1.0,6),1),
                                   'PV',
                                   'Wvip->sst',
                                   '',
                                   'VIP_SST_scale',
                                   savepath,
                                   y_tick_vals = np.arange(0,100,20),
                                   save_format='png',
                                   max_val=30.)
    plot_rates_matrix(sst_rates_mat,
                                   input_strengths,
                                   VIP_SST_scale,
                                   np.round(np.linspace(0.0,1.0,6),1),
                                   'Sst',
                                   'Wvip->sst',
                                   '',
                                   'VIP_SST_scale',
                                   savepath,
                                   y_tick_vals = np.arange(0,100,20),
                                   save_format='png',
                                   max_val=25.)
    plot_rates_matrix(vip_rates_mat,
                                   input_strengths,
                                   VIP_SST_scale,
                                   np.round(np.linspace(0.0,1.0,6),1),
                                   'Vip',
                                   'Wvip->sst',
                                   '',
                                   'VIP_SST_scale',
                                   savepath,
                                   y_tick_vals = np.arange(0,100,20),
                                   save_format='png',
                                   max_val=15.)
                
    gain_diff = pyr_rates_mat - pyr_rates_mat[:,0].reshape(len(input_strengths),1)
    vip_scale_equals_1_idx = np.argwhere(VIP_SST_scale>1.0)[0,0]
    
    plot_gain_matrix(gain_diff[:,:vip_scale_equals_1_idx],
                                input_strengths,
                                VIP_SST_scale[:vip_scale_equals_1_idx],
                                np.round(np.linspace(0.0,1.0,6),1),
                                'Pyr',
                                'Wvip->sst',
                                '',
                                'VIP_SST_scale',
                                savepath,
                                y_tick_vals = np.arange(0,100,20),
                                save_format='svg')

def PV_SST_ratio(W,net,savepath,steps_per_input=100):
    
    Max_input = 80.0
    N_input_steps = int(steps_per_input*Max_input)
    input_strengths = np.linspace(0., Max_input, N_input_steps+1)
    
    PV_SST_ratio = np.round(np.linspace(0.0,1.0,num=21),decimals=3)
    PV_SST_sum = W[0,1] + W[0,2]
    
    if os.path.isfile(savepath+'PV_SST_ratio_pyr_rates.npy'):
        pyr_rates_mat = np.load(savepath+'PV_SST_ratio_pyr_rates.npy')
        pv_rates_mat = np.load(savepath+'PV_SST_ratio_pv_rates.npy')
        sst_rates_mat = np.load(savepath+'PV_SST_ratio_sst_rates.npy')
        vip_rates_mat = np.load(savepath+'PV_SST_ratio_vip_rates.npy')
        pyr_rates_mat_noVIP = np.load(savepath+'PV_SST_ratio_pyr_rates_noVIP.npy')
        
    else:
    
        pyr_rates_mat = np.zeros((len(input_strengths),len(PV_SST_ratio)))
        pv_rates_mat = np.zeros((len(input_strengths),len(PV_SST_ratio)))
        sst_rates_mat = np.zeros((len(input_strengths),len(PV_SST_ratio)))
        vip_rates_mat = np.zeros((len(input_strengths),len(PV_SST_ratio)))
        pyr_rates_mat_noVIP = np.zeros((len(input_strengths),len(PV_SST_ratio)))
        
        for i_ratio,ratio in enumerate(PV_SST_ratio):
            
            print(str(ratio))
                    
            this_W = W.copy()
            this_W[0,1] = ratio * PV_SST_sum
            this_W[0,2] = (1.0 - ratio) * PV_SST_sum
        
            rates,currents = single_run(net,
                                        this_W,
                                        input_strengths,
                                        stimulus_oris=[0])
            Pyr_rates, PV_rates, Sst_rates, Vip_rates = extract_rates(rates[0])
            
            pyr_rates_mat[:,i_ratio] = Pyr_rates
            pv_rates_mat[:,i_ratio] = PV_rates
            sst_rates_mat[:,i_ratio] = Sst_rates
            vip_rates_mat[:,i_ratio] = Vip_rates
            
        np.save(savepath+'PV_SST_ratio_pyr_rates.npy',pyr_rates_mat)
        np.save(savepath+'PV_SST_ratio_pv_rates.npy',pv_rates_mat)
        np.save(savepath+'PV_SST_ratio_sst_rates.npy',sst_rates_mat)
        np.save(savepath+'PV_SST_ratio_vip_rates.npy',vip_rates_mat)
        
        W[2,3] = 0.0
        for i_ratio,ratio in enumerate(PV_SST_ratio):
            
            print(str(ratio))
                    
            this_W = W.copy()
            this_W[0,1] = ratio * PV_SST_sum
            this_W[0,2] = (1.0 - ratio) * PV_SST_sum
        
            rates,currents = single_run(net,
                                        this_W,
                                        input_strengths,
                                        stimulus_oris=[0])
            Pyr_rates, PV_rates, Sst_rates, Vip_rates = extract_rates(rates[0])
            
            pyr_rates_mat_noVIP[:,i_ratio] = Pyr_rates
        
        np.save(savepath+'PV_SST_ratio_pyr_rates_noVIP.npy',pyr_rates_mat_noVIP)
        
    plot_rates_matrix(pyr_rates_mat,
                                   input_strengths,
                                   PV_SST_ratio,
                                   np.round(np.linspace(0.0,1.0,6),1),
                                   'Pyr',
                                   'Wpv/(Wpv+Wsst)',
                                   '',
                                   'PV_SST_ratio',
                                   savepath,
                                   y_tick_vals = np.arange(0,100,20),
                                   save_format='png',
                                   max_val=30.)
    plot_rates_matrix(pv_rates_mat,
                                   input_strengths,
                                   PV_SST_ratio,
                                   np.round(np.linspace(0.0,1.0,6),1),
                                   'PV',
                                   'Wpv/(Wpv+Wsst)',
                                   '',
                                   'PV_SST_ratio',
                                   savepath,
                                   y_tick_vals = np.arange(0,100,20),
                                   save_format='png',
                                   max_val=8.)
    plot_rates_matrix(sst_rates_mat,
                                   input_strengths,
                                   PV_SST_ratio,
                                   np.round(np.linspace(0.0,1.0,6),1),
                                   'Sst',
                                   'Wpv/(Wpv+Wsst)',
                                   '',
                                   'PV_SST_ratio',
                                   savepath,
                                   y_tick_vals = np.arange(0,100,20),
                                   save_format='png',
                                   max_val=200.)
    plot_rates_matrix(vip_rates_mat,
                                   input_strengths,
                                   PV_SST_ratio,
                                   np.round(np.linspace(0.0,1.0,6),1),
                                   'Vip',
                                   'Wpv/(Wpv+Wsst)',
                                   '',
                                   'PV_SST_ratio',
                                   savepath,
                                   y_tick_vals = np.arange(0,100,20),
                                   save_format='png',
                                   max_val=15.)
                
    gain_diff = pyr_rates_mat - pyr_rates_mat_noVIP
    
    plot_gain_matrix(gain_diff[:,:-1],
                                input_strengths,
                                PV_SST_ratio[:-1],
                                np.round(np.linspace(0.0,1.0,6),1),
                                'Pyr',
                                'Wpv/(Wpv+Wsst)',
                                '',
                                'PV_SST_ratio',
                                savepath,
                                y_tick_vals = np.arange(0,100,20),
                                save_format='svg')

def plot_stability_matrix(stability,
                          input_strengths,
                          param_values,
                          x_tick_vals,
                          param_name,
                          title,
                          savename,
                          savepath,
                          y_tick_vals=np.arange(0,110,10),
                          label_fontsize=13,
                          save_format='svg'):
    
    stability = stability - 1.0
    
    if savename.find('VIP')!=-1:
        max_val = 0.6
        colorbar_ticks = [-.5,0,.5]
    else:
        max_val = 1.0    
        colorbar_ticks = [-1,-.5,0,.5,1.0]
      
    y_tick_loc = get_tick_loc(y_tick_vals,
                              np.arange(len(input_strengths)),
                              input_strengths
                              )
    
    x_tick_loc = get_tick_loc(x_tick_vals,
                              np.arange(len(param_values)),
                              param_values
                              )
    
    plt.figure(figsize=(5,4))
    ax = plt.subplot(111)
    ax.axis('equal')
    im = ax.imshow(stability,
                   cmap='RdBu_r',
                   interpolation='nearest',
                   vmin=-max_val,
                   vmax=max_val,
                   aspect='auto',
                   origin='lower'
                   )
    
    ax.set_yticks(y_tick_loc)
    ax.set_yticklabels([str(int(x)) for x in y_tick_vals],fontsize=label_fontsize)
    ax.set_ylabel('External input strength',fontsize=label_fontsize)
    ax.set_xticks(x_tick_loc)
    ax.set_xticklabels([str(x) for x in x_tick_vals],fontsize=label_fontsize)
    ax.set_xlabel(param_name,fontsize=label_fontsize)
    
    cbar = plt.colorbar(im,ax=ax,ticks=colorbar_ticks)
    cbar.ax.set_ylabel('E-E stability',rotation=270,fontsize=label_fontsize,labelpad=15.0)
    cbar.ax.tick_params(labelsize=label_fontsize)
    
    if save_format=='svg':
        plt.savefig(savepath+'stability_'+savename+'.svg',format='svg')
    else:
        plt.savefig(savepath+'stability_'+savename+'.png',dpi=300)
        
    plt.close()

def get_tick_loc(tick_vals,rows,row_vals):
    
    row_per_val_slope = (rows[-1] - rows[0]) / (row_vals[-1] - row_vals[0])
    offset = row_vals[0]
    tick_loc = row_per_val_slope * (tick_vals - offset)
    
    return tick_loc
    
def plot_rates_matrix(rates,
                       input_strengths,
                       param_values,
                       x_tick_vals,
                       celltype,
                       param_name,
                       title_suffix,
                       savename,
                       savepath,
                       y_tick_vals = np.arange(0,110,10),
                       label_fontsize=13,
                       save_format='png',
                       max_val=-1,
                       cbar_ticks=None
                       ):
    
    if max_val < 0:
        max_val = np.max(rates)
    
    y_tick_loc = get_tick_loc(y_tick_vals,
                              np.arange(len(input_strengths)),
                              input_strengths
                              )
    
    x_tick_loc = get_tick_loc(x_tick_vals,
                              np.arange(len(param_values)),
                              param_values
                              )
    
    plt.figure(figsize=(5,4))
    ax = plt.subplot(111)
    ax.axis('equal')
    im = ax.imshow(rates,
              cmap='Reds',
              interpolation='nearest',
              vmin=0.0,
              vmax=max_val,
              aspect='auto',
              origin='lower'
              )
    ax.set_yticks(y_tick_loc)
    ax.set_yticklabels([str(int(x)) for x in y_tick_vals],fontsize=label_fontsize)
    ax.set_ylabel('External input strength',fontsize=label_fontsize)
    ax.set_xticks(x_tick_loc)
    ax.set_xticklabels([str(x) for x in x_tick_vals],fontsize=label_fontsize)
    ax.set_xlabel(param_name,fontsize=label_fontsize)
    
    if cbar_ticks is not None:
        cbar = plt.colorbar(im,ax=ax,ticks=cbar_ticks)
    else:
        cbar = plt.colorbar(im,ax=ax)
    cbar.ax.set_ylabel(celltype + ' rate',rotation=270,fontsize=label_fontsize,labelpad=15.0)
    cbar.ax.tick_params(labelsize=label_fontsize)
    
    if save_format=='svg':
        plt.savefig(savepath+savename+'_'+celltype+'_rates.svg',format='svg')
    else:
        plt.savefig(savepath+savename+'_'+celltype+'_rates.png',dpi=300)
        
    plt.close()
    
def plot_gain_matrix(rates,
                   input_strengths,
                   param_values,
                   x_tick_vals,
                   celltype,
                   param_name,
                   title_suffix,
                   savename,
                   savepath,
                   y_tick_vals = np.arange(0,110,10),
                   label_fontsize=13,
                   save_format='svg'
                   ):
    
    y_tick_loc = get_tick_loc(y_tick_vals,
                              np.arange(len(input_strengths)),
                              input_strengths
                              )
    
    x_tick_loc = get_tick_loc(x_tick_vals,
                              np.arange(len(param_values)),
                              param_values
                              )
    
    gain = np.zeros(np.shape(rates))
    delta_input = input_strengths[1:] - input_strengths[:-1]
    gain[1:] = (rates[1:] - rates[:-1]) / (delta_input.reshape(rates.shape[0]-1,1))
    
    
    plt.figure(figsize=(5,4))
    ax = plt.subplot(111)
    ax.axis('equal')
    im = ax.imshow(100*gain,
              cmap='RdBu_r',
              interpolation='nearest',
              vmin=-30,#-np.max(np.abs(gain)),
              vmax=30,#np.max(np.abs(gain)),
              aspect='auto',
              origin='lower'
              )
    ax.set_yticks(y_tick_loc)
    ax.set_yticklabels([str(int(x)) for x in y_tick_vals],fontsize=label_fontsize)
    ax.set_ylabel('External input strength',fontsize=label_fontsize)
    ax.set_xticks(x_tick_loc)
    ax.set_xticklabels([str(x) for x in x_tick_vals],fontsize=label_fontsize)
    ax.set_xlabel(param_name,fontsize=label_fontsize)
    
    cbar = plt.colorbar(im,ax=ax)
    cbar.ax.set_ylabel(r'$\Delta$'+'Pyr gain with VIP (%)',rotation=270,fontsize=label_fontsize,labelpad=15.0)
    cbar.ax.tick_params(labelsize=label_fontsize)
    
    if save_format=='svg':
        plt.savefig(savepath+savename+'_'+celltype+'_gain.svg',format='svg')
    else:
        plt.savefig(savepath+savename+'_'+celltype+'_gain.png')
    
    plt.close()

def init_net(biased_overrep=4,
             biased_direction=0):
    
    cell_types = ['Pyr','PV','Sst','Vip']
    num_cells = [180,40,15,15]
    
    net = []
    for i_celltype,celltype in enumerate(cell_types):
        layer = np.arange(0,180,180/num_cells[i_celltype]).astype(np.float)
        if celltype=='Pyr' and biased_overrep>0.0:
            layer = np.append(layer,biased_direction*np.ones((biased_overrep,)))
        if celltype=='Sst':
            layer[:] = 0.0
        if celltype=='Vip':
            layer[:] = 0.0
        if celltype=='PV':
            layer[:] = 0.0

            
        net.append(layer)
        
    return net

def get_W(pv_frac=0.5,
          vip_scale=1.0,
          pv_sst_sum=2.5):
    
    W = np.zeros((4, 4)) # pyramidal, pv, som, vip
    W[0, 0] = 1.0 # 1.0
    
    W[0, 2] = -pv_sst_sum*(1.0-pv_frac) # pyr-som 
    W[0, 1] = -pv_sst_sum*pv_frac # pyr-pv (post-pre) 
    
    W[1, 0] = 4.0 # pv-pyr
    W[2, 0] = 6.0# som-pyr 
    W[3, 0] = 3.0# vip-pyr
    
    W[1, 1] = -0.5 # pv-pv 
    W[1, 2] = -0.3 # pv-som 
    
    W[3, 2] = -1.0 # vip-som
    W[2, 3] = -0.22*vip_scale # som-vip
    
    W[3, 1] = -0.0 # vip-pv
    W[0, 3] = -0.0 # pyr-vip
    
    W *= 0.037
    
    return W

def single_run(net,
               W,
               input_strengths,
               stimulus_oris=[0,45,90],
               sigma = 30.,
               sigma_in = 30,
               n_power = 2.,
               k = 0.04,
               spont_input=[2,2,2,10]
               ):
    
    G = get_connection_matrix(net,sigma=sigma)
    
    rates = []
    currents = []
    for i_stim,stim_ori in enumerate(stimulus_oris):
        ext_input = get_input_matrix(net,sigma_in,stim_ori=stim_ori)
        
        rates_over_inputs = []
        currents_over_inputs = []
        for i_input, input_strength in enumerate(input_strengths):
            print('input: ' + str(input_strength))
            
            # calculate total input to all of the neurons
            eff_input = multiply_inputs(ext_input,[input_strength,input_strength,0,0])
            eff_input = add_inputs(eff_input,spont_input)

            # solve for steady state
            if i_input>0: 
                r = euler_flexible_size(W, 
                                        G,
                                        eff_input, 
                                        init_rates=rates_over_inputs[-1],
                                        gain=k, 
                                        power=n_power, 
                                        tstop=10000)
            else:
                r = euler_flexible_size(W, 
                                        G,
                                        eff_input, 
                                        gain=k, 
                                        power=n_power, 
                                        tstop=10000)  
        
            c = build_current_list(W,G,r,eff_input)
                
            rates_over_inputs.append(r)
            currents_over_inputs.append(c)
    
        rates.append(rates_over_inputs)
        currents.append(currents_over_inputs)
    
    return rates, currents

def evolve_rates(r,
                 total_input,
                 dt=.1,
                 gain=.04,
                 power=2.,
                 tol=0.00001
                 ):
    
    #r_update = (dt/tau) * (r_stat - r)
    dt_over_tau = [dt/20.,dt/10.,dt/20.,dt/20.]
    
    r_stat = inputs_to_rates(total_input,gain=gain,power=power)
    minus_r = multiply_inputs(r,[-1,-1,-1,-1])
    delta_r = add_inputs(r_stat,minus_r)
    r_update = multiply_inputs(delta_r,dt_over_tau)
    
    return add_inputs(r,r_update), has_reached_tol(r_update,tol=tol)

def evolve_currents(W,G,r,external_input):

    total_input = []
    for i_post,inputs_to_post in enumerate(external_input):
        for i_pre,pre_rates in enumerate(r):
            if W[i_post,i_pre] != 0.0:
                conn_input = W[i_post,i_pre]*np.matmul(G[i_post][i_pre],pre_rates)
                inputs_to_post = inputs_to_post + conn_input
        total_input.append(inputs_to_post)
    
    return total_input

def build_current_list(W,G,r,external_input):

    c = []
    for i_post,ext_input in enumerate(external_input):
        currents_to_post = []
        currents_to_post.append(ext_input)
        for i_pre,pre_rates in enumerate(r):
            if W[i_post,i_pre] != 0.0:
                currents_to_post.append(W[i_post,i_pre]*np.matmul(G[i_post][i_pre],pre_rates))
            else:
                currents_to_post.append(np.zeros((len(ext_input),)))
        c.append(currents_to_post)

    return c

def multiply_inputs(inputs,factors_by_celltype):
    
    product_inputs = []
    for i_celltype, mult_factor in enumerate(factors_by_celltype):
        product_inputs.append(inputs[i_celltype] * mult_factor)
        
    return product_inputs

def add_inputs(inputs,inputs_to_celltype):

    summed_inputs = []
    for i_celltype, input_to_add in enumerate(inputs_to_celltype):
        summed_inputs.append(inputs[i_celltype] + input_to_add)
        
    return summed_inputs

def inputs_to_rates(inputs,gain=.04,power=2.):
    
    rates = []
    for i_celltype,cell_inputs in enumerate(inputs):
        rates.append(phi(cell_inputs, gain, power))
        
    return rates

def has_reached_tol(updates,tol=0.00001):
    for i_celltype,rate_updates in enumerate(updates):
        if np.any(rate_updates>tol):
            return False
    return True
    
def get_input_matrix(net,sigma_in,stim_ori=0.0,ori_max=180.0,bias_sigma=90.0):
    
    ori_med = ori_max/2.0
    
    layer_weight = [1.0,0.5,0.0,0.0]
    
    stim_ori = np.abs(stim_ori)
    
    ori_diff = np.arange(180).astype(np.float)
    ori_diff = np.where(ori_diff>ori_med,ori_max-ori_diff,ori_diff)
    sum_at_default = (np.exp(-.5 * (ori_diff**2) / 30.0**2) / 30. * 30.).sum()
    sum_at_sigma = (np.exp(-.5 * (ori_diff**2) / sigma_in**2) / sigma_in * 30.).sum()
    
    thalamic_bias = np.exp(-.5 * (np.mod(stim_ori,int(ori_max))**2) /bias_sigma**2)
    thalamic_bias /= np.mean(np.exp(-.5 * ((np.arange(0,180,180/6).astype(np.float))**2)/bias_sigma**2))
    
    ext_input = []
    for i_celltype, cell_peak_oris in enumerate(net):        
            
        ori_diff = np.abs(stim_ori - cell_peak_oris)
        ori_diff = np.where(ori_diff>ori_med,ori_max-ori_diff,ori_diff)
        layer_input = np.exp(-.5 * (ori_diff**2) / sigma_in**2) / sigma_in * 30.
        
        layer_input *= sum_at_default / sum_at_sigma
        
        if i_celltype==1:
            #calculate magnitude of thalamic inputs as if PV tuning is broad.
            cell_peak_oris = np.arange(0,180,180/len(cell_peak_oris)).astype(np.float)
            
            ori_diff = np.abs(stim_ori - cell_peak_oris)
            ori_diff = np.where(ori_diff>ori_med,ori_max-ori_diff,ori_diff)
            layer_input = np.exp(-.5 * (ori_diff**2) / sigma_in**2) / sigma_in * 30.
        
            layer_input *= sum_at_default / sum_at_sigma
            
            layer_input = np.ones((len(cell_peak_oris),))*layer_input.mean()
    
        layer_input *= layer_weight[i_celltype]
        layer_input *= thalamic_bias

        ext_input.append(layer_input)
    
    return ext_input

def euler_flexible_size(W,
                        G,
                        ext_input, 
                        init_rates = None,
                        gain=.04, 
                        power=2., 
                        tstop=500, 
                        dt=.1, 
                        tol=0.00001,
                        warn_converge=True):
    
    T = int(tstop/dt)
    
    if init_rates is None:
        r = inputs_to_rates(ext_input,gain=gain,power=power)
    else:
        r = init_rates

    for i in range(T):
        
        total_input = evolve_currents(W,G,r,ext_input)
            
        r, reached_tol = evolve_rates(r,total_input)
        
        if reached_tol:
            break
        
    if warn_converge and i==(T-1):
        print('Did not reach steady state!')
        
    return r       

def get_connection_matrix(net,ori_max=180.0,sigma=30.,sigma_broad=100.,sigma_vip=30.):

    ori_med = ori_max / 2.0
    
    ori_diff = np.arange(180).astype(np.float)
    ori_diff = np.where(ori_diff>ori_med,ori_max-ori_diff,ori_diff)
    normed_sum = (np.exp(-.5 * (ori_diff**2) / 30.0**2) / 30. * 30.).sum()
    
    conn_mat = []
    for i_post,post_oris in enumerate(net):
        conn_to_post = []
        for i_pre,pre_oris in enumerate(net):
            
            #shortest distance between pre and post ori preferences
            ori_mesh1, ori_mesh2 = np.meshgrid(pre_oris,post_oris)
            ori_diff = np.abs(ori_mesh1 - ori_mesh2)
            ori_diff = np.where(ori_diff>ori_med,ori_max-ori_diff,ori_diff)
            
            pre_to_post = np.exp(-.5 * ((ori_diff)**2) / (sigma**2)) / sigma * 30.

            if i_post==1 or i_pre==1:
                pre_to_post = np.exp(-.5 * ((ori_diff)**2) / (sigma_broad**2)) / sigma_broad * 30.

            if i_post==2 or i_pre==2:
                pre_to_post = np.exp(-.5 * ((ori_diff)**2) / (sigma_broad**2)) / sigma_broad * 30.
                
            if i_post==3 or i_pre==3:
                pre_to_post = np.exp(-.5 * ((ori_diff)**2) / (sigma_vip**2)) / sigma_vip * 30.
                
            #conserve total input to each neuron
            pre_to_post *= normed_sum / (pre_to_post.sum(axis=1,keepdims=True))

            conn_to_post.append(pre_to_post)
        conn_mat.append(conn_to_post)    
        
    return conn_mat
           
def phi(x, gain=1., power=2.):
    x[x < 0.] = 0.
    return gain * (x**power)

def plot_fraction_currents(currents,input_strengths,savename,savepath,save_format='svg',font_size=13):
       
    Pyr_currents, __, __, __ = extract_population_currents(currents)
          
    x_ticks = np.arange(0,81,20)
    y_ticks = np.linspace(0,0.2,3)
    
    external_exc = Pyr_currents[:,0]
    internal_exc = Pyr_currents[:,1]
    
    frac_external = external_exc / (external_exc + internal_exc)
    frac_internal = internal_exc / (external_exc + internal_exc)
    
    plt.figure(figsize=(5,4))
    ax1 = plt.subplot(121)
    ax1.plot(input_strengths,100.0 * frac_internal,c='k',linewidth=2.0)
    ax1.plot(input_strengths,100.0 * frac_external,c='k',linewidth=2.0,linestyle='dashed')

    ax1.legend(['network E','external E'])
    ax1.set_xlabel('External input strength',fontsize=font_size)
    ax1.set_ylabel('Percent of input',fontsize=font_size)
    ax1.set_xlim(-2,np.max(input_strengths))
    ax1.set_ylim(0,100)
    ax1.set_xticks(x_ticks)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    
    total_inhib = np.sum(-Pyr_currents[:,2:4],axis=1)
    frac_network_E = internal_exc / (internal_exc + total_inhib)
    
    ax2 = plt.subplot(122)
    ax2.plot(input_strengths,frac_network_E,c='k',linewidth=2.0)

    ax2.set_xlabel('External input strength',fontsize=font_size)
    ax2.set_ylabel('E_N / (E_N + I)',fontsize=font_size)
    ax2.set_xlim(-2,np.max(input_strengths))
    ax2.set_ylim(0,0.2)
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels([str(x) for x in x_ticks],fontsize=font_size)
    ax2.set_yticks(y_ticks)
    ax2.set_yticklabels([str(y) for y in y_ticks],fontsize=font_size)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
 
    plt.tight_layout()
    
    if save_format=='svg':
        plt.savefig(savepath+savename+'_frac_currents.svg',format='svg')    
    else:
        plt.savefig(savepath+savename+'_frac_currents.png',dpi=300)
    plt.close()

def plot_E_currents(currents,input_strengths,savename,savepath,save_format='svg',font_size=13):
       
    Pyr_currents, __, __, __ = extract_currents(currents)
    
    x_ticks = np.arange(0,81,20)
    y_ticks = np.arange(0,120,20)  
    
    external_exc = Pyr_currents[:,0]
    internal_exc = Pyr_currents[:,1]
    
    plt.figure(figsize=(5,4))
    ax = plt.subplot(121)
    ax.plot(external_exc,internal_exc,c='r',linewidth=2.0)
    ax.plot(external_exc,external_exc,c='r',linewidth=2.0,linestyle='dashed')

    PV_inhib = -Pyr_currents[:,2]
    SST_inhib = -Pyr_currents[:,3]
    total_inhib = np.sum(-Pyr_currents[:,2:4],axis=1)

    ax.plot(external_exc,total_inhib,c='b',linewidth=2.0)
    ax.plot(external_exc,PV_inhib,c='b',linewidth=2.0,linestyle='dashed')
    ax.plot(external_exc,SST_inhib,c='b',linewidth=2.0,linestyle='dotted')

    net_input = external_exc + internal_exc - total_inhib
    ax.plot(external_exc,net_input,c='g',linewidth=2.0)

    ax.legend(['network E','external E','total I','I_pv','I_sst','net input'])
    ax.set_xlabel('External input strength',fontsize=font_size)
    ax.set_ylabel('Current',fontsize=font_size)
    ax.set_xlim(-2,np.max(input_strengths))
    ax.set_ylim(0,100)
    #ax.set_yticks([0,5,10,15])
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(x) for x in x_ticks],fontsize=font_size)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(y) for y in y_ticks],fontsize=font_size)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
 
    plt.tight_layout()
    
    if save_format=='svg':
        plt.savefig(savepath+savename+'_E_currents.svg',format='svg')
    else:
        plt.savefig(savepath+savename+'_E_currents.png',dpi=300)
        
    plt.close()

def plot_rates(rates,input_strengths,savename,savepath,save_format='svg',font_size=13):
    
    #labels = ['Cux2','PV','Sst','Vip']
    colors = ['#a92e66','b','#7B5217','#b49139']
       
    Pyr_rates, PV_rates, Sst_rates, Vip_rates = extract_rates(rates)
             
    x_ticks = np.arange(0,81,20)
    y_ticks = np.arange(0,30,5)
    
    plt.figure(figsize=(5,4))
    ax = plt.subplot(121)
    ax.plot(input_strengths,Pyr_rates,c=colors[0],linewidth=2.0)
    ax.plot(input_strengths,PV_rates,c=colors[1],linewidth=2.0,linestyle='dashed')
    ax.plot(input_strengths,Sst_rates,c=colors[2],linewidth=2.0)
    ax.plot(input_strengths,Vip_rates,c=colors[3],linewidth=2.0)
    ax.set_xlabel('External input strength',fontsize=font_size)
    ax.set_ylabel('Firing rate',fontsize=font_size)
    ax.set_xlim(-2,np.max(input_strengths))
    ax.set_ylim(-1,25)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(x) for x in x_ticks],fontsize=font_size)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(y) for y in y_ticks],fontsize=font_size)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.tight_layout()
    
    if save_format=='svg':
        plt.savefig(savepath+savename+'.svg',format='svg')
    else:
        plt.savefig(savepath+savename+'.png',dpi=300)
    
    plt.close()

def plot_gain(rates,input_strengths,savename,savepath,save_format='svg'):
    
    #labels = ['Cux2','PV','Sst','Vip']
    colors = ['#a92e66','b','#7B5217','#b49139']
       
    Pyr_rates, PV_rates, Sst_rates, Vip_rates = extract_rates(rates)
              
    gain = np.ediff1d(Pyr_rates)
    
    plt.figure(figsize=(5,4))
    ax = plt.subplot(121)
    ax.plot(input_strengths[1:],gain,c=colors[0],linewidth=2.0)
    
    ax.set_xlabel('External input strength',fontsize=14)
    ax.set_ylabel('Gain',fontsize=14)
    ax.set_xlim(-2,np.max(input_strengths))
    #ax.set_ylim(-0.2,0.6)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()
    
    if save_format=='svg':
        plt.savefig(savepath+savename+'_gain.svg',format='svg')
    else:
        plt.savefig(savepath+savename+'_gain.png',dpi=300)
    
    plt.close()

def extract_currents(currents,which_neuron=0,num_presyn=5):
    
    num_input_steps = len(currents)
    
    Pyr_currents = np.zeros((num_input_steps,num_presyn))
    PV_currents = np.zeros((num_input_steps,num_presyn))
    Sst_currents = np.zeros((num_input_steps,num_presyn))
    Vip_currents = np.zeros((num_input_steps,num_presyn))
    for i_input,currents_at_input in enumerate(currents):
        for i_pre in range(num_presyn):
            Pyr_currents[i_input,i_pre] = currents_at_input[0][i_pre][which_neuron]
            PV_currents[i_input,i_pre] = currents_at_input[1][i_pre][which_neuron]
            Sst_currents[i_input,i_pre] = currents_at_input[2][i_pre][which_neuron]
            Vip_currents[i_input,i_pre] = currents_at_input[3][i_pre][which_neuron]
        
    return Pyr_currents, PV_currents, Sst_currents, Vip_currents

def extract_population_currents(currents,num_presyn=5):
    
    num_input_steps = len(currents)
    
    Pyr_currents = np.zeros((num_input_steps,num_presyn))
    PV_currents = np.zeros((num_input_steps,num_presyn))
    Sst_currents = np.zeros((num_input_steps,num_presyn))
    Vip_currents = np.zeros((num_input_steps,num_presyn))
    for i_input,currents_at_input in enumerate(currents):
        for i_pre in range(num_presyn):
            Pyr_currents[i_input,i_pre] = currents_at_input[0][i_pre].mean()
            PV_currents[i_input,i_pre] = currents_at_input[1][i_pre].mean()
            Sst_currents[i_input,i_pre] = currents_at_input[2][i_pre].mean()
            Vip_currents[i_input,i_pre] = currents_at_input[3][i_pre].mean()
        
    return Pyr_currents, PV_currents, Sst_currents, Vip_currents

def extract_rates(rates):
    
    num_input_steps = len(rates)
    
    Pyr_rates = np.zeros((num_input_steps,))
    PV_rates = np.zeros((num_input_steps,))
    Sst_rates = np.zeros((num_input_steps,))
    Vip_rates = np.zeros((num_input_steps,))
    for i_input,rates_at_input in enumerate(rates):
        Pyr_rates[i_input] = rates_at_input[0][0]
        PV_rates[i_input] = rates_at_input[1][0]
        Sst_rates[i_input] = rates_at_input[2][0]
        Vip_rates[i_input] = rates_at_input[3][0]
        
    return Pyr_rates, PV_rates, Sst_rates, Vip_rates

def extract_population_rates(rates):
    
    num_input_steps = len(rates)
    
    Pyr_rates = np.zeros((num_input_steps,))
    PV_rates = np.zeros((num_input_steps,))
    Sst_rates = np.zeros((num_input_steps,))
    Vip_rates = np.zeros((num_input_steps,))
    for i_input,rates_at_input in enumerate(rates):
        Pyr_rates[i_input] = rates_at_input[0].mean()
        PV_rates[i_input] = rates_at_input[1].mean()
        Sst_rates[i_input] = rates_at_input[2].mean()
        Vip_rates[i_input] = rates_at_input[3].mean()
        
    return Pyr_rates, PV_rates, Sst_rates, Vip_rates

def plot_summed_tuning(rates,input_strengths,savepath,directions_to_sample=[0,45,90],save_format='svg'):
    
    inputs_to_plot = np.arange(0,int(np.max(input_strengths))+1,2)
    directions_to_sample = np.array(directions_to_sample).astype(np.int)
    
    input_idx = np.zeros((len(inputs_to_plot),),dtype=np.int)
    for i_input,input_mag in enumerate(inputs_to_plot):
        first_idx = np.argwhere(input_strengths>=input_mag)[0,0]
        input_idx[i_input] = int(first_idx)
    
    pooled_mat = np.zeros((4,len(directions_to_sample),len(inputs_to_plot)))
    for i_ori,ori in enumerate(directions_to_sample):
        Pyr_rates, PV_rates, Sst_rates, Vip_rates = extract_population_rates(rates[i_ori])
        pooled_mat[0,i_ori,:] = Pyr_rates[input_idx] - Pyr_rates[0]
        pooled_mat[1,i_ori,:] = PV_rates[input_idx] - PV_rates[0]
        pooled_mat[2,i_ori,:] = Sst_rates[input_idx] - Sst_rates[0]
        pooled_mat[3,i_ori,:] = Vip_rates[input_idx] - Vip_rates[0]
    
    #center 0 degrees:
    shift_left = np.argwhere(directions_to_sample<0)[:,0]
    if len(shift_left)>0:
        shift_right = np.setxor1d(np.arange(len(directions_to_sample)),shift_left)
        temp = pooled_mat.copy()
        temp[:,:len(shift_left),:] = pooled_mat[:,shift_left,:]
        temp[:,len(shift_left):,:] = pooled_mat[:,shift_right,:]
        pooled_mat = temp
        temp2 = directions_to_sample[shift_left]
        temp2 = np.append(temp2,directions_to_sample[shift_right])
        directions_to_sample = temp2    
    
    y_tick_vals = np.arange(0,81,20)
    y_tick_loc = get_tick_loc(y_tick_vals,
                              np.arange(len(inputs_to_plot)),
                              inputs_to_plot
                              )
    
    labels = ['CUX2','PV','SST','VIP']
    for cell_type in range(4):
        
        plot_mat = pooled_mat[cell_type].T
        
        if labels[cell_type]=='SST':
            max_resp=20.0
        else:
            max_resp = 10.0#np.max(np.abs(plot_mat))
        
        plt.figure(figsize=(5,4))
        ax = plt.subplot(111)
        im=ax.imshow(plot_mat,vmin=-max_resp,vmax=max_resp,interpolation='nearest',cmap='RdBu_r',origin='lower')
        ax.set_xlabel('Direction',fontsize=14)
        ax.set_ylabel('External input strength',fontsize=14)
        ax.set_yticks(y_tick_loc)
        ax.set_yticklabels([str(int(x)) for x in y_tick_vals],fontsize=14)
        ax.set_xticks(np.arange(len(directions_to_sample)))
        ax.set_xticklabels([str(x) for x in directions_to_sample],fontsize=14)
        ax.set_title(labels[cell_type],fontsize=14)
        ax.set_aspect(1.0/ax.get_data_ratio())
        cbar = plt.colorbar(im,ax=ax,ticks=[-max_resp,0,max_resp])
        cbar.set_label('Firing rate - baseline',fontsize=14,rotation=270,labelpad=15.0)
        cbar.ax.tick_params(labelsize=14)
        
        if save_format=='svg':
            plt.savefig(savepath+labels[cell_type]+'_summed_tuning.svg',format='svg')    
        else:
            plt.savefig(savepath+labels[cell_type]+'_summed_tuning.png',dpi=300)
        
        plt.close()
    
def plot_connection_distributions(net,W,savepath,save_format='svg'):
    
    pop_names = ['CUX2','PV','SST','VIP']
    colors = ['#a92e66','b','#7B5217','#b49139']
    
    conn_mat = get_connection_matrix(net)
    
    inhib_mat = np.zeros((3,4))
    
    plt.figure(figsize=(3,3))
    ax = plt.subplot(111)
    for i_post,post_name in enumerate(pop_names):
        for i_pre,pre_name in enumerate(pop_names):
    
            conn_dist = W[i_post,i_pre]*conn_mat[i_post][i_pre][0,:]
            
            if i_pre==0:
                conn_dist = np.append(conn_dist[90:180],conn_dist[:90])
                if np.abs(conn_dist).sum()>0:
                    if i_post==1:
                        ax.plot(np.linspace(-90,90,180),
                                conn_dist,
                                colors[i_post]+'--',
                                linewidth=2.0)
                    else:
                        ax.plot(np.linspace(-90,90,180),
                                conn_dist,
                                c=colors[i_post],
                                linewidth=2.0)
                        
                    ax.text(55,0.15-0.018*i_post,pop_names[i_post],color=colors[i_post],fontsize=12)
            else:
                inhib_mat[i_pre-1,i_post] = conn_dist.sum()
                
        x_ticks = [-45,0,45]
        y_ticks = [0.0,0.05,0.1,0.15]
        
        ax.set_ylim(0.0,0.16)
        ax.set_xlim(-90,90)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(x) for x in x_ticks],fontsize=12)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([str(x) for x in y_ticks],fontsize=12)
        
        ax.set_xlabel(r'$\Delta$ Orientation',fontsize=12)
        ax.set_ylabel('Weight from Cux2',fontsize=12)

    plt.tight_layout()
    
    if save_format=='svg':
        plt.savefig(savepath+'cux2_distribution.svg',format='svg')
    else:    
        plt.savefig(savepath+'cux2_distribution.png',dpi=300)
    
    plt.close()
    
    plt.figure(figsize=(3,3))
    ax = plt.subplot(111)
    
    max_W = np.max(np.abs(inhib_mat))
    
    presynaptic = ['PV','SST','VIP']
    postsynaptic = ['CUX2','PV','SST','VIP']
    ax.imshow(inhib_mat,vmin=-max_W,vmax=max_W,cmap='RdBu_r',interpolation='nearest',origin='lower')
    ax.set_yticks(np.arange(len(presynaptic)))
    ax.set_yticklabels(presynaptic,fontsize=12)
    ax.set_xticks(np.arange(len(postsynaptic)))
    ax.set_xticklabels(postsynaptic,fontsize=12)
    ax.set_ylabel('Presynaptic',fontsize=12)
    ax.set_xlabel('Postsynaptic',fontsize=12)
    
    for i_pre, pre_type in enumerate(presynaptic):
        for i_post, post_type in enumerate(postsynaptic):
            if abs(inhib_mat[i_pre,i_post])>=(max_W/2.0):
                text_color = "w"
            else:
                text_color = "k"
            ax.text(i_post,i_pre,round(inhib_mat[i_pre,i_post],1),ha="center", va="center",color=text_color)
    
    plt.tight_layout()
    
    if save_format=='svg':
        plt.savefig(savepath+'inhib_conn_mat.svg',format='svg')
    else:
        plt.savefig(savepath+'inhib_conn_mat.png',dpi=300)
    
    plt.close()
    
if __name__ == '__main__':
    white_board()