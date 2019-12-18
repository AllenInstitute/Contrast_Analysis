#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 16:27:05 2019

@author: dan
"""

import numpy as np
import matplotlib.pyplot as plt

def white_board():
    
    savepath = '/Users/dan/Desktop/SSN/'
    
    run_model(savepath)

def run_model(savepath):
    
    W = np.zeros((4, 4)) # pyramidal, pv, som, vip
    W[0, 0] = 1.0 # 1.0
    
    W[0, 2] = -1.5 # pyr-som -0.5
    W[0, 1] = -1.0 # pyr-pv (post-pre) -0.5
    
    W[1, 0] = 4.0 # pv-pyr 1
    W[2, 0] = 6.0 # som-pyr .5
    W[3, 0] = 3.0 # vip-pyr 1.0
    
    W[1, 1] = -0.5 # pv-pv -.5
    W[1, 2] = -0.5 # pv-som -.33
    
    W[3, 2] = -1.0 # vip-som  -0.77
    W[2, 3] = -0.21 # som-vip, -.15
    
    W[3, 1] = -0.0 # vip-pv -0.22
    W[0, 3] = -0.0 # pyr-vip
    
    W *= 0.037
    
    net = init_net()
    
#    orig_pop = [184.,40.,15.,15.]
#    new_pop = [184,180,180,180]
#    for i_post in range(4):
#        for i_pre in range(4):
#            W[i_post,i_pre] *= orig_pop[i_pre] / new_pop[i_pre]
    
    N_input_steps = 41
    input_strengths = np.linspace(0., 40., N_input_steps)
    stim_oris = [0,30,60,90,-60,-30]

    rates = single_run(net,
                       W,
                       input_strengths,
                       stimulus_oris=stim_oris)
    
    plot_connection_distributions2(net,W,savepath)
    
    W /= 0.037
    for i_stim,stim_ori in enumerate(stim_oris):
        plot_rates(rates[i_stim],
                   input_strengths,
                   W,
                   'pyr_extra_0deg_at_'+str(int(stim_ori)),
                   savepath)
        plot_gain(rates[i_stim],
                   input_strengths,
                   W,
                   'pyr_extra_0deg_at_'+str(int(stim_ori)),
                   savepath)

    plot_summed_tuning(rates,
                       input_strengths,
                       savepath,
                       directions_to_sample=stim_oris)

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

def single_run(net,
               W,
               input_strengths,
               stimulus_oris=[0,45,90],
               sigma = 10.,
               sigma_in = 30,
               n_power = 2.,
               k = 0.04,
               spont_input=[2,2,2,10],
               run_input=[0,0,0,0]):
    
    rates = []
    for i_stim,stim_ori in enumerate(stimulus_oris):
        ext_input = get_input_matrix(net,sigma_in,stim_ori=stim_ori)
        
        rates_over_inputs = []
        for i_input, input_strength in enumerate(input_strengths):
            print('input: ' + str(input_strength))
            
            # calculate total input to all of the neurons
            eff_input = multiply_inputs(ext_input,[input_strength,input_strength,0,0])
            eff_input = add_inputs(eff_input,spont_input)
            eff_input = add_inputs(eff_input,run_input)

            # solve for steady state
            if i_input>0: 
                r = euler_flexible_size(net,
                                       W, 
                                       eff_input, 
                                       init_rates=rates_over_inputs[-1],
                                       gain=k, 
                                       power=n_power, 
                                       tstop=10000, 
                                       sigma=sigma)
            else:
                r = euler_flexible_size(net,
                                       W, 
                                       eff_input, 
                                       gain=k, 
                                       power=n_power, 
                                       tstop=10000, 
                                       sigma=sigma)  
                
            rates_over_inputs.append(r)
    
        rates.append(rates_over_inputs)
    
    return rates

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
    
    layer_weight = [1.0,1.0,0.0,0.0]
    
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

def euler_flexible_size(net,
                        W, 
                        eff_input, 
                        init_rates = None,
                        gain=.04, 
                        power=2., 
                        sigma=32.,
                        tstop=500, 
                        dt=.1, 
                        tol=0.00001):
    
    T = int(tstop/dt)
    
    dt_over_tau = [dt/20.,dt/10.,dt/20.,dt/20.]
    
    G = get_connection_matrix(net,sigma=sigma)
    
    if init_rates is None:
        r = inputs_to_rates(eff_input,gain=gain,power=power)
    else:
        r = init_rates

    for i in range(T):
        
        total_input = []
        for i_post,inputs_to_post in enumerate(eff_input):
            for i_pre,pre_rates in enumerate(r):
                if W[i_post,i_pre] != 0.0:
                    conn_input = W[i_post,i_pre]*np.matmul(G[i_post][i_pre],pre_rates)
                    inputs_to_post = inputs_to_post + conn_input
            total_input.append(inputs_to_post)
        
        r_stat = inputs_to_rates(total_input,gain=gain,power=power)
        minus_r = multiply_inputs(r,[-1,-1,-1,-1])
        delta_r = add_inputs(r_stat,minus_r)
        
        r_update = multiply_inputs(delta_r,dt_over_tau)
        r = add_inputs(r,r_update)
        
        if has_reached_tol(r_update,tol=tol):
            break
        
    if i==(T-1):
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

def plot_rates(rates,input_strengths,W,savename,savepath):
    
    labels = ['Cux2','PV','Sst','Vip']
    colors = ['#a92e66','b','#7B5217','#b49139']
       
    Pyr_rates, PV_rates, Sst_rates, Vip_rates = extract_rates(rates)
              
    plt.figure(figsize=(5,4))
    ax = plt.subplot(121)
    ax.plot(input_strengths,Pyr_rates,c=colors[0],linewidth=2.0)
    ax.plot(input_strengths,PV_rates,c=colors[1],linewidth=2.0,linestyle='dashed')
    ax.plot(input_strengths,Sst_rates,c=colors[2],linewidth=2.0)
    ax.plot(input_strengths,Vip_rates,c=colors[3],linewidth=2.0)
    #ax.legend(labels)
    ax.set_xlabel('Input',fontsize=14)
    ax.set_ylabel('Firing Rate',fontsize=14)
    ax.set_xlim(-2,np.max(input_strengths))
    ax.set_ylim(-2,15)
    ax.set_yticks([0,5,10,15])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    #ax.set_title(savename,fontsize=14)
    
#    ax = plt.subplot(122)
#    max_W = np.max(np.abs(W))
#    
#    plot_W = np.zeros((4,5))
#    plot_W[:,1:] = W
#    plot_W[0,0] = 1.0#external inputs only go to Pyr and PV
#    plot_W[1,0] = 1.0
#    
#    presynaptic = ['Ext','Cux2','PV','Sst','VIP']
#    postsynaptic = ['Cux2','PV','Sst','VIP']
#    ax.imshow(plot_W,vmin=-max_W,vmax=max_W,cmap='RdBu_r',interpolation='none',origin='lower')
#    ax.set_xticks(np.arange(len(presynaptic)))
#    ax.set_xticklabels(presynaptic,fontsize=12)
#    ax.set_yticks(np.arange(len(postsynaptic)))
#    ax.set_yticklabels(postsynaptic,fontsize=12)
#    ax.set_xlabel('Presynaptic',fontsize=14)
#    ax.set_ylabel('Postsynaptic',fontsize=14)
#    
#    for i_pre, pre_type in enumerate(presynaptic):
#        for i_post, post_type in enumerate(postsynaptic):
#            if abs(plot_W[i_post,i_pre])>=1.0:
#                text_color = "w"
#            else:
#                text_color = "k"
#            ax.text(i_pre,i_post,round(plot_W[i_post,i_pre],2),ha="center", va="center",color=text_color)
#   
    plt.tight_layout()
    plt.savefig(savepath+savename+'.png',dpi=300)
    plt.close()

def plot_gain(rates,input_strengths,W,savename,savepath):
    
    labels = ['Cux2','PV','Sst','Vip']
    colors = ['#a92e66','b','#7B5217','#b49139']
       
    Pyr_rates, PV_rates, Sst_rates, Vip_rates = extract_rates(rates)
              
    gain = np.ediff1d(Pyr_rates)
    
    plt.figure(figsize=(5,4))
    ax = plt.subplot(121)
    ax.plot(input_strengths[1:],gain,c=colors[0],linewidth=2.0)
    
    ax.set_xlabel('Input',fontsize=14)
    ax.set_ylabel('Gain',fontsize=14)
    ax.set_xlim(-2,np.max(input_strengths))
    ax.set_ylim(-0.2,0.6)
    #ax.set_yticks([0,5,10,15])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    #ax.set_title(savename,fontsize=14)
    

    plt.tight_layout()
    plt.savefig(savepath+savename+'_gain.png',dpi=300)
    plt.close()

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

def plot_summed_tuning(rates,input_strengths,savepath,directions_to_sample=[0,45,90]):
    
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
    
    y_ticks = np.arange(0,21,5)
    
    labels = ['Cux2','PV','Sst','Vip']
    for cell_type in range(4):
        
        plot_mat = pooled_mat[cell_type].T
        
        max_resp = 10.0#np.max(np.abs(plot_mat))
        
        plt.figure(figsize=(5,4))
        ax = plt.subplot(111)
        im=ax.imshow(plot_mat,vmin=-max_resp,vmax=max_resp,interpolation='none',cmap='RdBu_r',origin='lower')
        ax.set_xlabel('Direction',fontsize=14)
        ax.set_ylabel('Input',fontsize=14)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([str(int(x)) for x in inputs_to_plot[y_ticks]],fontsize=8)
        ax.set_xticks(np.arange(len(directions_to_sample)))
        ax.set_xticklabels([str(x) for x in directions_to_sample],fontsize=8)
        ax.set_title(labels[cell_type],fontsize=14)
        ax.set_aspect(1.0/ax.get_data_ratio())
        cbar = plt.colorbar(im,ax=ax,ticks=[-10,0,10])
        cbar.set_label('Response',fontsize=8,rotation=270,labelpad=15.0)
        plt.savefig(savepath+labels[cell_type]+'_summed_tuning.png',dpi=300)
        plt.close()

def plot_connection_distributions(net,W,savepath):
    
    pop_names = ['Cux2','PV','Sst','Vip']
    colors = ['#a92e66','b--','#7B5217','#b49139']
    
    conn_mat = get_connection_matrix(net)
    
    plt.figure(figsize=(6,6))
    
    for i_post,post_name in enumerate(pop_names):
        
        ax = plt.subplot(2,2,i_post+1)
        for i_pre,pre_name in enumerate(pop_names):
    
            conn_dist = W[i_post,i_pre]*conn_mat[i_post][i_pre][0,:]
            
            if i_pre==0:
                conn_dist = np.append(conn_dist[90:180],conn_dist[:90])
            else:
                conn_dist = conn_dist.sum()*np.ones((180,))/180.0

            if np.abs(conn_dist).sum()>0:
                if i_pre==1:
                    ax.plot(np.linspace(-90,90,180),
                            conn_dist,
                            colors[i_pre],
                            linewidth=2.0)
                else:
                    ax.plot(np.linspace(-90,90,180),
                            conn_dist,
                            c=colors[i_pre],
                            linewidth=2.0)
                
        #plot PV inputs last
        i_pre = 1
        conn_dist = W[i_post,i_pre]*conn_mat[i_post][i_pre][0,:]
        conn_dist = conn_dist.sum()*np.ones((180,))/180.0 
        if np.abs(conn_dist).sum()>0:
            ax.plot(np.linspace(-90,90,180),
                    conn_dist,
                    colors[i_pre],
                    linewidth=2.0)
        
        ax.plot([-90,90],[0,0],'k',linewidth=0.6)
        
        ax.text(0,0.18,'Postsynaptic '+post_name,fontsize=10,ha='center')
        #ax.set_ylim(-0.2,0.2)
        ax.set_xlim(-90,90)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xticks([-45,0,45])
        ax.set_yticks([-0.2,-0.1,0.0,0.1,0.2])
        
        if i_post<2:
            ax.set_xticklabels(['','',''],fontsize=0.1)
        if np.mod(i_post,2)==1:
            ax.set_yticklabels(['','','','',''],fontsize=0.1)
        
        if i_post>1:
            ax.set_xlabel(r'$\Delta$ Orientation',fontsize=10)
        if np.mod(i_post,2)==0:
            ax.set_ylabel('Connection Strength',fontsize=10)
            
        if i_post==3:
            ax.legend(['Presynaptic '+p for p in pop_names],fontsize=8)
    
    plt.tight_layout()
    plt.savefig(savepath+'connection_distribution.png',dpi=300)
    plt.close()
    
def plot_connection_distributions2(net,W,savepath):
    
    pop_names = ['Cux2','PV','Sst','Vip']
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
                        
                    ax.text(55,0.15-0.015*i_post,pop_names[i_post],color=colors[i_post])
            else:
                inhib_mat[i_pre-1,i_post] = conn_dist.sum()
                
        
        ax.set_ylim(0.0,0.16)
        ax.set_xlim(-90,90)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xticks([-45,0,45])
        ax.set_yticks([0.0,0.05,0.1,0.15])
        
        ax.set_xlabel(r'$\Delta$ Orientation',fontsize=10)
        ax.set_ylabel('Weight from Cux2',fontsize=10)
            
        #ax.legend([p for p in pop_names],fontsize=8)
    
    plt.tight_layout()
    plt.savefig(savepath+'cux2_distribution.png',dpi=300)
    plt.close()
    
    plt.figure(figsize=(3,3))
    ax = plt.subplot(111)
    
    max_W = np.max(np.abs(inhib_mat))
    
    presynaptic = ['PV','Sst','VIP']
    postsynaptic = ['Cux2','PV','Sst','VIP']
    ax.imshow(inhib_mat,vmin=-max_W,vmax=max_W,cmap='RdBu_r',interpolation='none',origin='lower')
    ax.set_yticks(np.arange(len(presynaptic)))
    ax.set_yticklabels(presynaptic,fontsize=12)
    ax.set_xticks(np.arange(len(postsynaptic)))
    ax.set_xticklabels(postsynaptic,fontsize=12)
    ax.set_ylabel('Presynaptic',fontsize=14)
    ax.set_xlabel('Postsynaptic',fontsize=14)
    
    for i_pre, pre_type in enumerate(presynaptic):
        for i_post, post_type in enumerate(postsynaptic):
            if abs(inhib_mat[i_pre,i_post])>=(max_W/2.0):
                text_color = "w"
            else:
                text_color = "k"
            ax.text(i_post,i_pre,round(inhib_mat[i_pre,i_post],1),ha="center", va="center",color=text_color)
    
    plt.tight_layout()
    plt.savefig(savepath+'inhib_conn_mat.png',dpi=300)
    plt.close()
    
if __name__ == '__main__':
    white_board()