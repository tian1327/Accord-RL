import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import pickle
import time
from matplotlib.ticker import StrMethodFormatter


def read_data(fn, NUMBER_SIMULATIONS, NUMBER_EPISODES_o):
    label = fn.split('_')[-2].split('/')[-1]

    # ----------------- Read data from file ----------------- #
    obj_opsrl = np.zeros((NUMBER_SIMULATIONS, NUMBER_EPISODES_o))
    con_opsrl = np.zeros((NUMBER_SIMULATIONS, NUMBER_EPISODES_o))

    for i in range(NUMBER_SIMULATIONS):
        
        filename = fn
        f = open(filename, 'rb')
        objs = []
        cons = []

        j = 0
        while 1:
            try:
                j += 1
                if label == 'CONTEXTUAL':
                    [R_est_err, C_est_err, min_eign_sbp_list, min_eign_cvd_list, NUMBER_SIMULATIONS, NUMBER_EPISODES, ObjRegret, ConRegret, pi_k, NUMBER_INFEASIBILITIES, q_k] = pickle.load(f) # load results chunk by chunk
                else:
                    [NUMBER_SIMULATIONS, NUMBER_EPISODES, ObjRegret, ConRegret, pi_k, NUMBER_INFEASIBILITIES, q_k] = pickle.load(f) # load results chunk by chunk
                objs.append(ObjRegret)
                cons.append(ConRegret)

            except EOFError:
                break
        f.close()

        flat_listobj = [item for sublist in objs for item in sublist] # flatten the list
        flat_listcon = [item for sublist in cons for item in sublist]
        
        #print(len(flat_listobj))
        obj_opsrl[i, :] = np.copy(flat_listobj[0:NUMBER_EPISODES_o])
        con_opsrl[i, :] = np.copy(flat_listcon[0:NUMBER_EPISODES_o])
    
    obj_opsrl_mean = np.mean(obj_opsrl, axis = 0)
    obj_opsrl_std = np.std(obj_opsrl, axis = 0)

    con_opsrl_mean = np.mean(con_opsrl, axis = 0)
    con_opsrl_std = np.std(con_opsrl, axis = 0)

    data = {}
    data['obj_opsrl_mean'] = obj_opsrl_mean
    data['obj_opsrl_std'] = obj_opsrl_std
    data['con_opsrl_mean'] = con_opsrl_mean
    data['con_opsrl_std'] = con_opsrl_std

    return data, label



NUMBER_SIMULATIONS = 1 
NUMBER_EPISODES_o = 30000

# take the second input argument as the number of episodes
if len(sys.argv) > 1:
    NUMBER_EPISODES_o = int(sys.argv[1]) 

L = 1 # marker point interval
mark_every_interval = 2000 # marker point interval


# scenario 1: use init-state dependent Constraints

# fn_list = ['../Contextual/output_final_initstate_dependentC/CONTEXTUAL_opsrl100.pkl',
#            'output_final_initstate_dependentC/DOPE_opsrl100.pkl',
#            'output_final_initstate_dependentC/OptPessLP_opsrl100.pkl', 
#            'output_final_initstate_dependentC/OptCMDP_opsrl100.pkl']


# scenario 2: DO NOT use init-state independent Constraints, this is what we use for the paper

fn_list = ['../Contextual/output_final/CONTEXTUAL_opsrl100.pkl',
           'output_final/DOPE_opsrl16.pkl',
           'output_final/OptPessLP_opsrl16.pkl', 
           'output_final/OptCMDP_opsrl16.pkl']


data_list = []
label_list = ['COPS-BG', 'DOPE', 'OptPessLP', 'OptCMDP']
color_list = ['red', 'blue', 'green', 'SaddleBrown']
marker_list = ['D', 'o', 's', 'v']
for fn in fn_list:
    data, _ = read_data(fn, NUMBER_SIMULATIONS, NUMBER_EPISODES_o)

    data_list.append(data)

print('label_list =', label_list)


# ----------------- Plot the results ----------------- #

x_o =  np.arange(0, NUMBER_EPISODES_o, L)

# plot the following 2 figures on the same plot using subplot
# objective regret
# constraint violation

import matplotlib.pyplot as plt

# create figure and subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 4))

# set facecolor and alpha for all subplots
for ax in axs:
    ax.patch.set_facecolor("lightsteelblue")
    ax.patch.set_alpha(0.4)

# plot the first subplot
for data, label, clr, mkr in zip(data_list, label_list, color_list, marker_list):
    obj_opsrl_mean = data['obj_opsrl_mean']
    con_opsrl_mean = data['con_opsrl_mean']
    axs[0].plot(x_o, obj_opsrl_mean[::L], label = label, color=clr, alpha=0.6, linewidth=2.5, marker=mkr,markersize='5', markeredgewidth='3',markevery=mark_every_interval)
    axs[1].plot(x_o, con_opsrl_mean[::L], label = label, color=clr, alpha=0.6, linewidth=2.5, marker=mkr,markersize='5', markeredgewidth='3',markevery=mark_every_interval)

axs[0].grid()
axs[0].ticklabel_format(style='sci', axis='both', scilimits=(0,0))
axs[0].legend(loc = 'upper left', prop={'size': 13})
axs[0].set_xlabel('Episode')
axs[0].set_ylabel('Objective Regret')
#axs[0].set_ylim([-0.1e3, 5e3])

# plot the second subplot
# axs[1].plot(x_o, con_opsrl_mean[::L], color='saddlebrown',label = label, alpha=0.6,linewidth=2.5, marker="D",markersize='8', markeredgewidth='3',markevery=mark_every_interval)
axs[1].grid()
axs[1].ticklabel_format(style='sci', axis='both', scilimits=(0,0))
axs[1].legend(loc = 'center right',prop={'size': 13})
axs[1].set_ylim([-0.1e3, 1e4])
axs[1].set_xlabel('Episode')
axs[1].set_ylabel('Constraint Regret')

# adjust layout and save the figure
plt.tight_layout()
plt.savefig("BGClass_regrets2.png", dpi=300, facecolor='w', edgecolor='w')
plt.savefig("../../NumericalResults/plots/png/BGClass_regrets.png", dpi=300, facecolor='w', edgecolor='w')
plt.savefig("../../NumericalResults/plots/pdf/BGClass_regrets.pdf", dpi=300, facecolor='w', edgecolor='w') 
plt.show()