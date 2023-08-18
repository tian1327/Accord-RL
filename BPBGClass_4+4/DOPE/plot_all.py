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
    con1_opsrl = np.zeros((NUMBER_SIMULATIONS, NUMBER_EPISODES_o))
    con2_opsrl = np.zeros((NUMBER_SIMULATIONS, NUMBER_EPISODES_o))

    for i in range(NUMBER_SIMULATIONS):
        
        filename = fn
        f = open(filename, 'rb')
        objs = []
        cons1 = []
        cons2 = []

        j = 0
        while 1:
            try:
                j += 1
                if label == 'CONTEXTUAL':
                    [R_est_err, C1_est_err,  C2_est_err, min_eign_sbp_list, min_eign_hba1c_list, min_eign_cvd_list, NUMBER_SIMULATIONS, NUMBER_EPISODES, ObjRegret, Con1Regret, Con2Regret, pi_k, NUMBER_INFEASIBILITIES, q_k] = pickle.load(f) # load results chunk by chunk
                else:
                    [NUMBER_SIMULATIONS, NUMBER_EPISODES, ObjRegret, Con1Regret, Con2Regret, pi_k, NUMBER_INFEASIBILITIES, q_k] = pickle.load(f) # load results chunk by chunk
                objs.append(ObjRegret)
                cons1.append(Con1Regret)
                cons2.append(Con2Regret)

            except EOFError:
                break
        f.close()

        flat_listobj = [item for sublist in objs for item in sublist] # flatten the list
        flat_listcon1 = [item for sublist in cons1 for item in sublist]
        flat_listcon2 = [item for sublist in cons2 for item in sublist]
        
        #print(len(flat_listobj))
        obj_opsrl[i, :] = np.copy(flat_listobj[0:NUMBER_EPISODES_o])
        con1_opsrl[i, :] = np.copy(flat_listcon1[0:NUMBER_EPISODES_o])
        con2_opsrl[i, :] = np.copy(flat_listcon2[0:NUMBER_EPISODES_o])
    
    obj_opsrl_mean = np.mean(obj_opsrl, axis = 0)
    obj_opsrl_std = np.std(obj_opsrl, axis = 0)

    con1_opsrl_mean = np.mean(con1_opsrl, axis = 0)
    con1_opsrl_std = np.std(con1_opsrl, axis = 0)

    con2_opsrl_mean = np.mean(con2_opsrl, axis = 0)
    con2_opsrl_std = np.std(con2_opsrl, axis = 0)

    data = {}
    data['obj_opsrl_mean'] = obj_opsrl_mean
    data['obj_opsrl_std'] = obj_opsrl_std
    data['con1_opsrl_mean'] = con1_opsrl_mean
    data['con1_opsrl_std'] = con1_opsrl_std
    data['con2_opsrl_mean'] = con2_opsrl_mean
    data['con2_opsrl_std'] = con2_opsrl_std


    return data, label



NUMBER_SIMULATIONS = 1 
NUMBER_EPISODES_o = 30000

# take the second input argument as the number of episodes
if len(sys.argv) > 1:
    NUMBER_EPISODES_o = int(sys.argv[1])

L = 1 # marker point interval
mark_every_interval = 2000 # marker point interval


fn_list = [
           '../Contextual/output_final/CONTEXTUAL_opsrl100.pkl',
           #'output/DOPE_opsrl100.pkl', # this RUN# 100 has 39 Cons1Regret, thus not used
           #'output/DOPE_opsrl200.pkl',
           'output_final/DOPE_opsrl200.pkl', # good DOPE, increased K0 to 1000, no Cons1Regret
           'output_final/OptPessLP_opsrl100.pkl', 
           'output_final/OptCMDP_opsrl100.pkl']

data_list = []
label_list = ['COPS-MM', 'DOPE', 'OptPessLP', 'OptCMDP']
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
# for ax in axs:
#     ax.patch.set_facecolor("lightsteelblue")
#     ax.patch.set_alpha(0.4)

# plot the first subplot
for data, label, clr, mkr in zip(data_list, label_list, color_list, marker_list):
    obj_opsrl_mean = data['obj_opsrl_mean']
    con1_opsrl_mean = data['con1_opsrl_mean']
    con2_opsrl_mean = data['con2_opsrl_mean']
    axs[0].plot(x_o, obj_opsrl_mean[::L], label = label, color=clr, alpha=0.6, linewidth=2.5, marker=mkr,markersize='5', markeredgewidth='3',markevery=mark_every_interval)
    # axs[1].plot(x_o, con1_opsrl_mean[::L], label = label+"_SBP", color=clr, alpha=0.6, linestyle="dotted", linewidth=2.5, marker=mkr, markersize='5', markeredgewidth='3',markevery=mark_every_interval)
    # axs[1].plot(x_o, con2_opsrl_mean[::L], label = label+'_HBA1C', color=clr, alpha=0.6, linewidth=2.5, marker=mkr, markersize='5', markeredgewidth='3',markevery=mark_every_interval)
    axs[1].plot(x_o, con1_opsrl_mean[::L], label = label, color=clr, alpha=0.6, linestyle="solid", linewidth=2.5, marker=mkr, markersize='5', markeredgewidth='3',markevery=mark_every_interval)
    

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
#axs[1].set_ylim([-0.1e3, 1e4])
axs[1].set_xlabel('Episode')
axs[1].set_ylabel('Constraint Regret')

# adjust layout and save the figure
plt.tight_layout()
plt.savefig("BPBGClass_regrets.png", dpi=300, facecolor='w', edgecolor='w')
plt.savefig("../../NumericalResults/plots/png/BPBGClass_regrets.png", dpi=300, facecolor='w', edgecolor='w')
plt.savefig("../../NumericalResults/plots/pdf/BPBGClass_regrets.pdf", dpi=300, facecolor='w', edgecolor='w') 

plt.show()