import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import pickle
import time
from matplotlib.ticker import StrMethodFormatter


NUMBER_SIMULATIONS = 1 
NUMBER_EPISODES_o = 5000

# take the second input argument as the number of episodes
if len(sys.argv) > 1:
    fn = sys.argv[1]
    NUMBER_EPISODES_o = int(sys.argv[2]) 

L = 1 # marker point interval
mark_every_interval = 2000 # marker point interval
label = fn.split('_')[0].split('/')[-1]

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

x_o =  np.arange(0, NUMBER_EPISODES_o, L)

# ----------------- Plot the results ----------------- #

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
axs[0].plot(x_o, obj_opsrl_mean[::L], label = label, color='saddlebrown', alpha=0.6, linewidth=2.5, marker="D",markersize='5', markeredgewidth='3',markevery=mark_every_interval)
axs[0].grid()
axs[0].ticklabel_format(style='sci', axis='both', scilimits=(0,0))
axs[0].legend(loc = 'lower right', prop={'size': 13})
axs[0].set_xlabel('Episode')
axs[0].set_ylabel('Objective Regret')

# plot the second subplot
axs[1].plot(x_o, con_opsrl_mean[::L], color='saddlebrown',label = label, alpha=0.6,linewidth=2.5, marker="D",markersize='8', markeredgewidth='3',markevery=mark_every_interval)
axs[1].grid()
axs[1].ticklabel_format(style='sci', axis='both', scilimits=(0,0))
axs[1].legend(loc = 'lower right',prop={'size': 13})
axs[1].set_ylim([-0.1e3, 5e3])
axs[1].set_xlabel('Episode')
axs[1].set_ylabel('Constraint Regret')

# adjust layout and save the figure
plt.tight_layout()
plt.savefig("output/all_plots.pdf")
plt.show()