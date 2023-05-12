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
    NUMBER_EPISODES_o = int(sys.argv[2]) + 1

L = 1 # marker point interval
mark_every_interval = 2000 # marker point interval

# ----------------- Read data from file ----------------- #
obj_opsrl = np.zeros((NUMBER_SIMULATIONS, NUMBER_EPISODES_o))
con_opsrl = np.zeros((NUMBER_SIMULATIONS, NUMBER_EPISODES_o))
R_err_opsrl = np.zeros((NUMBER_SIMULATIONS, NUMBER_EPISODES_o))
C_err_opsrl = np.zeros((NUMBER_SIMULATIONS, NUMBER_EPISODES_o))
min_eign_cvd = np.zeros((NUMBER_SIMULATIONS, NUMBER_EPISODES_o))
min_eign_sbp = np.zeros((NUMBER_SIMULATIONS, NUMBER_EPISODES_o))


for i in range(NUMBER_SIMULATIONS):
    
    filename = fn
    f = open(filename, 'rb')
    objs = []
    cons = []
    R_err = []
    C_err = []
    eigen_cvd = []
    eigen_sbp = []
    j = 0
    while 1:
        try:
            j += 1
            [R_est_err, C_est_err, min_eign_sbp_list, min_eign_cvd_list, NUMBER_SIMULATIONS, NUMBER_EPISODES, ObjRegret, ConRegret, pi_k, NUMBER_INFEASIBILITIES, q_k] = pickle.load(f) # load results chunk by chunk
            objs.append(ObjRegret)
            cons.append(ConRegret)
            R_err.append(R_est_err)
            C_err.append(C_est_err)
            eigen_cvd.append(min_eign_cvd_list)
            eigen_sbp.append(min_eign_sbp_list)

        except EOFError:
            break
    f.close()

    flat_listobj = [item for sublist in objs for item in sublist] # flatten the list
    flat_listcon = [item for sublist in cons for item in sublist]
    flat_R_err = [item for sublist in R_err for item in sublist]
    flat_C_err = [item for sublist in C_err for item in sublist]
    flat_eigen_cvd = [item for sublist in eigen_cvd for item in sublist]
    flat_eigen_sbp = [item for sublist in eigen_sbp for item in sublist]
    
    #print(len(flat_listobj))
    obj_opsrl[i, :] = np.copy(flat_listobj[0:NUMBER_EPISODES_o])
    con_opsrl[i, :] = np.copy(flat_listcon[0:NUMBER_EPISODES_o])
    R_err_opsrl[i, :] = np.copy(flat_R_err[0:NUMBER_EPISODES_o])
    C_err_opsrl[i, :] = np.copy(flat_C_err[0:NUMBER_EPISODES_o])
    min_eign_cvd[i, :] = np.copy(flat_eigen_cvd[0:NUMBER_EPISODES_o])
    min_eign_sbp[i, :] = np.copy(flat_eigen_sbp[0:NUMBER_EPISODES_o])

obj_opsrl_mean = np.mean(obj_opsrl, axis = 0)
obj_opsrl_std = np.std(obj_opsrl, axis = 0)

con_opsrl_mean = np.mean(con_opsrl, axis = 0)
con_opsrl_std = np.std(con_opsrl, axis = 0)

R_err_opsrl_mean = np.mean(R_err_opsrl, axis = 0)
R_err_opsrl_std = np.std(R_err_opsrl, axis = 0)

C_err_opsrl_mean = np.mean(C_err_opsrl, axis = 0)
C_err_opsrl_std = np.std(C_err_opsrl, axis = 0)

min_eign_cvd_mean = np.mean(min_eign_cvd, axis = 0)
min_eign_cvd_std = np.std(min_eign_cvd, axis = 0)

min_eign_sbp_mean = np.mean(min_eign_sbp, axis = 0)
min_eign_sbp_std = np.std(min_eign_sbp, axis = 0)

x_o =  np.arange(0, NUMBER_EPISODES_o, L)

# ----------------- Plot the results ----------------- #

# plot the following 3 figures on the same plot using subplot
# minimum eigenvalue of CVD
# objective regret
# constraint violation
# estimation error

import matplotlib.pyplot as plt

# create figure and subplots
fig, axs = plt.subplots(1, 4, figsize=(18, 4))

# set facecolor and alpha for all subplots
for ax in axs:
    ax.patch.set_facecolor("lightsteelblue")
    ax.patch.set_alpha(0.4)
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))

# plot the first subplot
axs[0].plot(x_o, min_eign_cvd_mean[::L], color='red',label = 'CVD min eigen', alpha=0.6,linewidth=2.5, marker="D",markersize='3', markeredgewidth='3',markevery=mark_every_interval)
axs[0].plot(x_o, min_eign_sbp_mean[::L], color='blue',label = 'hba1c min eigen', alpha=0.6,linewidth=2.5, marker="D",markersize='3', markeredgewidth='3',markevery=mark_every_interval)
axs[0].grid()
axs[0].legend(loc = 'upper left',prop={'size': 13})
axs[0].set_xlabel('Episode')
axs[0].set_ylabel('min eigenvalue')

# plot the second subplot
axs[1].plot(x_o, obj_opsrl_mean[::L], label = 'Contextual', color='saddlebrown', alpha=0.6, linewidth=2.5, marker="D",markersize='5', markeredgewidth='3',markevery=mark_every_interval)
axs[1].grid()
axs[1].legend(loc = 'upper left', prop={'size': 13})
axs[1].set_xlabel('Episode')
axs[1].set_ylabel('Objective Regret')

# plot the third subplot
axs[2].plot(x_o, con_opsrl_mean[::L], color='saddlebrown',label = 'Contextual', alpha=0.6,linewidth=2.5, marker="D",markersize='8', markeredgewidth='3',markevery=mark_every_interval)
axs[2].grid()
axs[2].legend(loc = 'upper right',prop={'size': 13})
axs[2].set_ylim([-0.1e3, 5e3])
axs[2].set_xlabel('Episode')
axs[2].set_ylabel('Constraint Regret')

# plot the fourth subplot
axs[3].plot(x_o, R_err_opsrl_mean[::L], color='red',label = 'CVD Model', alpha=0.6,linewidth=2.5, marker="D",markersize='3', markeredgewidth='3',markevery=mark_every_interval)
axs[3].plot(x_o, C_err_opsrl_mean[::L], color='blue',label = 'hba1c Model', alpha=0.6,linewidth=2.5, marker="D",markersize='3', markeredgewidth='3',markevery=mark_every_interval)
axs[3].grid()
axs[3].legend(loc = 'upper right',prop={'size': 13})
axs[3].set_ylim([0, 60])
axs[3].set_xlabel('Episode')
axs[3].set_ylabel('L2-Norm Error')

# adjust layout and save the figure
plt.tight_layout()
plt.savefig("output/all_plots.pdf")
plt.show()