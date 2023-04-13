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
    NUMBER_EPISODES_o = int(sys.argv[1]) + 1

L = 10 # marker point interval


obj_opsrl = np.zeros((NUMBER_SIMULATIONS, NUMBER_EPISODES_o))
con_opsrl = np.zeros((NUMBER_SIMULATIONS, NUMBER_EPISODES_o))

for i in range(NUMBER_SIMULATIONS):
    
    filename = 'output/opsrl10.pkl'
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

# print('obj_opsrl = ', obj_opsrl[-1])
# print("Objective Regret: ", obj_opsrl_mean[-1])

plt.rcParams.update({'font.size': 16})
ax = plt.gca()
ax.patch.set_facecolor("lightsteelblue")
ax.patch.set_alpha(0.4)
plt.locator_params(axis='y', nbins=6)
plt.locator_params(axis='x', nbins=10)

plt.plot(x_o, obj_opsrl_mean[::L], label = 'DOPE', color='saddlebrown', alpha=0.6, linewidth=2.5, marker="D",markersize='5', markeredgewidth='3',markevery=60)
# plt.fill_between(x_o, obj_opsrl_mean[::L] - obj_opsrl_std[::L] ,obj_opsrl_mean[::L] + obj_opsrl_std[::L], alpha=0.2, linewidth=2.5, edgecolor='saddlebrown', facecolor='saddlebrown')
plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
plt.grid()
plt.legend(loc = 'upper left', prop={'size': 13})

# make x y axis square
# ax.set_aspect('equal', adjustable='box')

# set y axis limit
# ax.set_ylim([-0.1e3, 5e3])

plt.xlabel('Episode')
plt.ylabel('Objective Regret')
plt.tight_layout()
plt.savefig("output/Objective_Regret.pdf")
# plt.close()
plt.show()


# times = np.arange(1, NUMBER_EPISODES_o+1)
# squareroot = [int(b) / int(m) for b,m in zip(obj_opsrl_mean, np.sqrt(times))]
# plt.plot(range(NUMBER_EPISODES_o),squareroot)
# #plt.fill_between(range(NUMBER_EPISODES), ObjRegret_mean - ObjRegret_std, ObjRegret_mean + ObjRegret_std, alpha = 0.5)
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
# plt.grid()
# plt.xlabel('Episodes')
# plt.ylabel('Objective Regret square root curve')
# plt.tight_layout()
# plt.savefig("objectiveregretSQRTIn.pdf")
# plt.show()


ax = plt.gca()
ax.patch.set_facecolor("lightsteelblue")
ax.patch.set_alpha(0.4)

plt.plot(x_o, con_opsrl_mean[::L], color='saddlebrown',label = 'DOPE', alpha=0.6,linewidth=2.5, marker="D",markersize='8', markeredgewidth='3',markevery=60)
# plt.fill_between(x_o, con_opsrl_mean[::L] - con_opsrl_std[::L] ,con_opsrl_mean[::L] + con_opsrl_std[::L], alpha=0.2, linewidth=2.5, edgecolor='saddlebrown', facecolor='saddlebrown')

# ax.set_ylim([-0.1e3, 8.5e3])
plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
# set y axis limit
ax.set_ylim([-0.1e3, 5e3])
plt.grid()
plt.legend(loc = 'upper right',prop={'size': 13})
plt.xlabel('Episode')
plt.ylabel('Constraint Regret')
# make x y axis square
# ax.set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig("output/Constraint_Regret.pdf")
plt.show()