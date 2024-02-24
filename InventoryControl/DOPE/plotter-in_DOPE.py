import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import pickle
import time
from matplotlib.ticker import StrMethodFormatter


a = 0.8 # alpha

NUMBER_SIMULATIONS = 1 # 4

# NUMBER_EPISODES_o = 300000
NUMBER_EPISODES_o = int(1e4)
# NUMBER_EPISODES = 300000
NUMBER_EPISODES = int(1e4)

obj_opsrl = np.zeros((NUMBER_SIMULATIONS ,NUMBER_EPISODES_o))
con_opsrl = np.zeros((NUMBER_SIMULATIONS ,NUMBER_EPISODES_o))
# obj_efroni = np.zeros((NUMBER_SIMULATIONS ,NUMBER_EPISODES))
# con_efroni = np.zeros((NUMBER_SIMULATIONS ,NUMBER_EPISODES))
#obj_ucrl = np.zeros((NUMBER_SIMULATIONS ,NUMBER_EPISODES))
#con_ucrl = np.zeros((NUMBER_SIMULATIONS ,NUMBER_EPISODES))
# obj_Tao = np.zeros((NUMBER_SIMULATIONS ,NUMBER_EPISODES))
# con_Tao = np.zeros((NUMBER_SIMULATIONS ,NUMBER_EPISODES))


for i in range(NUMBER_SIMULATIONS):
    print(i)
    
    # filename = '../opsrl-in5' + str(i+1) +'.pckl'
    # filename = 'opsrl-seed100-in100.pckl'
    # filename = 'opsrl-ns7-in10.pckl'
    filename = 'opsrl-in10.pckl'
    
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
            # if j == 4:
            #     break
        except EOFError:
            break
    f.close()
    flat_listobj = [item for sublist in objs for item in sublist] # flatten the list
    flat_listcon = [item for sublist in cons for item in sublist]
    #print(len(flat_listobj))
    obj_opsrl[i, :] = np.copy(flat_listobj[0:NUMBER_EPISODES_o]) # take the first 300001 elements
    con_opsrl[i, :] = np.copy(flat_listcon[0:NUMBER_EPISODES_o])

    """
    filename = '../efroni-in' + str(i+1) +'.pckl'
    f = open(filename, 'rb')
    objs = []
    cons = []
    j = 0
    while 1:
        try:
            j += 1
            [NUMBER_SIMULATIONS, NUMBER_EPISODES, ObjRegret, ConRegret, pi_k, NUMBER_INFEASIBILITIES, q_k] = pickle.load(f)
            objs.append(ObjRegret)
            cons.append(ConRegret)
            # if j == 4:
            #     break
        except EOFError:
            break
    f.close()
    flat_listobj = [item for sublist in objs for item in sublist]
    flat_listcon = [item for sublist in cons for item in sublist]
    
    #print(len(flat_listobj))
    
    obj_efroni[i, :] = np.copy(flat_listobj[0:NUMBER_EPISODES_o])
    con_efroni[i, :] = np.copy(flat_listcon[0:NUMBER_EPISODES_o])
    """
    
    
    """
    filename = '../Tao-In' + str(i+1) +'.pckl'
    f = open(filename, 'rb')
    objs = []
    cons = []
    j = 0
    while 1:
        try:
            j += 1
           
            [NUMBER_SIMULATIONS, NUMBER_EPISODES, ObjRegret, ConRegret, pi_k, NUMBER_INFEASIBILITIES, q_k] = pickle.load(f)
            objs.append(ObjRegret)
            cons.append(ConRegret)
            # if j == 4:
            #     break
            
           
            
        except EOFError:
            break
    f.close()
   
    flat_listobj = [item for sublist in objs for item in sublist]
   
    flat_listcon = [item for sublist in cons for item in sublist]
    
    
    obj_Tao[i, :] = np.copy(flat_listobj[0:NUMBER_EPISODES_o])
    con_Tao[i, :] = np.copy(flat_listcon[0:NUMBER_EPISODES_o])
    """


obj_opsrl_mean = np.mean(obj_opsrl, axis = 0)
obj_opsrl_std = np.std(obj_opsrl, axis = 0)
# obj_efroni_mean = np.mean(obj_efroni, axis = 0)
# obj_efroni_std = np.std(obj_efroni, axis = 0)
#obj_ucrl_mean = np.mean(obj_ucrl, axis = 0)
#obj_ucrl_std = np.std(obj_ucrl, axis = 0)
# obj_Tao_mean = np.mean(obj_Tao, axis = 0)
# obj_Tao_std = np.std(obj_Tao, axis = 0)


con_opsrl_mean = np.mean(con_opsrl, axis = 0)
con_opsrl_std = np.std(con_opsrl, axis = 0)
# con_efroni_mean = np.mean(con_efroni, axis = 0)
# con_efroni_std = np.std(con_efroni, axis = 0)
#con_ucrl_mean = np.mean(con_ucrl, axis = 0)
#con_ucrl_std = np.std(con_ucrl, axis = 0)
# con_Tao_mean = np.mean(con_Tao, axis = 0)
# con_Tao_std = np.std(con_Tao, axis = 0)

# NUMBER_EPISODES_o = 300000
NUMBER_EPISODES_o = int(1e4)
# NUMBER_EPISODES = 300000
NUMBER_EPISODES = int(1e4)


L = 1000

x = np.arange(0,NUMBER_EPISODES,L)
x_o =  np.arange(0,NUMBER_EPISODES_o,L)

plt.rcParams.update({'font.size': 16})
ax = plt.gca()
ax.patch.set_facecolor("lightsteelblue")
ax.patch.set_alpha(0.4)
plt.locator_params(axis='y', nbins=6)
plt.locator_params(axis='x', nbins=10)


plt.plot(x_o, obj_opsrl_mean[::L], label = 'DOPE', color='saddlebrown', alpha=0.6,linewidth=2.5, marker="D",markersize='5', markeredgewidth='3',markevery=40)
plt.fill_between(x_o, obj_opsrl_mean[::L] - obj_opsrl_std[::L] ,obj_opsrl_mean[::L] + obj_opsrl_std[::L], alpha=0.2, linewidth=2.5, edgecolor='saddlebrown', facecolor='saddlebrown')

# plt.plot(x, obj_Tao_mean[::L], color='royalblue',alpha=0.6, label = 'OptPess-LP',linewidth=2.5,marker="o",markersize='5', markeredgewidth='4',markevery=40)
# plt.fill_between(x, obj_Tao_mean[::L] - obj_Tao_std[::L] ,obj_Tao_mean[::L] + obj_Tao_std[::L], alpha=0.2, linewidth=2.5, edgecolor='royalblue', facecolor='royalblue')#

# plt.plot(x, obj_efroni_mean[::L], label = 'OptCMDP', color='darkorange',alpha=1.0, linewidth=2.5,marker="x",markersize='8', markeredgewidth='2',markevery=40)
# plt.fill_between(x, obj_efroni_mean[::L] - obj_efroni_std[::L] ,obj_efroni_mean[::L] + obj_efroni_std[::L], alpha=1.0, linewidth=2.5, edgecolor='darkorange', facecolor='darkorange')

#plt.plot(x, obj_ucrl_mean[::L], color='forestgreen', alpha = 0.6, label = 'UCRL2',linewidth=2.5,marker="*",markersize='10', markeredgewidth='2',markevery=40)
#plt.fill_between(x, obj_ucrl_mean[::L] - obj_ucrl_std[::L] ,obj_ucrl_mean[::L] + obj_ucrl_std[::L], alpha=0.2, linewidth=2.5, edgecolor='forestgreen', facecolor='forestgreen')#

#plt.title('Queue Control Problem')
#plt.xticks(ticks=np.arange(0,NUMBER_EPISODES,50000))
#plt.yticks(ticks=np.arange(0,NUMBER_EPISODES,50000))
plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
plt.grid()
plt.legend(loc = 'upper left',prop={'size': 13})

# set y axis limit
ax.set_ylim([-0.1e6,1.55e6])

plt.xlabel('Episode')
plt.ylabel('Objective Regret')
plt.tight_layout()
plt.savefig("objectiveregretIn.png")
plt.show()


times = np.arange(1, NUMBER_EPISODES_o+1)
squareroot = [int(b) / int(m) for b,m in zip(obj_opsrl_mean, np.sqrt(times))]
plt.plot(range(NUMBER_EPISODES_o),squareroot)
#plt.fill_between(range(NUMBER_EPISODES), ObjRegret_mean - ObjRegret_std, ObjRegret_mean + ObjRegret_std, alpha = 0.5)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.grid()
plt.xlabel('Episodes')
plt.ylabel('Objective Regret square root curve')
plt.tight_layout()
plt.savefig("objectiveregretSQRTIn.png")
plt.show()


ax = plt.gca()
ax.patch.set_facecolor("lightsteelblue")
ax.patch.set_alpha(0.4)

plt.plot(x_o, con_opsrl_mean[::L], color='saddlebrown',label = 'DOPE', alpha=0.6,linewidth=2.5, marker="D",markersize='8', markeredgewidth='3',markevery=60)
plt.fill_between(x_o, con_opsrl_mean[::L] - con_opsrl_std[::L] ,con_opsrl_mean[::L] + con_opsrl_std[::L], alpha=0.2, linewidth=2.5, edgecolor='saddlebrown', facecolor='saddlebrown')

# plt.plot(x_o, con_Tao_mean[::L], color='royalblue', label = 'OptPess-LP',linewidth=2.5,marker="o",markersize='5', markeredgewidth='4',markevery=40)
# plt.fill_between(x_o, con_Tao_mean[::L] - con_Tao_std[::L] ,con_Tao_mean[::L] + con_Tao_std[::L],  alpha=0.2, linewidth=2.5, edgecolor='royalblue', facecolor='royalblue')#

# plt.plot(x_o, con_efroni_mean[::L], color='darkorange',label = 'OptCMDP',alpha=1.0, linewidth=2.5,marker="x",markersize='8', markeredgewidth='2',markevery=40)
# plt.fill_between(x_o, con_efroni_mean[::L] - con_efroni_std[::L] ,con_efroni_mean[::L] + con_efroni_std[::L], alpha=0.1, linewidth=3.5, edgecolor='darkorange', facecolor='darkorange')
#plt.plot(x_o, con_ucrl_mean, color='#3F7F4C', label = 'UCRL2')
#plt.fill_between(x_o, con_ucrl_mean - con_ucrl_std ,con_ucrl_mean + con_ucrl_std, alpha=a, edgecolor='#7EFF99', facecolor='#7EFF99', linewidth=0)#
#plt.title('Queue Control Problem')
ax.set_ylim([-0.1e3,8.5e3])
plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
plt.grid()
plt.legend(loc = 'center right',prop={'size': 13})
plt.xlabel('Episode')
plt.ylabel('Constraint Regret')
plt.tight_layout()
plt.savefig("constraintregretIn.png")
plt.show()

