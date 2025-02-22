#Imports
import numpy as np
import pandas as pd
from UtilityMethods import utils
import matplotlib.pyplot as plt
import time
import os
import math
import pickle
import sys
import random
from tqdm import tqdm


# control parameters
NUMBER_EPISODES = 3e4
RUN_NUMBER = 100 #Change this field to set the seed for the experiment.
use_gurobi = False

if len(sys.argv) > 1:
    use_gurobi = sys.argv[1]

NUMBER_SIMULATIONS = 1
random.seed(int(RUN_NUMBER))
np.random.seed(int(RUN_NUMBER))

# remove the filename = 'output/opsrl' + str(RUN_NUMBER) + '.pkl' to avoid reading old data
old_filename = 'output/OptCMDP_opsrl' + str(RUN_NUMBER) + '.pkl'
if os.path.exists(old_filename):
    os.remove(old_filename)
    print("Removed old file: ", old_filename)


# Initialize:
with open('output/model.pkl', 'rb') as f:
    [P, R, C1, C2, INIT_STATE_INDEX, INIT_STATES_LIST, state_code_to_index, CONSTRAINT1_list, C1_b_list, CONSTRAINT2_list, C2_b_list,
     N_STATES, N_ACTIONS, ACTIONS_PER_STATE, EPISODE_LENGTH, DELTA] = pickle.load(f)

with  open('output/solution.pkl', 'rb') as f:
    [opt_policy_con_list, opt_value_LP_con_list, opt_cost1_LP_con_list, opt_cost2_LP_con_list, opt_q_con_list] = pickle.load(f) 

with open('output/base.pkl', 'rb') as f:
    [pi_b_list, val_b_list, cost1_b_list, cost2_b_list, q_b_list] = pickle.load(f)

EPS = 1 # not used
M = 1024* N_STATES*EPISODE_LENGTH**2/EPS**2 # not used

# CONSTRAINT = RUN_NUMBER # +++++

CONSTRAINT = CONSTRAINT1_list[-1]
C_b = C1_b_list[-1]


Cb = C_b
print("CONSTRAINT =", CONSTRAINT)
print("Cb =", Cb)
print("CONSTRAINT - Cb =", CONSTRAINT - Cb)
print("N_STATES =", N_STATES)
print("N_ACTIONS =", N_ACTIONS)

NUMBER_EPISODES = int(NUMBER_EPISODES)
NUMBER_SIMULATIONS = int(NUMBER_SIMULATIONS)
ACTIONS = np.arange(N_ACTIONS)

ObjRegret2 = np.zeros((NUMBER_SIMULATIONS, NUMBER_EPISODES))
Con1Regret2 = np.zeros((NUMBER_SIMULATIONS, NUMBER_EPISODES))
Con2Regret2 = np.zeros((NUMBER_SIMULATIONS, NUMBER_EPISODES))
NUMBER_INFEASIBILITIES = np.zeros((NUMBER_SIMULATIONS, NUMBER_EPISODES))

for sim in range(NUMBER_SIMULATIONS):

    util_methods = utils(EPS, DELTA, M, P, R, C1, C2, INIT_STATE_INDEX, 
                         EPISODE_LENGTH, N_STATES, N_ACTIONS, ACTIONS_PER_STATE, CONSTRAINT, Cb, CONSTRAINT, Cb, use_gurobi, "OptCMDP") # set the utility methods for each run

    ep_count = np.zeros((N_STATES, N_ACTIONS)) # initialize the counter for each run
    ep_count_p = np.zeros((N_STATES, N_ACTIONS, N_STATES))
    ep_emp_reward = {} # initialize the empirical rewards and costs for each run
    ep_emp_cost1 = {}
    ep_emp_cost2 = {}
    for s in range(N_STATES):
        ep_emp_reward[s] = {}
        ep_emp_cost1[s] = {}
        ep_emp_cost2[s] = {}
        for a in range(N_ACTIONS):
            ep_emp_reward[s][a] = 0
            ep_emp_cost1[s][a] = 0
            ep_emp_cost2[s][a] = 0

    objs = [] # objective regret for current run
    cons1 = []
    cons2 = []
    max_cost1 = 0
    max_cost2 = 0
    select_baseline_policy_ct = 0 
    for episode in range(NUMBER_EPISODES):

        # sample a initial state s uniformly from the list of initial states INIT_STATES_LIST
        # INIT_STATES_LIST = ['1', '2']
        s_code = np.random.choice(INIT_STATES_LIST, 1, replace = True)[0]
        s_idx_init = state_code_to_index[s_code]
        # s_idx_init = 0 # +++++
        util_methods.update_mu(s_idx_init)    

        # set opt_value_LP_con corresponding to the initial state
        opt_value_LP_con = opt_value_LP_con_list[s_idx_init]

        CONSTRAINT1 = CONSTRAINT1_list[s_idx_init]
        CONSTRAINT2 = CONSTRAINT2_list[s_idx_init]

        C1b = C1_b_list[s_idx_init]
        C2b = C2_b_list[s_idx_init]
        util_methods.setConstraint(CONSTRAINT1, CONSTRAINT2)
        util_methods.setCb(C1b, C2b)        

        util_methods.setCounts(ep_count_p, ep_count)
        util_methods.update_empirical_model(0) 
        util_methods.update_empirical_rewards_costs(ep_emp_reward, ep_emp_cost1, ep_emp_cost2)
        util_methods.compute_confidence_intervals_OptCMDP()

        t1 = time.time()
        pi_k, val_k, cost1_k, cost2_k, log, q_k = util_methods.compute_extended_LP() # +++++
        t2 = time.time()
        dtime = t2 - t1
        #print("Time to solve the extended LP:", dtime)
        
        if log != 'Optimal':  #Added this part to resolve issues about infeasibility. Because I am not sure about the value of K0, this condition would take care of that

            # print("+++++Infeasible solution in Extended LP, select the baseline policy instead")
            # pi_k = pi_b
            # val_k = val_b
            # cost_k = cost_b
            # q_k = q_b
            # select_baseline_policy_ct += 1

            print("+++++Infeasible solution in Extended LP, select the random policy instead")
            pi_k, val_k, cost1_k, cost2_k, log, q_k = util_methods.compute_extended_LP_random() # use uniform probability to select the action
            select_baseline_policy_ct += 1                    
        
        max_cost1 = max(max_cost1, cost1_k[s_idx_init, 0])
        max_cost2 = max(max_cost2, cost2_k[s_idx_init, 0])
        print('s_idx_init={}, cost1_k[s_idx_init, 0]={:.2f}, CONS1={:.2f}, max_cost1={:2f}, cost2_k[s_idx_init, 0]={:.2f}, CONS2={:.2f}, max_cost2={:.2f},'.format(
               s_idx_init, cost1_k[s_idx_init, 0], CONSTRAINT1, max_cost1, cost2_k[s_idx_init, 0], CONSTRAINT2, max_cost2)) 

        if episode == 0:
            ObjRegret2[sim, episode] = abs(val_k[s_idx_init, 0] - opt_value_LP_con[s_idx_init, 0]) # for episode 0, calculate the objective regret, we care about the value of a policy at the initial state
            Con1Regret2[sim, episode] = max(0, cost1_k[s_idx_init, 0] - CONSTRAINT1) # constraint regret, we care about the cumulative cost of a policy at the initial state
            Con2Regret2[sim, episode] = max(0, cost2_k[s_idx_init, 0] - CONSTRAINT2) # constraint regret, we care about the cumulative cost of a policy at the initial state
            objs.append(ObjRegret2[sim, episode])
            cons1.append(Con1Regret2[sim, episode])
            cons2.append(Con2Regret2[sim, episode])
            if cost1_k[s_idx_init, 0] > CONSTRAINT1 or cost2_k[s_idx_init, 0] > CONSTRAINT2:
                NUMBER_INFEASIBILITIES[sim, episode] = 1
        else:
            ObjRegret2[sim, episode] = ObjRegret2[sim, episode - 1] + abs(val_k[s_idx_init, 0] - opt_value_LP_con[s_idx_init, 0]) # calculate the objective regret, note this is cumulative sum upto k episode, beginninng of page 8 in the paper
            Con1Regret2[sim, episode] = Con1Regret2[sim, episode - 1] + max(0, cost1_k[s_idx_init, 0] - CONSTRAINT1) # cumulative sum of constraint regret
            Con2Regret2[sim, episode] = Con2Regret2[sim, episode - 1] + max(0, cost2_k[s_idx_init, 0] - CONSTRAINT2) # cumulative sum of constraint regret
            objs.append(ObjRegret2[sim, episode])
            cons1.append(Con1Regret2[sim, episode])
            cons2.append(Con2Regret2[sim, episode])
            if cost1_k[s_idx_init, 0] > CONSTRAINT1 or cost2_k[s_idx_init, 0] > CONSTRAINT2:
                NUMBER_INFEASIBILITIES[sim, episode] = NUMBER_INFEASIBILITIES[sim, episode - 1] + 1 # count the number of infeasibilities until k episode
            else:
                NUMBER_INFEASIBILITIES[sim, episode] = NUMBER_INFEASIBILITIES[sim, episode - 1]

        print('Episode {}, ObjRegt = {:.2f}, Cons1Regt = {:.2f}, Cons2Regt = {:.2f}, #Infeas = {}, #Select_random_policy = {}, Time = {:.2f}'.format(
              episode, ObjRegret2[sim, episode], Con1Regret2[sim, episode], Con2Regret2[sim, episode], NUMBER_INFEASIBILITIES[sim, episode], select_baseline_policy_ct, dtime))

        # reset the counters
        ep_count = np.zeros((N_STATES, N_ACTIONS))
        ep_count_p = np.zeros((N_STATES, N_ACTIONS, N_STATES))
        for s in range(N_STATES):
            ep_emp_reward[s] = {}
            ep_emp_cost1[s] = {}
            ep_emp_cost2[s] = {}
            for a in range(N_ACTIONS):
                ep_emp_reward[s][a] = 0
                ep_emp_cost1[s][a] = 0                
                ep_emp_cost2[s][a] = 0

        s = s_idx_init
        for h in range(EPISODE_LENGTH): # for each step in current episode
            prob = pi_k[s, h, :]
            
            a = int(np.random.choice(ACTIONS, 1, replace = True, p = prob)) # select action based on the policy/probability
            next_state, rew, cost1, cost2 = util_methods.step(s, a, h) # take the action and get the next state, reward and cost
            ep_count[s, a] += 1 # update the counter
            ep_count_p[s, a, next_state] += 1
            ep_emp_reward[s][a] += rew
            ep_emp_cost1[s][a] += cost1 # this is the SBP feedback
            ep_emp_cost2[s][a] += cost2 # this is the SBP feedback
            s = next_state

        # dump results out every xxx episodes
        if episode != 0 and episode%1000== 0:

            filename = 'output/OptCMDP_opsrl' + str(RUN_NUMBER) + '.pkl'
            f = open(filename, 'ab')
            pickle.dump([NUMBER_SIMULATIONS, NUMBER_EPISODES, objs , cons1, cons2, pi_k, NUMBER_INFEASIBILITIES, q_k], f)
            f.close()
            objs = []
            cons1 = []
            cons2 = []

        elif episode == NUMBER_EPISODES-1: # dump results out at the end of the last episode
            filename = 'output/OptCMDP_opsrl' + str(RUN_NUMBER) + '.pkl'
            f = open(filename, 'ab')
            pickle.dump([NUMBER_SIMULATIONS, NUMBER_EPISODES, objs , cons1, cons2, pi_k, NUMBER_INFEASIBILITIES, q_k], f)
            f.close()
        
# take average/std over multiple simulation runs
ObjRegret_mean = np.mean(ObjRegret2, axis = 0) 
Con1Regret_mean = np.mean(Con1Regret2, axis = 0)
Con2Regret_mean = np.mean(Con2Regret2, axis = 0)
ObjRegret_std = np.std(ObjRegret2, axis = 0)
Con1Regret_std = np.std(Con1Regret2, axis = 0)
Con2Regret_std = np.std(Con2Regret2, axis = 0)

# save the results as a pickle file
filename = 'output/OptCMDP_regrets_' + str(RUN_NUMBER) + '.pkl'
with open(filename, 'wb') as f:
    pickle.dump([NUMBER_SIMULATIONS, NUMBER_EPISODES, ObjRegret_mean, ObjRegret_std, Con1Regret_mean, Con1Regret_std, Con2Regret_mean, Con2Regret_std], f)