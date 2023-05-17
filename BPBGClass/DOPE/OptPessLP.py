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
random.seed(RUN_NUMBER)
np.random.seed(RUN_NUMBER)

# remove the filename = 'output/opsrl' + str(RUN_NUMBER) + '.pkl' to avoid reading old data
old_filename = 'output/OptPessLP_opsrl' + str(RUN_NUMBER) + '.pkl'
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

CONSTRAINT = CONSTRAINT1_list[8]
C_b = C1_b_list[8]

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

Z = math.log(16 * N_STATES**2 * N_ACTIONS * EPISODE_LENGTH * NUMBER_EPISODES / DELTA) 
alpha = 1.0 + N_STATES*EPISODE_LENGTH + 4.0 * EPISODE_LENGTH * (1.0 + N_STATES*EPISODE_LENGTH) / (CONSTRAINT - Cb)
beta = 1.0 + N_STATES*EPISODE_LENGTH 
print("Z =", Z)
print("alpha =", alpha)
print("beta =", beta)


for sim in range(NUMBER_SIMULATIONS):

    util_methods = utils(EPS, DELTA, M, P, R, C1, C2, INIT_STATE_INDEX, 
                         EPISODE_LENGTH, N_STATES, N_ACTIONS, ACTIONS_PER_STATE, CONSTRAINT, Cb, CONSTRAINT, Cb, use_gurobi, "OptPessLP") # set the utility methods for each run

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

    objs = []
    cons1 = []
    cons2 = []
    select_baseline_policy_ct = 0 
    infeasible_elp_ct = 0
    for episode in range(NUMBER_EPISODES):

        util_methods.setCounts(ep_count_p, ep_count)
        util_methods.update_empirical_model(0) 
        util_methods.update_empirical_rewards_costs(ep_emp_reward, ep_emp_cost1, ep_emp_cost2)
        util_methods.compute_confidence_intervals_OptPessLP(Z, alpha, beta)
        util_methods.update_R_C_tao(alpha)

        # sample a initial state s uniformly from the list of initial states INIT_STATES_LIST
        s_code = np.random.choice(INIT_STATES_LIST, 1, replace = True)[0]
        s_idx_init = state_code_to_index[s_code]
        util_methods.update_mu(s_idx_init)

        # set corresponding base policy and optimal policy
        pi_b = pi_b_list[s_idx_init]
        val_b = val_b_list[s_idx_init]
        cost1_b = cost1_b_list[s_idx_init]
        cost2_b = cost2_b_list[s_idx_init]
        q_b = q_b_list[s_idx_init]

        opt_value_LP_con = opt_value_LP_con_list[s_idx_init]       

        CONSTRAINT1 = CONSTRAINT1_list[s_idx_init]
        CONSTRAINT2 = CONSTRAINT2_list[s_idx_init]
        C1b = C1_b_list[s_idx_init]
        C2b = C2_b_list[s_idx_init]
        util_methods.setConstraint(CONSTRAINT1, CONSTRAINT2)
        util_methods.setCb(C1b, C2b)             

        # evaluate the baseline policy under current estimated P_hat and R_Tao and C_Tao
        q_base, value_base, cost1_base, cost2_base = util_methods.FiniteHorizon_Policy_evaluation(util_methods.P_hat, pi_b, util_methods.R_Tao, util_methods.C1_Tao, util_methods.C2_Tao)

        if cost1_base[s_idx_init,0] >= (CONSTRAINT1 + C1b)/2 or cost2_base[s_idx_init,0] >= (CONSTRAINT2 + C2b)/2: # follow the baseline policy if the cost is too high
            pi_k = pi_b
            q_k = q_b
            val_k = val_b
            cost1_k = cost1_b
            cost2_k = cost2_b
            select_baseline_policy_ct += 1
            dtime = 0 
                                            
        else: # otherwise, solve the extended LP
            time1 = time.time()
            pi_k, val_k, cost1_k, cost2_k, log, q_k = util_methods.compute_extended_LP() # +++++
            dtime = time.time() - time1

            if log != 'Optimal':  #Added this part to resolve issues about infeasibility. Because I am not sure about the value of K0, this condition would take care of that

                print("+++++Infeasible solution in Extended LP, select the baseline policy instead")
                pi_k = pi_b
                val_k = val_b
                cost1_k = cost1_b
                cost2_k = cost2_b
                q_k = q_b
                infeasible_elp_ct += 1             
                
        print('s_idx_init =', s_idx_init)
        print('cost1_k[s_idx_init, 0] =', cost1_k[s_idx_init, 0])
        print('cost2_k[s_idx_init, 0] =', cost2_k[s_idx_init, 0])        
        if episode == 0:
            ObjRegret2[sim, episode] = abs(val_k[s_idx_init, 0] - opt_value_LP_con[s_idx_init, 0])
            Con1Regret2[sim, episode] = max(0, cost1_k[s_idx_init, 0] - CONSTRAINT1)
            Con2Regret2[sim, episode] = max(0, cost2_k[s_idx_init, 0] - CONSTRAINT2)
            objs.append(ObjRegret2[sim, episode])
            cons1.append(Con1Regret2[sim, episode])
            cons2.append(Con2Regret2[sim, episode])
            if cost1_k[s_idx_init, 0] > CONSTRAINT1 or cost2_k[s_idx_init, 0] > CONSTRAINT2:
                NUMBER_INFEASIBILITIES[sim, episode] = 1

        else:
            ObjRegret2[sim, episode] = ObjRegret2[sim, episode - 1] + abs(val_k[s_idx_init, 0] - opt_value_LP_con[s_idx_init, 0])
            Con1Regret2[sim, episode] = Con1Regret2[sim, episode - 1] + max(0, cost1_k[s_idx_init, 0] - CONSTRAINT1)
            Con2Regret2[sim, episode] = Con2Regret2[sim, episode - 1] + max(0, cost2_k[s_idx_init, 0] - CONSTRAINT2)
            objs.append(ObjRegret2[sim, episode])
            cons1.append(Con1Regret2[sim, episode])
            cons2.append(Con2Regret2[sim, episode])
            if cost1_k[s_idx_init, 0] > CONSTRAINT1 or cost2_k[s_idx_init, 0] > CONSTRAINT2:
                NUMBER_INFEASIBILITIES[sim, episode] = NUMBER_INFEASIBILITIES[sim, episode - 1] + 1
            else:
                NUMBER_INFEASIBILITIES[sim, episode] = NUMBER_INFEASIBILITIES[sim, episode - 1]       

        print('Episode {}, ObjRegt = {:.2f}, Cons1Regt = {:.2f}, Cons2Regt = {:.2f}, #Infeas = {}, #Select_baseline = {}, #Infeasible_ELP = {}, Time = {:.2f}'.format(
              episode, ObjRegret2[sim, episode], Con1Regret2[sim, episode], Con2Regret2[sim, episode], NUMBER_INFEASIBILITIES[sim, episode], select_baseline_policy_ct, infeasible_elp_ct, dtime))

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

            filename = 'output/OptPessLP_opsrl' + str(RUN_NUMBER) + '.pkl'
            f = open(filename, 'ab')
            pickle.dump([NUMBER_SIMULATIONS, NUMBER_EPISODES, objs , cons1, cons2, pi_k, NUMBER_INFEASIBILITIES, q_k], f)
            f.close()
            objs = []
            cons1 = []
            cons2 = []

        elif episode == NUMBER_EPISODES-1: # dump results out at the end of the last episode
            filename = 'output/OptPessLP_opsrl' + str(RUN_NUMBER) + '.pkl'
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
filename = 'output/OptPessLP_regrets_' + str(RUN_NUMBER) + '.pkl'
with open(filename, 'wb') as f:
    pickle.dump([NUMBER_SIMULATIONS, NUMBER_EPISODES, ObjRegret_mean, ObjRegret_std, Con1Regret_mean, Con1Regret_std, Con2Regret_mean, Con2Regret_std], f)