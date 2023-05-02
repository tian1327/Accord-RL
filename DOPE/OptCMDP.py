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

start_time = time.time()

# control parameters
NUMBER_EPISODES = 1e6
RUN_NUMBER = 100 #Change this field to set the seed for the experiment.
use_gurobi = True



NUMBER_SIMULATIONS = 1
random.seed(RUN_NUMBER)
np.random.seed(RUN_NUMBER)

# remove the filename = 'output/opsrl' + str(RUN_NUMBER) + '.pkl' to avoid reading old data
old_filename = 'output/OptCMDP_opsrl' + str(RUN_NUMBER) + '.pkl'
if os.path.exists(old_filename):
    os.remove(old_filename)
    print("Removed old file: ", old_filename)


# Initialize:
with open('output/model.pkl', 'rb') as f:
    [P, R, C, INIT_STATE_INDEX, INIT_STATES_LIST, state_code_to_index, CONSTRAINT, C_b,
     N_STATES, N_ACTIONS, ACTIONS_PER_STATE, EPISODE_LENGTH, DELTA] = pickle.load(f)

with  open('output/solution.pkl', 'rb') as f:
    [opt_policy_con, opt_value_LP_con, opt_cost_LP_con, opt_q_con] = pickle.load(f) 

with open('output/base.pkl', 'rb') as f:
    [pi_b, val_b, cost_b, q_b] = pickle.load(f)

EPS = 1 # not used
M = 1024* N_STATES*EPISODE_LENGTH**2/EPS**2 # not used

# CONSTRAINT = 100

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
ConRegret2 = np.zeros((NUMBER_SIMULATIONS, NUMBER_EPISODES))
NUMBER_INFEASIBILITIES = np.zeros((NUMBER_SIMULATIONS, NUMBER_EPISODES))

L = math.log(2 * N_STATES * N_ACTIONS * EPISODE_LENGTH * NUMBER_EPISODES / DELTA) # for transition probabilities P_hat
L_prime = 2 * math.log(6 * N_STATES* N_ACTIONS * EPISODE_LENGTH * NUMBER_EPISODES / DELTA) # for SBP, CVDRisk
# page 11 in word document, to calculated the confidence intervals for the transition probabilities beta
print("L =", L)
print("L_prime =", L_prime)

for sim in range(NUMBER_SIMULATIONS):

    util_methods = utils(EPS, DELTA, M, P, R, C, INIT_STATE_INDEX, 
                         EPISODE_LENGTH, N_STATES, N_ACTIONS, ACTIONS_PER_STATE, CONSTRAINT, Cb, use_gurobi, "OptCMDP") # set the utility methods for each run

    ep_count = np.zeros((N_STATES, N_ACTIONS)) # initialize the counter for each run
    ep_count_p = np.zeros((N_STATES, N_ACTIONS, N_STATES))
    ep_emp_reward = {} # initialize the empirical rewards and costs for each run
    ep_emp_cost = {}
    for s in range(N_STATES):
        ep_emp_reward[s] = {}
        ep_emp_cost[s] = {}
        for a in range(N_ACTIONS):
            ep_emp_reward[s][a] = 0
            ep_emp_cost[s][a] = 0

    objs = [] # objective regret for current run
    cons = []
    select_baseline_policy_ct = 0 
    for episode in range(NUMBER_EPISODES):

        util_methods.setCounts(ep_count_p, ep_count)
        util_methods.update_empirical_model(0) 
        util_methods.update_empirical_rewards_costs(ep_emp_reward, ep_emp_cost)
        util_methods.compute_confidence_intervals_OptCMDP(L, L_prime, 1)

        t1 = time.time()
        pi_k, val_k, cost_k, log, q_k = util_methods.compute_extended_LP(0, Cb) 
        t2 = time.time()
        dtime = t2 - t1
        print("Time to solve the extended LP:", dtime)
        
        if log != 'Optimal':  #Added this part to resolve issues about infeasibility. Because I am not sure about the value of K0, this condition would take care of that

            # print("+++++Infeasible solution in Extended LP, select the baseline policy instead")
            # pi_k = pi_b
            # val_k = val_b
            # cost_k = cost_b
            # q_k = q_b
            # select_baseline_policy_ct += 1

            print("+++++Infeasible solution in Extended LP, select the random policy instead")
            pi_k, val_k, cost_k, log, q_k = util_methods.compute_extended_LP_random() # use uniform probability to select the action
            select_baseline_policy_ct += 1        
        
        
        # sample a initial state s uniformly from the list of initial states INIT_STATES_LIST
        s_code = np.random.choice(INIT_STATES_LIST, 1, replace = True)[0]
        s_idx_init = state_code_to_index[s_code]
        util_methods.update_mu(s_idx_init)        
        
        if episode == 0:
            ObjRegret2[sim, episode] = abs(val_k[s_idx_init, 0] - opt_value_LP_con[s_idx_init, 0]) # for episode 0, calculate the objective regret, we care about the value of a policy at the initial state
            ConRegret2[sim, episode] = max(0, cost_k[s_idx_init, 0] - CONSTRAINT) # constraint regret, we care about the cumulative cost of a policy at the initial state
            objs.append(ObjRegret2[sim, episode])
            cons.append(ConRegret2[sim, episode])
            if cost_k[s_idx_init, 0] > CONSTRAINT:
                NUMBER_INFEASIBILITIES[sim, episode] = 1
        else:
            ObjRegret2[sim, episode] = ObjRegret2[sim, episode - 1] + abs(val_k[s_idx_init, 0] - opt_value_LP_con[s_idx_init, 0]) # calculate the objective regret, note this is cumulative sum upto k episode, beginninng of page 8 in the paper
            ConRegret2[sim, episode] = ConRegret2[sim, episode - 1] + max(0, cost_k[s_idx_init, 0] - CONSTRAINT) # cumulative sum of constraint regret
            objs.append(ObjRegret2[sim, episode])
            cons.append(ConRegret2[sim, episode])
            if cost_k[s_idx_init, 0] > CONSTRAINT:
                NUMBER_INFEASIBILITIES[sim, episode] = NUMBER_INFEASIBILITIES[sim, episode - 1] + 1 # count the number of infeasibilities until k episode

        print('Episode {}, ObjRegt = {:.2f}, ConsRegt = {:.2f}, #Infeas = {}, #Select_baseline = {}, Time = {:.2f}'.format(
              episode, ObjRegret2[sim, episode], ConRegret2[sim, episode], NUMBER_INFEASIBILITIES[sim, episode], select_baseline_policy_ct, dtime))

        # reset the counters
        ep_count = np.zeros((N_STATES, N_ACTIONS))
        ep_count_p = np.zeros((N_STATES, N_ACTIONS, N_STATES))
        for s in range(N_STATES):
            ep_emp_reward[s] = {}
            ep_emp_cost[s] = {}
            for a in range(N_ACTIONS):
                ep_emp_reward[s][a] = 0
                ep_emp_cost[s][a] = 0        
        

        s = s_idx_init
        for h in range(EPISODE_LENGTH): # for each step in current episode
            prob = pi_k[s, h, :]
            
            a = int(np.random.choice(ACTIONS, 1, replace = True, p = prob)) # select action based on the policy/probability
            next_state, rew, cost = util_methods.step(s, a, h) # take the action and get the next state, reward and cost
            ep_count[s, a] += 1 # update the counter
            ep_count_p[s, a, next_state] += 1
            ep_emp_reward[s][a] += rew
            ep_emp_cost[s][a] += cost # this is the SBP feedback
            s = next_state

        # dump results out every xxx episodes
        if episode != 0 and episode%1000== 0:

            filename = 'output/OptCMDP_opsrl' + str(RUN_NUMBER) + '.pkl'
            f = open(filename, 'ab')
            pickle.dump([NUMBER_SIMULATIONS, NUMBER_EPISODES, objs , cons, pi_k, NUMBER_INFEASIBILITIES, q_k], f)
            f.close()
            objs = []
            cons = []

        elif episode == NUMBER_EPISODES-1: # dump results out at the end of the last episode
            filename = 'output/OptCMDP_opsrl' + str(RUN_NUMBER) + '.pkl'
            f = open(filename, 'ab')
            pickle.dump([NUMBER_SIMULATIONS, NUMBER_EPISODES, objs , cons, pi_k, NUMBER_INFEASIBILITIES, q_k], f)
            f.close()
        
# take average/std over multiple simulation runs
ObjRegret_mean = np.mean(ObjRegret2, axis = 0) 
ConRegret_mean = np.mean(ConRegret2, axis = 0)
ObjRegret_std = np.std(ObjRegret2, axis = 0)
ConRegret_std = np.std(ConRegret2, axis = 0)

# save the results as a pickle file
filename = 'output/OptCMDP_regrets_' + str(RUN_NUMBER) + '.pkl'
with open(filename, 'wb') as f:
    pickle.dump([NUMBER_SIMULATIONS, NUMBER_EPISODES, ObjRegret_mean, ObjRegret_std, ConRegret_mean, ConRegret_std], f)