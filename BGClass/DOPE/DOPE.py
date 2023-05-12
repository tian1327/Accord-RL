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
alpha_k = 0.002

use_gurobi = False
RUN_NUMBER = 16 #Change this field to set the seed for the experiment, and change the CONSTRAINT value

if len(sys.argv) > 1:
    use_gurobi = sys.argv[1]

NUMBER_SIMULATIONS = 1
random.seed(int(RUN_NUMBER))
np.random.seed(int(RUN_NUMBER))

# make the output directory if it doesn't exist
if not os.path.exists('output'):
    os.makedirs('output')
    print("Created output/ directory")

# remove the filename = 'output/opsrl' + str(RUN_NUMBER) + '.pkl' to avoid reading old data
old_filename = 'output/DOPE_opsrl' + str(RUN_NUMBER) + '.pkl'
if os.path.exists(old_filename):
    os.remove(old_filename)
    print("Removed old file: ", old_filename)


# Initialize:
with open('output/model.pkl', 'rb') as f:
    [P, R, C, INIT_STATE_INDEX, INIT_STATES_LIST, state_code_to_index, CONSTRAINT, C_b,
     N_STATES, N_ACTIONS, ACTIONS_PER_STATE, EPISODE_LENGTH, DELTA] = pickle.load(f)

with  open('output/solution.pkl', 'rb') as f:
    [opt_policy_con_list, opt_value_LP_con_list, opt_cost_LP_con_list, opt_q_con_list] = pickle.load(f) 

with open('output/base.pkl', 'rb') as f:
    [pi_b_list, val_b_list, cost_b_list, q_b_list] = pickle.load(f)

EPS = 1 # not used
M = 1024* N_STATES*EPISODE_LENGTH**2/EPS**2 # not used

CONSTRAINT = RUN_NUMBER# +++++

Cb = C_b
print("CONSTRAINT =", CONSTRAINT)
print("Cb =", Cb)
print("CONSTRAINT - Cb =", CONSTRAINT - Cb)
print("N_STATES =", N_STATES)
print("N_ACTIONS =", N_ACTIONS)

# define k0
K0 = alpha_k * N_STATES**2 *N_ACTIONS *EPISODE_LENGTH**4/((CONSTRAINT - Cb)**2) # equation in Page 7 for DOPE paper
# K0 = -1
#K0 = 2000

print()
print("alpha_k =", alpha_k)
print("K0 =", K0)
print("number of episodes =", NUMBER_EPISODES)
assert K0 < NUMBER_EPISODES, "K0 is greater than the number of episodes"

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

# pause for 5 seconds to allow the user to read the output
time.sleep(5)

for sim in range(NUMBER_SIMULATIONS):

    util_methods = utils(EPS, DELTA, M, P, R, C, INIT_STATE_INDEX, 
                         EPISODE_LENGTH, N_STATES, N_ACTIONS, ACTIONS_PER_STATE, CONSTRAINT, Cb, use_gurobi) # set the utility methods for each run

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
    for episode in range(NUMBER_EPISODES):

        # sample a initial state s uniformly from the list of initial states INIT_STATES_LIST
        s_code = np.random.choice(INIT_STATES_LIST, 1, replace = True)[0]
        s_idx_init = state_code_to_index[s_code]
        util_methods.update_mu(s_idx_init)

        # set corresponding base policy and optimal policy
        pi_b = pi_b_list[s_idx_init]
        val_b = val_b_list[s_idx_init]
        cost_b = cost_b_list[s_idx_init]
        q_b = q_b_list[s_idx_init]

        opt_value_LP_con = opt_value_LP_con_list[s_idx_init]


        if episode <= K0: # use the safe base policy when the episode is less than K0
            pi_k = pi_b
            val_k = val_b
            cost_k = cost_b
            q_k = q_b
            util_methods.setCounts(ep_count_p, ep_count) # add the counts to the utility methods counter
            util_methods.update_empirical_model(0) # update the transition probabilities P_hat based on the counter
            util_methods.update_empirical_rewards_costs(ep_emp_reward, ep_emp_cost)
            util_methods.compute_confidence_intervals_DOPE(L, L_prime, 1) # compute the confidence intervals for the transition probabilities beta
            dtime = 0

        else: # use the DOPE policy when the episode is greater than K0
            util_methods.setCounts(ep_count_p, ep_count)
            util_methods.update_empirical_model(0) # here we only update the transition probabilities P_hat after finishing 1 full episode
            util_methods.update_empirical_rewards_costs(ep_emp_reward, ep_emp_cost)
            util_methods.compute_confidence_intervals_DOPE(L, L_prime, 1)

            t1 = time.time()
            # +++++ select policy using the extended LP, by solving the DOP problem, equation (10)
            pi_k, val_k, cost_k, log, q_k = util_methods.compute_extended_LP() 
            t2 = time.time()
            dtime = t2 - t1
            # print("\nTime for extended LP = {:.2f} s".format(dtime))
            
            if log != 'Optimal':  #Added this part to resolve issues about infeasibility. Because I am not sure about the value of K0, this condition would take care of that
                print('+++++Infeasible in Extended LP, switch to base policy')
                pi_k = pi_b
                val_k = val_b
                cost_k = cost_b
                q_k = q_b

            else:
                pass
                #print('+++++In episode', episode, 'found optimal policy')



        print('s_idx_init=', s_idx_init)
        #print('cost_b[s_idx_init, 0]=', cost_b[s_idx_init, 0])
        print('cost_k[s_idx_init, 0]=', cost_k[s_idx_init, 0])

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
            else:
                NUMBER_INFEASIBILITIES[sim, episode] = NUMBER_INFEASIBILITIES[sim, episode - 1]            

        print('Episode {}, ObjRegt = {:.2f}, ConsRegt = {:.2f}, #Infeas = {}, Time = {:.2f}'.format(
              episode, ObjRegret2[sim, episode], ConRegret2[sim, episode], NUMBER_INFEASIBILITIES[sim, episode], dtime))

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

        # dump results out every 50000 episodes
        if episode != 0 and episode%1000== 0:

            filename = 'output/DOPE_opsrl' + str(RUN_NUMBER) + '.pkl'
            f = open(filename, 'ab')
            pickle.dump([NUMBER_SIMULATIONS, NUMBER_EPISODES, objs , cons, pi_k, NUMBER_INFEASIBILITIES, q_k], f)
            f.close()
            objs = []
            cons = []

        elif episode == NUMBER_EPISODES-1: # dump results out at the end of the last episode
            filename = 'output/DOPE_opsrl' + str(RUN_NUMBER) + '.pkl'
            f = open(filename, 'ab')
            pickle.dump([NUMBER_SIMULATIONS, NUMBER_EPISODES, objs , cons, pi_k, NUMBER_INFEASIBILITIES, q_k], f)
            f.close()
        
# take average/std over multiple simulation runs
ObjRegret_mean = np.mean(ObjRegret2, axis = 0) 
ConRegret_mean = np.mean(ConRegret2, axis = 0)
ObjRegret_std = np.std(ObjRegret2, axis = 0)
ConRegret_std = np.std(ConRegret2, axis = 0)

# save the results as a pickle file
filename = 'output/DOPE_regrets_' + str(RUN_NUMBER) + '.pkl'
with open(filename, 'wb') as f:
    pickle.dump([NUMBER_SIMULATIONS, NUMBER_EPISODES, ObjRegret_mean, ObjRegret_std, ConRegret_mean, ConRegret_std], f)