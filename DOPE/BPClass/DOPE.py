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
NUMBER_EPISODES = 1e5
alpha = 1000000
NUMBER_SIMULATIONS = 1
# temp = sys.argv[1:]
# RUN_NUMBER = int(temp[0])
RUN_NUMBER = 10 #Change this field to set the seed for the experiment.

random.seed(RUN_NUMBER)
np.random.seed(RUN_NUMBER)

# Initialize:
with open('model.pkl', 'rb') as f:
    [P, R, C, INIT_STATE_INDEX, INIT_STATES_LIST, state_code_to_index, CONSTRAINT, 
     N_STATES, N_ACTIONS, actions_per_state, EPISODE_LENGTH, DELTA] = pickle.load(f)

with  open('solution.pkl', 'rb') as f:
    [opt_policy_con, opt_value_LP_con, opt_cost_LP_con, opt_q_con] = pickle.load(f) 

with open('base.pkl', 'rb') as f:
    [pi_b, val_b, cost_b, q_b] = pickle.load(f)

EPS = 1 # not used
M = 1024* N_STATES*EPISODE_LENGTH**2/EPS**2 # not used

Cb = cost_b[0, 0]
print("CONSTRAINT =", CONSTRAINT)
print("Cb =", Cb)
print("CONSTRAINT - Cb =", CONSTRAINT - Cb)
print("N_STATES =", N_STATES)
print("N_ACTIONS =", N_ACTIONS)

# define k0
# K0 = alpha * N_STATES**2 *N_ACTIONS *EPISODE_LENGTH**4/((CONSTRAINT - Cb)**2) # equation in Page 7 for DOPE paper
K0 = alpha * (EPISODE_LENGTH/(CONSTRAINT - Cb))**2  # equation in Page 9 of the word document 

print()
print("alpha =", alpha)
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

for sim in tqdm(range(NUMBER_SIMULATIONS)):

    util_methods = utils(EPS, DELTA, M, P, R, C, INIT_STATE_INDEX, 
                         EPISODE_LENGTH, N_STATES, N_ACTIONS, actions_per_state, CONSTRAINT, Cb) # set the utility methods for each run

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
    for episode in tqdm(range(NUMBER_EPISODES)): # loop for episodes

        found_optimal = False
        
        if episode <= K0: # use the safe base policy when the episode is less than K0
            pi_k = pi_b
            val_k = val_b
            cost_k = cost_b
            q_k = q_b
            util_methods.setCounts(ep_count_p, ep_count) # add the counts to the utility methods counter
            util_methods.update_empirical_model(0) # update the transition probabilities P_hat based on the counter
            util_methods.update_empirical_rewards_costs(ep_emp_reward, ep_emp_cost)
            util_methods.compute_confidence_intervals(L, L_prime, 1) # compute the confidence intervals for the transition probabilities beta

        else: # use the DOPE policy when the episode is greater than K0
            util_methods.setCounts(ep_count_p, ep_count)
            util_methods.update_empirical_model(0) # here we only update the transition probabilities P_hat after finishing 1 full episode
            util_methods.update_empirical_rewards_costs(ep_emp_reward, ep_emp_cost)
            util_methods.compute_confidence_intervals(L, L_prime, 1)

            t1 = time.time()
            # +++++ select policy using the extended LP, by solving the DOP problem, equation (10)
            pi_k, val_k, cost_k, log, q_k = util_methods.compute_extended_LP(0, Cb) 
            t2 = time.time()
            print("time for extended LP = {:.2f} s".format(t2 - t1))
            
            if log != 'Optimal':  #Added this part to resolve issues about infeasibility. Because I am not sure about the value of K0, this condition would take care of that
                pi_k = pi_b
                val_k = val_b
                cost_k = cost_b
                q_k = q_b
                print('log:', log)
            else:
                if not found_optimal:
                    print('log:', log)
                    print('In episode', episode, 'found optimal policy')
                    print('k0 should be at least', episode)
                    found_optimal = True
        
        if episode == 0:
            ObjRegret2[sim, episode] = abs(val_k[0, 0] - opt_value_LP_con[0, 0]) # for episode 0, calculate the objective regret, we care about the value of a policy at the initial state
            ConRegret2[sim, episode] = max(0, cost_k[0, 0] - CONSTRAINT) # constraint regret, we care about the cumulative cost of a policy at the initial state
            objs.append(ObjRegret2[sim, episode])
            cons.append(ConRegret2[sim, episode])
            if cost_k[0, 0] > CONSTRAINT:
                NUMBER_INFEASIBILITIES[sim, episode] = 1
        else:
            ObjRegret2[sim, episode] = ObjRegret2[sim, episode - 1] + abs(val_k[0, 0] - opt_value_LP_con[0, 0]) # calculate the objective regret, note this is cumulative sum upto k episode, beginninng of page 8 in the paper
            ConRegret2[sim, episode] = ConRegret2[sim, episode - 1] + max(0, cost_k[0, 0] - CONSTRAINT) # cumulative sum of constraint regret
            objs.append(ObjRegret2[sim, episode])
            cons.append(ConRegret2[sim, episode])
            if cost_k[0, 0] > CONSTRAINT:
                NUMBER_INFEASIBILITIES[sim, episode] = NUMBER_INFEASIBILITIES[sim, episode - 1] + 1 # count the number of infeasibilities until k episode
        
        # reset the counters
        ep_count = np.zeros((N_STATES, N_ACTIONS))
        ep_count_p = np.zeros((N_STATES, N_ACTIONS, N_STATES))
        for s in range(N_STATES):
            ep_emp_reward[s] = {}
            ep_emp_cost[s] = {}
            for a in range(N_ACTIONS):
                ep_emp_reward[s][a] = 0
                ep_emp_cost[s][a] = 0        
        
        # s = 0 # initial state is always fixed to 0 +++++
        # s = INIT_STATE_INDEX # needs to sample unformly from the available init states in the dataset

        # sample a initial state s uniformly from the list of initial states INIT_STATES_LIST
        s_code = np.random.choice(INIT_STATES_LIST, 1, replace = True)
        s = state_code_to_index[s_code[0]]
        # print('s = ', s)
        # print('s_idx = ', s_idx)
        

        # update self.mu
        util_methods.update_mu(s)

        for h in range(EPISODE_LENGTH): # for each step in current episode
            prob = pi_k[s, h, :]
            
            if sum(prob) != 1:
               print(s, h)
               print(prob)
            
            # # check if prob has any negative values
            # for i in range(len(prob)):
            #     if prob[i] < 0:
            #         # print("negative prob: ", prob[i])
            #         prob[i] = 0
            
            # assign uniform prob to prob, used for testing the code only
            # for i in range(len(prob)):
            #         prob[i] = 1/len(prob)

            a = int(np.random.choice(ACTIONS, 1, replace = True, p = prob)) # select action based on the policy/probability
            next_state, rew, cost = util_methods.step(s, a, h) # take the action and get the next state, reward and cost
            ep_count[s, a] += 1 # update the counter
            ep_count_p[s, a, next_state] += 1
            ep_emp_reward[s][a] += rew
            ep_emp_cost[s][a] += cost
            s = next_state

        # dump results out every 50000 episodes
        if episode != 0 and episode%50000== 0:

            filename = 'opsrl' + str(RUN_NUMBER) + '.pkl'
            f = open(filename, 'ab')
            pickle.dump([NUMBER_SIMULATIONS, NUMBER_EPISODES, objs , cons, pi_k, NUMBER_INFEASIBILITIES, q_k], f)
            f.close()
            objs = []
            cons = []

        elif episode == NUMBER_EPISODES-1: # dump results out at the end of the last episode
            filename = 'opsrl' + str(RUN_NUMBER) + '.pkl'
            f = open(filename, 'ab')
            pickle.dump([NUMBER_SIMULATIONS, NUMBER_EPISODES, objs , cons, pi_k, NUMBER_INFEASIBILITIES, q_k], f)
            f.close()
        
# take average/std over multiple simulation runs
ObjRegret_mean = np.mean(ObjRegret2, axis = 0) 
ConRegret_mean = np.mean(ConRegret2, axis = 0)
ObjRegret_std = np.std(ObjRegret2, axis = 0)
ConRegret_std = np.std(ConRegret2, axis = 0)

# save the results as a pickle file
filename = 'regrets_' + str(RUN_NUMBER) + '.pkl'
with open(filename, 'wb') as f:
    pickle.dump([NUMBER_SIMULATIONS, NUMBER_EPISODES, ObjRegret_mean, ObjRegret_std, ConRegret_mean, ConRegret_std], f)

# print(NUMBER_INFEASIBILITIES)
# print(util_methods.NUMBER_OF_OCCURANCES[0])



"""
print("\nPlotting the results ...")

title = 'OPSRL' + str(RUN_NUMBER)
plt.figure()
plt.plot(range(NUMBER_EPISODES), ObjRegret_mean)
plt.fill_between(range(NUMBER_EPISODES), ObjRegret_mean - ObjRegret_std, ObjRegret_mean + ObjRegret_std, alpha = 0.5)
plt.grid()
plt.xlabel('Episodes')
plt.ylabel('Objective Regret')
plt.title(title)
plt.savefig(title + '_ObjectiveRegret.png')
plt.show()

time = np.arange(1, NUMBER_EPISODES+1)
squareroot = [int(b) / int(m) for b,m in zip(ObjRegret_mean, np.sqrt(time))]

plt.figure()
plt.plot(range(NUMBER_EPISODES),squareroot)
#plt.fill_between(range(NUMBER_EPISODES), ObjRegret_mean - ObjRegret_std, ObjRegret_mean + ObjRegret_std, alpha = 0.5)
plt.grid()
plt.xlabel('Episodes')
plt.ylabel('Objective Regret square root curve')
plt.title(title)
plt.savefig(title + '_ObjectiveRegretSQRT.png')
plt.show()

plt.figure()
plt.plot(range(NUMBER_EPISODES), ConRegret_mean)
plt.fill_between(range(NUMBER_EPISODES), ConRegret_mean - ConRegret_std, ConRegret_mean + ConRegret_std, alpha = 0.5)
plt.grid()
plt.xlabel('Episodes')
plt.ylabel('Constraint Regret')
plt.title(title)
plt.savefig(title + '_ConstraintRegret.png')
plt.show()
"""