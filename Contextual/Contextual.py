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

context_fea = ['baseline_age', 'female', 'race_whiteother',
                'edu_baseline_1',
                'edu_baseline_2',
                'edu_baseline_3',
                'cvd_hx_baseline', 
                'baseline_BMI', 
                # 'baseline_BMI_discrete',
                # 'cigarett_baseline',
                'cigarett_baseline_1',
               ]

def generate_random_patient():
    res = np.zeros(len(context_fea))

    res[0] = np.random.randint(45, 80) # age
    res[1] = np.random.randint(0, 2) # female
    res[2] = np.random.randint(0, 2) # race_whiteother
    edu = np.random.randint(0, 4)
    if edu == 0:
        res[3] = 1
    elif edu == 1:
        res[4] = 1
    elif edu == 2:
        res[5] = 1
    res[6] = np.random.randint(0, 2) # cvd_hx_baseline
    res[7] = np.random.uniform(18.5, 45) # baseline_BMI
    res[8] = np.random.randint(0, 2) # cigarett_baseline_1
    # print(res)
    # print(res.shape)

    return res



def discretize_sbp(sbp):
    if sbp < 120:
        return 0
    elif sbp < 140:
        return 1
    else:
        return 2

start_time = time.time()



# control parameters
NUMBER_EPISODES = 1e6
alpha_k = 1
sample_data = False # whether to sample data from the dataset or randomly generate data
random_action = True # whether to use random action or use the optimal action


NUMBER_SIMULATIONS = 1
RUN_NUMBER = 10 #Change this field to set the seed for the experiment.

random.seed(RUN_NUMBER)
np.random.seed(RUN_NUMBER)

# remove the filename = 'output/opsrl' + str(RUN_NUMBER) + '.pkl' to avoid reading old data
old_filename = 'output/opsrl' + str(RUN_NUMBER) + '.pkl'
if os.path.exists(old_filename):
    os.remove(old_filename)
    print("Removed old file: ", old_filename)

# Initialize:
with open('output/model_contextual.pkl', 'rb') as f:
    [P, CONTEXT_VEC_LENGTH, ACTION_CODE_LENGTH, CONTEXT_VECTOR_dict, INIT_STATE_INDEX, INIT_STATES_LIST, state_code_to_index, state_index_to_code, action_index_to_code,
    CONSTRAINT, Cb, N_STATES, N_ACTIONS, ACTIONS_PER_STATE, EPISODE_LENGTH, DELTA] = pickle.load(f)

# load the trained CVDRisk_estimator and SBP_feedback_estimator from pickle file
R_model = pickle.load(open('output/CVDRisk_estimator_BP.pkl', 'rb'))
C_model = pickle.load(open('output/SBP_feedback_estimator.pkl', 'rb'))

EPS = 1 # not used
M = 1024* N_STATES*EPISODE_LENGTH**2/EPS**2 # not used

# Cb = cost_b[0, 0]
Cb = Cb
print("CONSTRAINT =", CONSTRAINT)
print("Cb =", Cb)
print("CONSTRAINT - Cb =", CONSTRAINT - Cb)
print("N_STATES =", N_STATES)
print("N_ACTIONS =", N_ACTIONS)

# define k0
K0 = alpha_k * (EPISODE_LENGTH/(Cb-CONSTRAINT))**2  
k0 = -1 # no baseline

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
L_prime = 2 * math.log(6 * N_STATES* N_ACTIONS * EPISODE_LENGTH * NUMBER_EPISODES / DELTA) # for SBP, CVDRisk, not used in Contextual algorithm
# page 11 in word document, to calculated the confidence intervals for the transition probabilities beta
print("L =", L)
print("L_prime =", L_prime)

for sim in range(NUMBER_SIMULATIONS):

    util_methods = utils(EPS, DELTA, M, P, R_model, C_model, CONTEXT_VEC_LENGTH, ACTION_CODE_LENGTH, INIT_STATE_INDEX, state_index_to_code, action_index_to_code,
                         EPISODE_LENGTH, N_STATES, N_ACTIONS, ACTIONS_PER_STATE, CONSTRAINT, Cb) # set the utility methods for each run

    # for empirical estimate of transition probabilities P_hat
    ep_count = np.zeros((N_STATES, N_ACTIONS)) # initialize the counter for each run
    ep_count_p = np.zeros((N_STATES, N_ACTIONS, N_STATES))

    # for logistic regression of CVDRisk_feedback and linear regression SBP_feedback
    ep_sbp_discrete = [] # record the SBP feedback discrete for each step in a episode
    ep_sbp_cont = [] # record the SBP feedback continuous for each step in a episode
    ep_action_code = [] # record the action code for each step in a episode
    ep_cvdrisk = [] # record the CVDRisk for each step in a episode

    objs = [] # objective regret for current run
    cons = []
    R_est_err = []
    C_est_err = []
    min_eign_cvd_list = []
    min_eign_sbp_list = []

    #first_infeasible = True
    #found_optimal = False

    # global pi_b_prev, val_b_prev, cost_b_prev, q_b_prev
    # pi_b_prev = None
    # val_b_prev = None
    # cost_b_prev = None
    # q_b_prev = None

    episode = 0
    while episode < NUMBER_EPISODES:

        if sample_data:
            #---------- sample a patient from CONTEXT_VECTOR_dict
            patient = np.random.choice(list(CONTEXT_VECTOR_dict.keys()), 1, replace = True)[0]
            context_vec = CONTEXT_VECTOR_dict[patient][0]
            sbp_discrete_init = CONTEXT_VECTOR_dict[patient][1]
        else:
            #---------- generate a random patient
            context_vec = generate_random_patient()
            sbp_discrete_init = np.random.choice([0, 1, 2], 1, replace = True)[0]

        # print('len(context_vec) =', len(context_vec))
        util_methods.set_context(context_vec) # set the context vector for the current episode

        # sample a initial state s uniformly from the list of initial states INIT_STATES_LIST
        s_code = np.random.choice(INIT_STATES_LIST, 1, replace = True)[0]
        s_idx_init = state_code_to_index[s_code]
        util_methods.update_mu(s_idx_init)

        # calculate the R and C based on the true R and C models, regenerate for each episode
        util_methods.calculate_true_R_C(context_vec)

        # get the optimal and baseline policy for current patient with context_vec, and initial state s_idx
        opt_policy_con, opt_value_LP_con, opt_cost_LP_con, opt_q_con, status = util_methods.compute_opt_LP_Constrained(0, 'Optimal Policy -') 

        util_methods.update_CONSTRAINT(Cb) # set the C to Cb for calculating the baseline policy
        pi_b, val_b, cost_b, q_b, status = util_methods.compute_opt_LP_Constrained(0, 'Baseline Policy -')
        util_methods.update_CONSTRAINT(CONSTRAINT) # reset the C to the original value

        util_methods.update_episode(episode) # update the episode number for the utility methods
        
        # for some cases the baseline policy maynot be feasible, in this case, we use the previous feasible baseline policy
        if status == 'Infeasible':
            print("Baseline policy is infeasible, skip to the next patient")
            continue # simply skip this patient

            # print("Baseline policy is infeasible")
            # pi_b = pi_b_prev
            # val_b = val_b_prev
            # cost_b = cost_b_prev
            # q_b = q_b_prev

            # print('pi_k =', pi_k)
            # print('pi_b_prev =', pi_b_prev)

        # else: # record the current feasible baseline policy

            # print("Baseline policy is feasible, record the current baseline policy")
            # pi_b_prev = pi_b
            # val_b_prev = val_b
            # cost_b_prev = cost_b
            # q_b_prev = q_b

        if episode <= K0: # use the safe base policy when the episode is less than K0
            pi_k = pi_b
            val_k = val_b
            cost_k = cost_b
            q_k = q_b

            util_methods.setCounts(ep_count_p, ep_count) # add the counts to the utility methods counter
            util_methods.update_empirical_model(0) # update the transition probabilities P_hat based on the counter
            util_methods.add_ep_rewards_costs(ep_sbp_discrete, ep_sbp_cont, ep_action_code, ep_cvdrisk) # add the collected SBP and action index to the history data for regression
            # R_est_error, C_est_error = util_methods.run_regression_rewards_costs(episode) # update the regression models for SBP and CVDRisk
            # util_methods.compute_confidence_intervals(L, L_prime, 1)
            R_est_error, C_est_error = 0, 0 
            min_eign_cvd, min_eign_sbp = 0, 0
            dtime = 0

        else: # when the episode is greater than K0, solve the extended LP to get the policy
            util_methods.setCounts(ep_count_p, ep_count)
            util_methods.update_empirical_model(0) # here we only update the transition probabilities P_hat after finishing 1 full episode
            util_methods.add_ep_rewards_costs(ep_sbp_discrete, ep_sbp_cont, ep_action_code, ep_cvdrisk) # add the collected SBP and action index to the history data for regression
            R_est_error, C_est_error = util_methods.run_regression_rewards_costs(episode) # update the regression models for SBP and CVDRisk
            min_eign_cvd, min_eign_sbp = util_methods.compute_confidence_intervals(L, L_prime, 1)
            # util_methods.compute_confidence_intervals_2(L, L_prime, 1)

            t1 = time.time()
            # +++++ select policy using the extended LP, by solving the DOP problem, equation (10)
            pi_k, val_k, cost_k, log, q_k = util_methods.compute_extended_LP(0, Cb) 
            t2 = time.time()
            dtime = t2 - t1
            # print("\nTime for extended LP = {:.2f} s".format(dtime))

            if log != 'Optimal':
                print('Infeasible solution in Extended LP, continue to the next patient')
                continue

            # if log != 'Optimal':  #Added this part to resolve issues about infeasibility. Because I am not sure about the value of K0, this condition would take care of that
            #     pi_k = pi_b
            #     val_k = val_b
            #     cost_k = cost_b
            #     q_k = q_b
            #     if first_infeasible:
            #         print('\nlog:', log)
            #         first_infeasible = False
            # else:
            #     if not found_optimal:
            #         print('\nlog:', log)
            #         print('In episode', episode, 'found optimal policy')
            #         print('k0 should be at least', episode)
            #         found_optimal = True
        
        R_est_err.append(R_est_error)
        C_est_err.append(C_est_error)
        min_eign_cvd_list.append(min_eign_cvd)
        min_eign_sbp_list.append(min_eign_sbp)

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

        print('Episode {}, s_idx_init= {}, ObjRegt = {:.2f}, ConsRegt = {:.2f}, #Infeas = {}, Time = {:.2f}\n'.format(
              episode, s_idx_init, ObjRegret2[sim, episode], ConRegret2[sim, episode], NUMBER_INFEASIBILITIES[sim, episode], dtime))

        # reset the counters
        ep_count = np.zeros((N_STATES, N_ACTIONS))
        ep_count_p = np.zeros((N_STATES, N_ACTIONS, N_STATES))
        ep_sbp_discrete = [] # record the SBP for each step in a episode, for the current timestep
        ep_sbp_discrete.append(sbp_discrete_init)
        ep_sbp_cont = [] # record the SBP feedback continuous for each step in a episode
        ep_action_code = [] # record the action code for each step in a episode
        ep_cvdrisk = [] # record the CVDRisk for each step in a episode    
        
        s = s_idx_init # set the state to the initial state
        for h in range(EPISODE_LENGTH): # for each step in current episode
            prob = pi_k[s, h, :]
            
            # if sum(prob) != 1:
            #    print(s, h)
            #    print(prob)
            
            # # check if prob has any negative values
            # for i in range(len(prob)):
            #     if prob[i] < 0:
            #         # print("negative prob: ", prob[i])
            #         prob[i] = 0
            
            # assign uniform prob to prob, used for testing the code only
            # for i in range(len(prob)):
            #         prob[i] = 1/len(prob)

            if random_action:
                # sample actions uniformly
                a = int(np.random.choice(ACTIONS, 1, replace = True))
            else:
                a = int(np.random.choice(ACTIONS, 1, replace = True, p = prob)) # select action based on the policy/probability

            next_state, rew, cost = util_methods.step(s, a, h) # take the action and get the next state, reward and cost
            current_sbp_discrete = ep_sbp_discrete[h] # get the SBP for the current timestep
            # print('current_sbp_discrete = ', current_sbp_discrete)
            # print('type(current_sbp_discrete) = ', type(current_sbp_discrete))
            sbp_xa_vec, cvd_xsa_vec = util_methods.make_x_a_vector(context_vec, current_sbp_discrete, a)
            sbp_fb_dis = discretize_sbp(cost)
            util_methods.add_design_vector(sbp_xa_vec, cvd_xsa_vec)

            ep_count[s, a] += 1 # update the counter
            ep_count_p[s, a, next_state] += 1
            ep_sbp_cont.append(cost) # this is the SBP feedback continuous
            if h != EPISODE_LENGTH - 1:       
                ep_sbp_discrete.append(sbp_fb_dis)

            ep_action_code.append(action_index_to_code[a]) 
            ep_cvdrisk.append(rew)

            s = next_state

        # dump results out every 50000 episodes
        if episode != 0 and episode%200== 0:

            filename = 'output/opsrl' + str(RUN_NUMBER) + '.pkl'
            f = open(filename, 'ab')
            pickle.dump([R_est_err, C_est_err, min_eign_sbp_list, min_eign_cvd_list, NUMBER_SIMULATIONS, NUMBER_EPISODES, objs , cons, pi_k, NUMBER_INFEASIBILITIES, q_k], f)
            f.close()
            objs = []
            cons = []
            R_est_err = []
            C_est_err = []
            min_eign_sbp_list = []
            min_eign_cvd_list = []

        elif episode == NUMBER_EPISODES-1: # dump results out at the end of the last episode
            filename = 'opsrl' + str(RUN_NUMBER) + '.pkl'
            f = open(filename, 'ab')
            pickle.dump([R_est_err, C_est_err, min_eign_sbp_list, min_eign_cvd_list, NUMBER_SIMULATIONS, NUMBER_EPISODES, objs , cons, pi_k, NUMBER_INFEASIBILITIES, q_k], f)
            f.close()
        
        episode += 1
        
# take average/std over multiple simulation runs
ObjRegret_mean = np.mean(ObjRegret2, axis = 0) 
ConRegret_mean = np.mean(ConRegret2, axis = 0)
ObjRegret_std = np.std(ObjRegret2, axis = 0)
ConRegret_std = np.std(ConRegret2, axis = 0)

# save the results as a pickle file
filename = 'contextual_regrets_' + str(RUN_NUMBER) + '.pkl'
with open(filename, 'wb') as f:
    pickle.dump([NUMBER_SIMULATIONS, NUMBER_EPISODES, ObjRegret_mean, ObjRegret_std, ConRegret_mean, ConRegret_std], f)