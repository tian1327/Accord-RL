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
               'edu_baseline_1', 'edu_baseline_2', 'edu_baseline_3',
               'cvd_hx_baseline', 'baseline_BMI', 'cigarett_baseline_1',
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

# def discretize_sbp(sbp):
#     if sbp < 120:
#         return 0
#     elif sbp < 140:
#         return 1
#     else:
#         return 2


#--------------------------------------------------------------------------------------

# control parameters
NUMBER_EPISODES = 3e4
alpha_k = 1e5 # control K0, but not used
sample_data = True # whether to sample data from the dataset or randomly generate data
random_action = False # whether to use random action or use the optimal action
RUN_NUMBER = 100 #Change this field to set the seed for the experiment.

use_gurobi = False # whether to use gurobi to solve the optimization problem
NUMBER_SIMULATIONS = 1

if len(sys.argv) > 1:
    use_gurobi = sys.argv[1]

print('sample_data =', sample_data)
print('random_action =', random_action)

#--------------------------------------------------------------------------------------
random.seed(int(RUN_NUMBER))
np.random.seed(int(RUN_NUMBER))

# remove old file
old_filename = 'output/CONTEXTUAL_opsrl' + str(RUN_NUMBER) + '.pkl'
if os.path.exists(old_filename):
    os.remove(old_filename)
    print("Removed old file: ", old_filename)

# Initialize:
with open('output/model_contextual_BG.pkl', 'rb') as f:
    [P, CONTEXT_VEC_LENGTH, ACTION_CODE_LENGTH, CONTEXT_VECTOR_dict, INIT_STATE_INDEX, INIT_STATES_LIST, 
    state_code_to_index, state_index_to_code, action_index_to_code,
    CONSTRAINT_list, C_b_list, N_STATES, N_ACTIONS, ACTIONS_PER_STATE, EPISODE_LENGTH, DELTA] = pickle.load(f)

STATE_CODE_LENGTH = len(state_index_to_code[0])
print("STATE_CODE_LENGTH =", STATE_CODE_LENGTH)

# load the trained CVDRisk_estimator and SBP_feedback_estimator from pickle file
R_model = pickle.load(open('output/CVDRisk_estimator_BG.pkl', 'rb'))
C_model = pickle.load(open('output/A1C_feedback_estimator_BG.pkl', 'rb'))


# CONSTRAINT = RUN_NUMBER # +++++

CONSTRAINT = CONSTRAINT_list[-1]
C_b = C_b_list[-1]

Cb = C_b
#Cb = 150

print("CONSTRAINT =", CONSTRAINT)
print("Cb =", Cb)
print("CONSTRAINT - Cb =", CONSTRAINT - Cb)
print("N_STATES =", N_STATES)
print("N_ACTIONS =", N_ACTIONS)

# define k0
K0 = alpha_k * (EPISODE_LENGTH/(Cb-CONSTRAINT))**2  
K0 = -1 # no baseline
#K0 = 100 # warm up episodes for infeasible solution in extended LP with cold start

print()
print("alpha_k =", alpha_k)
print("++ K0 =", K0)
print("number of episodes =", NUMBER_EPISODES)
assert K0 < NUMBER_EPISODES, "K0 is greater than the number of episodes"

NUMBER_EPISODES = int(NUMBER_EPISODES)
NUMBER_SIMULATIONS = int(NUMBER_SIMULATIONS)
ACTIONS = np.arange(N_ACTIONS)

ObjRegret2 = np.zeros((NUMBER_SIMULATIONS, NUMBER_EPISODES))
ConRegret2 = np.zeros((NUMBER_SIMULATIONS, NUMBER_EPISODES))
NUMBER_INFEASIBILITIES = np.zeros((NUMBER_SIMULATIONS, NUMBER_EPISODES))

L = math.log(2 * N_STATES * N_ACTIONS * EPISODE_LENGTH * NUMBER_EPISODES / DELTA) # for transition probabilities P_hat
print("L =", L)

# not used here
EPS = 0
M = 0

#--------------------------------------------------------------------------------------

for sim in range(NUMBER_SIMULATIONS):

    util_methods = utils(EPS, DELTA, M, P, R_model, C_model, CONTEXT_VEC_LENGTH, ACTION_CODE_LENGTH, STATE_CODE_LENGTH,
                         INIT_STATE_INDEX, state_index_to_code, action_index_to_code,
                         EPISODE_LENGTH, N_STATES, N_ACTIONS, ACTIONS_PER_STATE, CONSTRAINT, Cb, use_gurobi) 

    # for empirical estimate of transition probabilities P_hat
    ep_count = np.zeros((N_STATES, N_ACTIONS)) 
    ep_count_p = np.zeros((N_STATES, N_ACTIONS, N_STATES))

    # for logistic regression of CVDRisk_feedback and linear regression SBP_feedback
    ep_context_vec = None
    ep_sbp_cont = [] # record the SBP feedback continuous for each step in a episode
    ep_hba1c_cont = [] # record the HbA1c feedback continuous for each step in a episode
    ep_state_code = []
    ep_action_code = [] # record the action code for each step in a episode
    ep_cvdrisk = [] # record the CVDRisk for each step in a episode

    objs = [] # objective regret for current run
    cons = []
    R_est_err = []
    C_est_err = []
    min_eign_cvd_list = []
    min_eign_hba1c_list = []

    select_baseline_policy_ct = 0
    episode = 0
    while episode < NUMBER_EPISODES:

        if sample_data:
            # sample a patient from CONTEXT_VECTOR_dict
            patient = np.random.choice(list(CONTEXT_VECTOR_dict.keys()), 1, replace = True)[0]
            context_vec = CONTEXT_VECTOR_dict[patient] # we only use the context vector
        else:
            # generate a random patient
            context_vec = generate_random_patient()

        util_methods.set_context(context_vec) # set the context vector for the current episode

        # sample a initial state s uniformly from the list of initial states INIT_STATES_LIST
        s_code = np.random.choice(INIT_STATES_LIST, 1, replace = True)[0]
        s_idx_init = state_code_to_index[s_code]
        util_methods.update_mu(s_idx_init)
        # print('\ns_code =', s_code, ', s_idx_init =', s_idx_init)

        CONSTRAINT = CONSTRAINT_list[s_idx_init]
        Cb = C_b_list[s_idx_init]
        util_methods.setConstraint(CONSTRAINT)
        util_methods.setCb(Cb)        

        # calculate the R and C based on the true R and C models, regenerate for each episode/patient
        util_methods.calculate_true_R_C(context_vec)

        # get the optimal and baseline policy for current patient with context_vec, and initial state s_idx
        opt_policy_con, opt_value_LP_con, opt_cost_LP_con, opt_q_con, status = util_methods.compute_opt_LP_Constrained(0, 'Optimal Policy -') 

        util_methods.update_CONSTRAINT(Cb) # set the C to Cb for calculating the baseline policy
        pi_b, val_b, cost_b, q_b, status = util_methods.compute_opt_LP_Constrained(0, 'Baseline Policy -')
        util_methods.update_CONSTRAINT(CONSTRAINT) # reset the C to the original value

        util_methods.update_episode(episode) # update the episode number for the utility methods
        
        if status != 'Optimal':
            print("Baseline policy is {}, skip to the next patient".format(status))
            continue # simply skip this patient

        if episode <= K0: # use the safe base policy when the episode is less than K0
            pi_k = pi_b
            val_k = val_b
            cost_k = cost_b
            q_k = q_b

            util_methods.setCounts(ep_count_p, ep_count) # add the counts to the utility methods counter
            util_methods.update_empirical_model(0) # update the transition probabilities P_hat based on the counter
            util_methods.add_ep_rewards_costs(ep_context_vec, ep_state_code, ep_action_code, ep_sbp_cont, ep_hba1c_cont, ep_cvdrisk) # add the collected SBP and action index to the history data for regression
            R_est_error, C_est_error = 0, 0 
            min_eign_cvd, min_eign_sbp = 0, 0
            dtime = 0

        else: # when the episode is greater than K0, solve the extended LP to get the policy
            t1 = time.time() 
            util_methods.setCounts(ep_count_p, ep_count)
            util_methods.update_empirical_model(0) # here we only update the transition probabilities P_hat after finishing 1 full episode
            util_methods.add_ep_rewards_costs(ep_context_vec, ep_state_code, ep_action_code, ep_sbp_cont, ep_hba1c_cont, ep_cvdrisk) # add the collected SBP and action index to the history data for regression
            R_est_error, C_est_error = util_methods.run_regression_rewards_costs_BG(episode) # update the regression models for SBP/Hba1c and CVDRisk
            min_eign_cvd, min_eign_hba1c = util_methods.compute_confidence_intervals_BG(L)
            # util_methods.compute_confidence_intervals_2(L, L_prime, 1)

            if random_action:
                pi_k, val_k, cost_k, log, q_k = util_methods.compute_extended_LP_random() # use uniform probability to select the action
            else:
                pi_k, val_k, cost_k, log, q_k = util_methods.compute_extended_LP() # +++++ select policy using the extended LP, by solving the DOP problem, equation (10)
            
            t2 = time.time()
            dtime = t2 - t1

            if log != 'Optimal':
                #print('+++++Infeasible solution in Extended LP, continue to the next patient')
                #continue

                print('+++++Infeasible solution in Extended LP, select the baseline policy instead')
                select_baseline_policy_ct += 1

                pi_k = pi_b
                val_k = val_b
                cost_k = cost_b
                q_k = q_b
        
        R_est_err.append(R_est_error)
        C_est_err.append(C_est_error)
        min_eign_cvd_list.append(min_eign_cvd)
        min_eign_hba1c_list.append(min_eign_hba1c)

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

        print('RUN_NUMBER: {}, Episode {}, s_idx_init= {}, ObjRegt = {:.2f}, ConsRegt = {:.2f}, Infeas = {}, Infeas in EXLP = {}, Time = {:.2f}\n'.format(
              RUN_NUMBER, episode, s_idx_init, ObjRegret2[sim, episode], ConRegret2[sim, episode], 
              NUMBER_INFEASIBILITIES[sim, episode], select_baseline_policy_ct, dtime))

        # reset the counters
        ep_count = np.zeros((N_STATES, N_ACTIONS))
        ep_count_p = np.zeros((N_STATES, N_ACTIONS, N_STATES))
        ep_context_vec = context_vec # record the context vector for the current episode
        ep_sbp_cont = [] # record the SBP feedback continuous for each step in a episode
        ep_hba1c_cont = [] # record the Hba1c feedback continuous for each step in a episode
        a_list = []
        ep_action_code = [] # record the action code for each step in a episode
        ep_state_code = []
        ep_cvdrisk = [] # record the CVDRisk for each step in a episode  
        
        s = s_idx_init # set the state to the initial state
        for h in range(EPISODE_LENGTH): # for each step in current episode
            
            prob = pi_k[s, h, :]           
            if random_action:
                a = int(np.random.choice(ACTIONS, 1, replace = True)) # sample actions uniformly
            else:
                a = int(np.random.choice(ACTIONS, 1, replace = True, p = prob)) # select action based on the policy/probability

            a_list.append(a)
            ep_action_code.append(action_index_to_code[a]) 
            ep_state_code.append(state_index_to_code[s])

            next_state, rew, cost = util_methods.step(s, a, h) # take the action and get the next state, reward and cost

            xa_vec, cvd_xsa_vec = util_methods.make_x_a_vector(ep_context_vec, s, a)
            
            util_methods.add_design_vector(xa_vec)

            ep_count[s, a] += 1 # update the counter
            ep_count_p[s, a, next_state] += 1
            ep_sbp_cont.append(0) # we are not getting SBP feedback in BGClass case 
            ep_hba1c_cont.append(cost)
            ep_cvdrisk.append(rew)

            s = next_state
            # sbp_fb_dis = discretize_sbp(cost)
            #s = sbp_fb_dis # use the sbp_fb_dis as the next state
        
        # print('a_list = ', a_list)
        # print('context_vec = ', context_vec)
        # print('ep_action_code = ', ep_action_code)
        # print('ep_state_code = ', ep_state_code)
        # print('ep_sbp_cont = ', ep_sbp_cont)
        # print('ep_cvdrisk = ', ep_cvdrisk)


        # pi_k = None
        # q_k = None
        # dump results out every x episodes
        if episode != 0 and episode%500== 0:

            filename = 'output/CONTEXTUAL_opsrl' + str(RUN_NUMBER) + '.pkl'
            f = open(filename, 'ab')
            pickle.dump([R_est_err, C_est_err, min_eign_hba1c_list, min_eign_cvd_list, NUMBER_SIMULATIONS, NUMBER_EPISODES, objs , cons, pi_k, NUMBER_INFEASIBILITIES, q_k], f)
            f.close()
            objs = []
            cons = []
            R_est_err = []
            C_est_err = []
            min_eign_hba1c_list = []
            min_eign_cvd_list = []

            filename = 'output/CONTEXTUAL_BG_regr.pkl'
            with open(filename, 'wb') as f:
                pickle.dump([util_methods.hba1c_regr, util_methods.cvdrisk_regr], f)

        elif episode == NUMBER_EPISODES-1: # dump results out at the end of the last episode
            filename = 'output/CONTEXTUAL_opsrl' + str(RUN_NUMBER) + '.pkl'
            f = open(filename, 'ab')
            pickle.dump([R_est_err, C_est_err, min_eign_hba1c_list, min_eign_cvd_list, NUMBER_SIMULATIONS, NUMBER_EPISODES, objs , cons, pi_k, NUMBER_INFEASIBILITIES, q_k], f)
            f.close()
        
            filename = 'output/CONTEXTUAL_BG_regr.pkl'
            with open(filename, 'wb') as f:
                pickle.dump([util_methods.hba1c_regr, util_methods.cvdrisk_regr], f)        

        episode += 1
        
# take average/std over multiple simulation runs
ObjRegret_mean = np.mean(ObjRegret2, axis = 0) 
ConRegret_mean = np.mean(ConRegret2, axis = 0)
ObjRegret_std = np.std(ObjRegret2, axis = 0)
ConRegret_std = np.std(ConRegret2, axis = 0)

# save the results as a pickle file
filename = 'output/CONTEXTUAL_regrets_' + str(RUN_NUMBER) + '.pkl'
with open(filename, 'wb') as f:
    pickle.dump([NUMBER_SIMULATIONS, NUMBER_EPISODES, ObjRegret_mean, ObjRegret_std, ConRegret_mean, ConRegret_std], f)