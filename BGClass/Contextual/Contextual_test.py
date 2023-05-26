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

sbp_discrete_code_dict = {'0': '0', '1': '1',
                          '2': '2', '3': '2',}

hba1c_discrete_code_dict = {'0': '0', '1': '0', 
                            '2': '1', '3': '1', 
                            '4': '2', '5': '2', 
                            '6': '2', '7': '2'}

context_fea = ['baseline_age', 'female', 'race_whiteother',
               'edu_baseline_1', 'edu_baseline_2', 'edu_baseline_3',
               'cvd_hx_baseline', 'baseline_BMI', 'cigarett_baseline_1',
               ]

action_features = ['Bingu', 'Thiaz', 'Sulfon', 'Meglit' ] 
action_to_med_list = {}

action_to_med_list['0000'] = ['BGClass_None']
action_to_med_list['1000'] = ['Bingu']
action_to_med_list['0100'] = ['Thiaz']
action_to_med_list['0010'] = ['Sulfon']
action_to_med_list['0001'] = ['Meglit']
action_to_med_list['1100'] = ['Bingu', 'Thiaz']
action_to_med_list['1010'] = ['Bingu', 'Sulfon']
action_to_med_list['1001'] = ['Bingu', 'Meglit']
action_to_med_list['0110'] = ['Thiaz', 'Sulfon']
action_to_med_list['0101'] = ['Thiaz', 'Meglit']
action_to_med_list['0011'] = ['Sulfon', 'Meglit']
action_to_med_list['1110'] = ['Bingu', 'Thiaz', 'Sulfon']
action_to_med_list['1101'] = ['Bingu', 'Thiaz', 'Meglit']
action_to_med_list['1011'] = ['Bingu', 'Sulfon', 'Meglit']
action_to_med_list['0111'] = ['Thiaz', 'Sulfon', 'Meglit']
action_to_med_list['1111'] = ['Bingu', 'Thiaz', 'Sulfon', 'Meglit']

#------------------------------------------------------
# control parameters

RUN_NUMBER = 150 #Change this field to set the seed for the experiment.

use_gurobi = False # whether to use gurobi to solve the optimization problem
NUMBER_SIMULATIONS = 1

if len(sys.argv) > 1:
    use_gurobi = sys.argv[1]

#--------------------------------------------------------
random.seed(RUN_NUMBER)
np.random.seed(RUN_NUMBER)

# Initialize:
with open('output_final2/model_contextual_BG.pkl', 'rb') as f:
    [P, CONTEXT_VEC_LENGTH, ACTION_CODE_LENGTH, CONTEXT_VECTOR_dict, INIT_STATE_INDEX, INIT_STATES_LIST, 
    state_code_to_index, state_index_to_code, action_index_to_code,
    CONSTRAINT, C_b, N_STATES, N_ACTIONS, ACTIONS_PER_STATE, EPISODE_LENGTH, DELTA] = pickle.load(f)

STATE_CODE_LENGTH = len(state_index_to_code[0])
print("STATE_CODE_LENGTH =", STATE_CODE_LENGTH)

# load the offline trained true model: CVDRisk_estimator and SBP_feedback_estimator from pickle file
R_model = pickle.load(open('output_final2/CVDRisk_estimator_BG.pkl', 'rb'))
C_model = pickle.load(open('output_final2/A1C_feedback_estimator_BG.pkl', 'rb'))

# load the estimated hba1c and CVDRisk models from pickle file
filename = 'output_final2/CONTEXTUAL_BG_regr.pkl'
with open(filename, 'rb') as f:
    [hba1c_regr, cvdrisk_regr] = pickle.load(f)

Cb = C_b[-1]
#Cb = 150

CONSTRAINT = CONSTRAINT[-1]

print("CONSTRAINT =", CONSTRAINT)
print("Cb =", Cb)
print("CONSTRAINT - Cb =", CONSTRAINT - Cb)
print("N_STATES =", N_STATES)
print("N_ACTIONS =", N_ACTIONS)

NUMBER_SIMULATIONS = int(NUMBER_SIMULATIONS)
ACTIONS = np.arange(N_ACTIONS)

# not used here
EPS = 0
M = 0

#--------------------------------------- Run the simulation based on learned COPS model
print("\nRun the simulation based on learned COPS model")
for sim in range(NUMBER_SIMULATIONS):

    util_methods = utils(EPS, DELTA, M, P, R_model, C_model, CONTEXT_VEC_LENGTH, ACTION_CODE_LENGTH, STATE_CODE_LENGTH,
                         INIT_STATE_INDEX, state_index_to_code, action_index_to_code,
                         EPISODE_LENGTH, N_STATES, N_ACTIONS, ACTIONS_PER_STATE, CONSTRAINT, Cb, use_gurobi) 

    util_methods.set_regr(hba1c_regr, cvdrisk_regr)

    # for logistic regression of CVDRisk_feedback and linear regression SBP_feedback
    maskid_full =[]
    visit_num_full = []
    context_vec_dict_full = {}
    for fea in context_fea:
        context_vec_dict_full[fea] = []

    state_code_full = []
    action_code_full = [] # record the action code for each step in a episode
    med_list_full = [] # record the med for each step in a episode
    sbp_cont_full = [] # record the SBP feedback continuous for each step in a episode
    hba1c_cont_full = [] # record the HbA1c feedback continuous for each step in a episode
    cvdrisk_full = [] # record the CVDRisk for each step in a episode

    patient_list = list(CONTEXT_VECTOR_dict.keys())
    print("patient_list[0] =", patient_list[0])
    print("len(patient_list) =", len(patient_list))

    ct = 0
    for patient in tqdm(patient_list):

        # if ct >2:
        #     break

        ct += 1

        # print("patient =", patient)
        maskid_full.extend([patient] * EPISODE_LENGTH)        

        context_vec = CONTEXT_VECTOR_dict[patient][0]
        # print("context_vec =", context_vec)
        util_methods.set_context(context_vec) # set the context vector for the current episode

        s_idx_init = int(CONTEXT_VECTOR_dict[patient][1])

        util_methods.update_mu(s_idx_init)
        # print("s_idx_init =", s_idx_init)

        # calculate the R and C based on the true R and C models, regenerate for each episode/patient
        util_methods.calculate_true_R_C(context_vec)

        # calculate the R and C based on the estimated R and C models, regenerate for each episode/patient_hat 
        util_methods.calculate_est_R_C()
        # Notice here, we should have used the empirical estimates of P after learning for 3e4 episodes, but we use the true P here for simplicity
        # the true P and estimated P after 3e4 episodes are very close, so it should not make a big difference
        util_methods.set_P_hat(P) 
        
        # compute the optimal policy using the learned R_hat and C_hat, and P_hat, all confidence intervals are 0 
        pi_k, val_k, cost_k, log, q_k = util_methods.compute_extended_LP() 
         
        # reset the data collector
        visit_num = []
        ep_context_vec = context_vec # record the context vector for the current episode
        ep_sbp_cont = [] # record the SBP feedback continuous for each step in a episode
        ep_hba1c_cont = [] # record the HbA1c feedback continuous for each step in a episode
        a_list = []
        ep_action_code = [] # record the action code for each step in a episode
        ep_med_list = []
        ep_state_code = []
        ep_cvdrisk = [] # record the CVDRisk for each step in a episode  
        
        s = s_idx_init # set the state to the initial state
        for h in range(EPISODE_LENGTH): # for each step in current episode
            
            visit_num.append(h)

            # add the context vector to the data collector
            for i, fea in enumerate(context_fea):
                context_vec_dict_full[fea].append(ep_context_vec[i])
            

            prob = pi_k[s, h, :]           
            a = int(np.random.choice(ACTIONS, 1, replace = True, p = prob)) # select action based on the policy/probability

            a_list.append(a)
            action_code = action_index_to_code[a]
            ep_action_code.append(action_code)
            med_lst = action_to_med_list[action_code]
            ep_med_list.append('+'.join(med_lst))
            ep_state_code.append(state_index_to_code[s])

            next_state, rew, cost = util_methods.step(s, a, h, False) # take the action and get the next state, reward and cost

            ep_sbp_cont.append(-1)
            ep_hba1c_cont.append(cost) 
            ep_cvdrisk.append(rew)

            s = next_state

        # after EPISODE_LENGTH, write the collected data per episodes to file output_final/Contextual_test_BPClass.csv
        visit_num_full.extend(visit_num)
        state_code_full.extend(ep_state_code)
        action_code_full.extend(ep_action_code)
        med_list_full.extend(ep_med_list)
        sbp_cont_full.extend(ep_sbp_cont)
        hba1c_cont_full.extend(ep_hba1c_cont)
        cvdrisk_full.extend(ep_cvdrisk)


    # after looping through all patients, write the collected data to file output_final/Contextual_test_BPClass.csv
    df = pd.DataFrame({'MaskId': maskid_full, 'Visit_num': visit_num_full})
    for fea in context_fea:
        df[fea] = context_vec_dict_full[fea]
    df['state_code'] = state_code_full
    df['action_code'] = action_code_full
    df['med_list'] = med_list_full
    df['sbp_fb'] = sbp_cont_full
    df['hba1c_fb'] = hba1c_cont_full
    df['cvdrisk_fb'] = cvdrisk_full
    # print('df.info() =', df.info())

    # df.to_csv('output_final/Contextual_test_BPClass.csv', index=False)


#--------------------------------------- Run the simulation for Clinician 
print("\nRun the simulation for Clinician")

# load knn model and scalar model from output_final/knn_model.pkl and output_final/scaler_model.pkl
knn_model = pickle.load(open('output_final2/knn_model.pkl', 'rb'))
labels = pickle.load(open('output_final2/knn_model_label.pkl', 'rb'))
scaler_model = pickle.load(open('output_final2/scaler_model.pkl', 'rb'))

for sim in range(NUMBER_SIMULATIONS):

    util_methods = utils(EPS, DELTA, M, P, R_model, C_model, CONTEXT_VEC_LENGTH, ACTION_CODE_LENGTH, STATE_CODE_LENGTH,
                         INIT_STATE_INDEX, state_index_to_code, action_index_to_code,
                         EPISODE_LENGTH, N_STATES, N_ACTIONS, ACTIONS_PER_STATE, CONSTRAINT, Cb, use_gurobi) 

    util_methods.set_regr(hba1c_regr, cvdrisk_regr)

    # for logistic regression of CVDRisk_feedback and linear regression SBP_feedback
    maskid_full =[]
    visit_num_full = []
    context_vec_dict_full = {}
    for fea in context_fea:
        context_vec_dict_full[fea] = []

    state_code_full = []
    action_code_full = [] # record the action code for each step in a episode
    med_list_full = [] # record the med for each step in a episode
    sbp_cont_full = [] # record the SBP feedback continuous for each step in a episode
    hba1c_cont_full = [] # record the HbA1c feedback continuous for each step in a episode
    cvdrisk_full = [] # record the CVDRisk for each step in a episode

    patient_list = list(CONTEXT_VECTOR_dict.keys())
    print("patient_list[0] =", patient_list[0])
    print("len(patient_list) =", len(patient_list))

    ct = 0
    for patient in tqdm(patient_list):

        # if ct >2:
        #     break

        ct += 1

        # print("\n +++++patient =", patient)
        maskid_full.extend([patient] * EPISODE_LENGTH)        

        context_vec = CONTEXT_VECTOR_dict[patient][0]
        # print("context_vec =", context_vec)
        util_methods.set_context(context_vec) # set the context vector for the current episode

        s_idx_init = int(CONTEXT_VECTOR_dict[patient][1])

        util_methods.update_mu(s_idx_init)
        # print("s_idx_init =", s_idx_init)

        state_index_recorded = CONTEXT_VECTOR_dict[patient][2]
        action_index_recorded = CONTEXT_VECTOR_dict[patient][3]


        # calculate the R and C based on the true R and C models, regenerate for each episode/patient
        util_methods.calculate_true_R_C(context_vec)
         
        # reset the data collector
        visit_num = []
        ep_context_vec = context_vec # record the context vector for the current episode
        ep_sbp_cont = [] # record the SBP feedback continuous for each step in a episode
        ep_hba1c_cont = [] # record the HbA1c feedback continuous for each step in a episode
        a_list = []
        ep_action_code = [] # record the action code for each step in a episode
        ep_med_list = []
        ep_state_code = []
        ep_cvdrisk = [] # record the CVDRisk for each step in a episode  
        
        s = s_idx_init # set the state to the initial state
        for h in range(EPISODE_LENGTH): # for each step in current episode
            # print('h =', h)
            visit_num.append(h)

            # add the context vector to the data collector
            for i, fea in enumerate(context_fea):
                context_vec_dict_full[fea].append(ep_context_vec[i])
            
            # here we select the action to be taken by the clinician
            # 1. if the visit number is 0, we follow the same action taken by the clinician in the first visit
            # 2. for the following visit, if the state observed is different from the state in the recorded raw data,
            #    we will find the 5 nearest neighbors defined by the (context_vec, state) vector, and randomly select one action from the 5 actions

            if h == 0: # follow clinician's action in the first visit
                if s != state_index_recorded[h]:
                    raise ValueError('The initial state is different from the state in the recorded raw data')

                a = action_index_recorded[h]
            else:
                if h >= len(state_index_recorded) or  s != state_index_recorded[h]:
                    # find the 5 nearest neighbors defined by the (context_vec, state) vector
                    # randomly select one action from the 5 actions

                    target_vector = np.concatenate((context_vec, [s]))
                    # print("target_vector =", target_vector)
                    normalized_target_vector = scaler_model.transform([target_vector])
                    # print("normalized_target_vector =", normalized_target_vector)
                    distances, indices = knn_model.kneighbors(normalized_target_vector)
                    # print("distances =", distances)
                    # print("indices =", indices)
                    actions_NN = [labels[i] for i in indices[0]]
                    # print("actions_NN =", actions_NN)

                    # sample the actons based on the distance, the closer the distance, the higher the probability
                    # calculate the sampling probability based on the distance, the closer the distance, the higher the probability
                    # the probability is proportional to 1/distance
                    # normalize the probability to sum to 1

                    # for each of the distance, take the max of distance and 0.001
                    # this is to avoid the case when the distance is 0, which will cause the probability to be infinity
                    dist = np.maximum(distances[0], 0.0001)
                    # print("dist =", dist)
                    weights = 1/dist
                    prob = weights/np.sum(weights)
                    # print("prob =", prob)
                    a= int(np.random.choice(actions_NN, 1, replace = True, p=prob))

                else:
                    a = action_index_recorded[h]

            a_list.append(a)
            action_code = action_index_to_code[a]
            ep_action_code.append(action_code)
            med_lst = action_to_med_list[action_code]
            ep_med_list.append('+'.join(med_lst))
            ep_state_code.append(state_index_to_code[s])

            next_state, rew, cost = util_methods.step(s, a, h, False) # take the action and get the next state, reward and cost

            ep_sbp_cont.append(-1)
            ep_hba1c_cont.append(cost) 
            ep_cvdrisk.append(rew)

            s = next_state

        # after EPISODE_LENGTH, write the collected data per episodes to file output_final/Contextual_test_BPClass.csv
        visit_num_full.extend(visit_num)
        state_code_full.extend(ep_state_code)
        action_code_full.extend(ep_action_code)
        med_list_full.extend(ep_med_list)
        sbp_cont_full.extend(ep_sbp_cont)
        hba1c_cont_full.extend(ep_hba1c_cont)
        cvdrisk_full.extend(ep_cvdrisk)


    # after looping through all patients, write the collected data to file output_final/Contextual_test_BPClass.csv
    df['state_code_cln'] = state_code_full
    df['action_code_cln'] = action_code_full
    df['med_list_cln'] = med_list_full
    df['sbp_fb_cln'] = sbp_cont_full
    df['hba1c_fb_cln'] = hba1c_cont_full
    df['cvdrisk_fb_cln'] = cvdrisk_full


    df.to_csv('output_final2/Contextual_test_BGClass.csv', index=False)
    print('Results saved to output_final2/Contextual_test_BGClass.csv')