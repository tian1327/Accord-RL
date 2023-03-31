#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 16:00:01 2021

@author: Anonymous
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from UtilityMethods import utils
import sys
#import gym
import pickle
import time
import pulp as p
import math
from copy import copy
import pprint as pp
import itertools
from tqdm import tqdm


"""
This script is used to generate:
1. The optimal policy/value/cost/q under constrained/unconstrained for the inventory control Problem
2. The policy/value/cost/q for safe base case
3. The P, R, C, N_STATES, actions, EPISODE_LENGTH, DELTA for running DOPE
"""

# Global variables

IS_VISIT_DEPENDENT = False # whether the above empirical estimates are visit-dependent or not
DATA = '../../../../Codes/Accord/data/ACCORD_BPClass_v2.csv'
EPISODE_LENGTH = 7

delta = 0.01 # bound


EPS = 0.01 # not used
M = 0 # not used


CONSTRAINT = EPISODE_LENGTH/2

C_b = CONSTRAINT/5  #Change this if you want different baseline policy. here is 0.2C

NUMBER_EPISODES = 1e4
NUMBER_SIMULATIONS = 1

# Functions

# define a function to calculate the empirical estimate of R, C, P
def calculate_empirical_estimate_PRC(DATA, IS_VISIT_DEPENDENT):

    df = pd.read_csv(DATA)
    
    # add the state and action code columns
    action_code = []
    state_code = []
    for i in range(len(df)):
        row = df.iloc[i]
        s_code = ''
        a_code = ''
        for state_fea in state_features:
            s_code += str(row[state_fea])
        
        for action_fea in action_features:
            a_code += str(row[action_fea])
        
        action_code.append(a_code)
        state_code.append(s_code)
    
    df['action_code'] = action_code
    df['state_code'] = state_code
    print('Finished adding action_code and state_code columns')

    #------------- calculate the empirical estimate of P, R, C

    # initialize R, C, P
    R = {} # dictionary of reward matrices, this is the CVDRisk empirical estimate based on entire dataset
    C = {} # dictionary of cost matrices, this is SBP empirical estimate based on entire dataset
    P = {} # dictionary of transition probability matrices, based on the entire dataset

    for s in range(N_STATES):
        l = len(actions)
        R[s] = np.zeros(l)
        C[s] = np.zeros(l)
        P[s] = {}    
        for a in range(N_ACTIONS):
            C[s][a] = 0
            P[s][a] = np.zeros(N_STATES)
            R[s][a] = 0
    print('Finished initializing R, C, P')
         
    count_s_a = {} # count the number of times state s and action a appear in the dataset
    count_s_a_d = {} # count the number of times state s, action a, and next state s' appear in the dataset
    sum_r_s_a = {} # sum of the reward of state s and action a
    sum_c_s_a = {} # sum of the cost of state s and action a

    # loop through each patient in the dataset
    for i in tqdm(range(100001, 110252)):
        df_patient = df[df['MaskID'] == i]

        # loop through each visit of the patient
        for j in range(len(df_patient)-1): # loop before last visit
            row = df_patient.iloc[j]
            s_code = row['state_code']
            a_code = row['action_code']
            ns_code = df_patient.iloc[j+1]['state_code']

            # convert from code to index
            s = state_code_to_index[s_code]
            a = action_code_to_index[a_code]
            s_ = state_code_to_index[ns_code]

            r = df_patient.iloc[j]['CVDRisk_feedback']
            c = df_patient.iloc[j]['sbp_feedback']
    
            if (s, a) not in count_s_a:
                count_s_a[(s, a)] = 1
                sum_r_s_a[(s, a)] = r 
                sum_c_s_a[(s, a)] = c
            else:
                count_s_a[(s, a)] += 1
                sum_r_s_a[(s, a)] += r
                sum_c_s_a[(s, a)] += c

            if (s, a, s_) not in count_s_a_d:
                count_s_a_d[(s, a, s_)] = 1
            else:
                count_s_a_d[(s, a, s_)] += 1
    print('Finished counting by looping through the dataset')

    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            if (s, a) in count_s_a:
                R[s][a] = sum_r_s_a[(s, a)]/max(count_s_a[(s, a)],1)
                C[s][a] = sum_c_s_a[(s, a)]/max(count_s_a[(s, a)],1)
            
            for s_ in range(N_STATES):
                if (s, a, s_) in count_s_a_d:
                    P[s][a][s_] = count_s_a_d[(s, a, s_)]/max(count_s_a[(s, a)],1)
    
    print('Finished calculating the empirical estimate of P, R, C')
    
    #------------- check the sparsity of P, R, C
    print('\nSparsity of P, R, C:')

    # count how many zeros in P
    count = 0
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            for s_ in range(N_STATES):
                if P[s][a][s_] == 0:
                    count += 1
    print('Number of zeros in P: ', count, 'percentage: ', count/(N_STATES*N_ACTIONS*N_STATES))

    # count how many zeros in R
    count = 0
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            if R[s][a] == 0:
                count += 1
    print('Number of zeros in R: ', count, 'percentage: ', count/(N_STATES*N_ACTIONS))

    # count how many zeros in C
    count = 0
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            if C[s][a] == 0:
                count += 1
    print('Number of zeros in C: ', count, 'percentage: ', count/(N_STATES*N_ACTIONS))


    # do we need to handle the sparsity of the matrices?
    
    return P, R, C
    

# state space, actions available in each state are always the same
state_features = ['sbp_discrete','hba1c_discrete','TC_discrete','hdl_discrete','BMI_discrete',] # might need to remove 'BMI_discrete'
fea1 = ['0', '1', '2', '3'] # possible values for sbp_discrete
fea2 = ['0', '1', '2', '3', '4', '5', '6', '7']
fea3 = ['0', '1', '2', '3']
fea4 = ['0', '1', '2', '3']
fea5 = ['0', '1', '2', '3']

combinations = itertools.product(fea1, fea2, fea3, fea4, fea5)
states = [''.join(i) for i in combinations]
print('len(states) =', len(states))
# print(states[-5:])
N_STATES = len(states) # number of states = 2048
state_code_to_index = {code: i for i, code in enumerate(states)}


# action space, 000000000 means bpclass_none, 111111111 means all bpmed class are precribed
action_features = ['Diur', 'ACE', 'Beta-blocker', 'CCB', 'ARB', 
                    'Alpha-Beta-blocker', 'Alpha-blocker', 'Sympath', 'Vasod'] # we donot include 'bpclass_none' as a action, because 000000000 means bpclass_none

combinations = list(itertools.product('01', repeat=len(action_features)))
actions = [''.join(i) for i in combinations]
print('len(actions) =', len(actions))
N_ACTIONS = len(actions) # number of actions = 512
action_code_to_index = {code: i for i, code in enumerate(actions)}


# P R C
P, R, C = calculate_empirical_estimate_PRC(DATA, IS_VISIT_DEPENDENT)            

# normalize rewards and costs to be between 0 and 1
r_max = R[0][0]
c_max = C[0][0]

for s in range(N_STATES):
    for a in range(N_ACTIONS):
        if C[s][a] > c_max:
            c_max = C[s][a]
        if R[s][a] > r_max:
            r_max = R[s][a]

print("r_max =", r_max)
print("c_max =", c_max)

for s in range(N_STATES):
    for a in range(N_ACTIONS):
        C[s][a] = C[s][a]/c_max
        R[s][a] = R[s][a]/r_max

print('Finnished normalizing rewards and costs')


util_methods_1 = utils(EPS, delta, M, P, R, C, EPISODE_LENGTH, N_STATES, actions, CONSTRAINT, C_b)
opt_policy_con, opt_value_LP_con, opt_cost_LP_con, opt_q_con = util_methods_1.compute_opt_LP_Constrained(0) # constrained MDP
opt_policy_uncon, opt_value_LP_uncon, opt_cost_LP_uncon, opt_q_uncon = util_methods_1.compute_opt_LP_Unconstrained(0) # unconstrained = standard MDP, not used in DOPE
f = open('solution.pckl', 'wb')
pickle.dump([opt_policy_con, opt_value_LP_con, opt_cost_LP_con, opt_q_con, opt_policy_uncon, opt_value_LP_uncon, opt_cost_LP_uncon, opt_q_uncon], f)
f.close()


util_methods_1 = utils(EPS, delta, M, P,R,C,EPISODE_LENGTH,N_STATES,actions,C_b,C_b)
policy_b, value_b, cost_b, q_b = util_methods_1.compute_opt_LP_Constrained(0)
f = open('base.pckl', 'wb')
pickle.dump([policy_b, value_b, cost_b, q_b], f)
f.close()


f = open('model.pckl', 'wb')
pickle.dump([NUMBER_SIMULATIONS, NUMBER_EPISODES, P, R, C, CONSTRAINT, N_STATES, actions, EPISODE_LENGTH, delta], f)
f.close()

print('\n*******')
print("opt_value_LP_uncon[0, 0] =",opt_value_LP_uncon[0, 0])
print("opt_value_LP_con[0, 0] =",opt_value_LP_con[0, 0])
print("value_b[0, 0] =",value_b[0, 0])

