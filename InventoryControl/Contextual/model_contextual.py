import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from UtilityMethods import utils
import sys
import pickle
import time
import pulp as p
import math
from copy import copy
import pprint as pp
from tqdm import tqdm
import pprint as pp
import os

# stop
# state space, actions available in each state vary with state

# calculate the demand d as a dot product of X and true_theta + noise
def calculate_demand(X, theta):
    # return np.dot(X, theta) + np.random.choice([-2, -1, 0, 1, 2], 1)
    return np.dot(X, theta) + np.random.choice([0, 1], 1)


def cal_d_prob(X, true_theta):
    
    N = X.shape[0]
    # print('N:', N)
    d_list = []
    for i in range(N):
        d = calculate_demand(X[i, :], true_theta)
        d_list.append(int(d[0]))

    # print(d_list[:20])
    d_set = set(d_list)
    # print(f'd_set: {d_set}')

    # calculate the occurance of demand d
    d_prob = {}
    for d in d_list:
        if d in d_prob:
            d_prob[d] += 1
        else:
            d_prob[d] = 1

    for k, v in d_prob.items():
        d_prob[k] = v / len(d_list)
        # print(f'P(d={k}): {d_prob[k]}')
    
    return d_prob

def h(x): # holding cost
    return x

def f(x): # revenue function
    if x > 0:
        return 8*x/100 # why devide by 100?
    return 0

def O(x): # purchase cost
    if x > 0:
        return 4 + 2*x # k is the fixed cost, 2 is the unit cost, x is the number of units purchased
    return 0


if __name__ == '__main__':

    N_STATES = 7

    # build the action space for each state
    ACTIONS_PER_STATE = {}
    for s in range(N_STATES):
        ACTIONS_PER_STATE[s] = []
        for a in range(N_STATES-s):
            ACTIONS_PER_STATE[s].append(a) 
    print('Actions for State 0:', ACTIONS_PER_STATE[0])
    print('Actions for State 1:', ACTIONS_PER_STATE[1])
    print('Actions for State 6:', ACTIONS_PER_STATE[6])

    # calculatet the distribution probability of demand d 
    true_theta = np.array([1, 2, 2, 1, 
                        -1, -2, -2, -1])

    # create an numpy array of 5000x10, where each row has 10 demand values
    # the first 5 values are positive which are sampled uniformly from the set {0, 1}
    # the next 5 values are negative which are sampled uniformly from the set {0, -1}
    def sample_X(n_samples=5000):
        X = np.zeros((n_samples, 8))
        for i in range(n_samples):
            X[i, :4] = np.random.choice([0, 1], 4)
            X[i, 4:] = np.random.choice([0, -1], 4)
        return X

    X = sample_X()
    print(X.shape)
    print(X[:5, :])

    d_prob = cal_d_prob(X, true_theta)

    R = {} # dictionary of reward matrices
    C = {} # dictionary of cost matrices
    P = {} # dictionary of transition probability matrices

    # calculate the P matrix
    for s in range(N_STATES):
        l = len(ACTIONS_PER_STATE[s])
        R[s] = np.zeros(l)
        C[s] = np.zeros(l)
        P[s] = {}
        for a in ACTIONS_PER_STATE[s]:
            C[s][a] = O(a) + h(s+a) # cost of taking action a in state s = order cost + holding cost, why is h(s+a) instead of h(s)?
            P[s][a] = np.zeros(N_STATES) # transition probability matrix
            for d, prob in d_prob.items(): 
                if d < 0: # handle the negative demand
                    d = 0
                s_ = s + a - d # next state
                if s_ < 0:
                    s_ = 0
                elif s_ > N_STATES - 1:
                    s_ = N_STATES - 1 # make sure next state is within bounds [0, N_STATES-1]
                    
                P[s][a][s_] += prob # assign transition probability based on demand probability
                
            R[s][a] = 0

    # fill in the R matrix
    for s in range(N_STATES):
        for a in ACTIONS_PER_STATE[s]:        
            for d, prob in d_prob.items():
                s_ = min(max(0, s+a-d), N_STATES-1)
                if s + a - d >= 0:
                    R[s][a] += P[s][a][s_]*f(d) # probability of demand d * revenue from demand d = expected revenue
                else:
                    R[s][a] += 0

    r_max = R[0][0]
    c_max = C[0][0]

    for s in range(N_STATES):
        for a in ACTIONS_PER_STATE[s]:
            if C[s][a] > c_max:
                c_max = C[s][a]
            if R[s][a] > r_max:
                r_max = R[s][a]

    print("r_max =", r_max)
    print("c_max =", c_max)

    # normalize rewards and costs to be between 0 and 1
    for s in range(N_STATES):
        for a in ACTIONS_PER_STATE[s]:
            C[s][a] = C[s][a]/c_max
            R[s][a] = R[s][a]/r_max

    EPISODE_LENGTH = 7
    CONSTRAINT = EPISODE_LENGTH/2
    C_b = CONSTRAINT/5 

    EPS = 0.01
    M = 0
    delta = 0.01

    # create output directory if it does not exist
    if not os.path.exists('output'):
        os.makedirs('output')

    # dump the model settings and parameters to a pickle file
    CONTEXT_VEC_LENGTH = 8
    ACTION_CODE_LENGTH = 1
    INIT_STATE_INDEX = 0
    print('CONTEXT_VEC_LENGTH =', CONTEXT_VEC_LENGTH)
    print('ACTION_CODE_LENGTH =', ACTION_CODE_LENGTH)

    with open('output/model_contextual.pkl', 'wb') as f:
        pickle.dump([P, CONTEXT_VEC_LENGTH, ACTION_CODE_LENGTH, INIT_STATE_INDEX, true_theta, 
                    CONSTRAINT, C_b, N_STATES, ACTIONS_PER_STATE, EPISODE_LENGTH, delta], f)

    # # constrained and unconstrained optimal solution
    # util_methods_1 = utils(EPS, delta, M, P, R, C, EPISODE_LENGTH, N_STATES, ACTIONS_PER_STATE, CONSTRAINT, C_b)
    # opt_policy_con, opt_value_LP_con, opt_cost_LP_con, opt_q_con = util_methods_1.compute_opt_LP_Constrained(0) # constrained MDP
    # opt_policy_uncon, opt_value_LP_uncon, opt_cost_LP_uncon, opt_q_uncon = util_methods_1.compute_opt_LP_Unconstrained(0) # unconstrained = standard MDP, not used in DOPE
    # f = open('output/solution-in.pckl', 'wb')
    # pickle.dump([opt_policy_con, opt_value_LP_con, opt_cost_LP_con, opt_q_con, opt_policy_uncon, opt_value_LP_uncon, opt_cost_LP_uncon, opt_q_uncon], f)
    # f.close()

    # # base solution
    # util_methods_1 = utils(EPS, delta, M, P, R , C, EPISODE_LENGTH, N_STATES, ACTIONS_PER_STATE, C_b, C_b)
    # policy_b, value_b, cost_b, q_b = util_methods_1.compute_opt_LP_Constrained(0)
    # f = open('output/base-in.pckl', 'wb')
    # pickle.dump([policy_b, value_b, cost_b, q_b], f)
    # f.close()

    # f = open('output/model-in.pckl', 'wb')
    # pickle.dump([NUMBER_SIMULATIONS, NUMBER_EPISODES, P, R, C, CONSTRAINT, N_STATES, ACTIONS_PER_STATE, EPISODE_LENGTH, delta], f)
    # f.close()

    # print('\n*******')
    # print("opt_value_LP_uncon[0, 0] =",opt_value_LP_uncon[0, 0])
    # print("opt_value_LP_con[0, 0] =",opt_value_LP_con[0, 0])
    # print("value_b[0, 0] =",value_b[0, 0])