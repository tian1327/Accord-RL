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


"""
This script is used to generate:
1. The optimal policy/value/cost/q under constrained/unconstrained for the inventory control Problem
2. The policy/value/cost/q for safe base case
3. The P, R, C, N_STATES, actions, EPISODE_LENGTH, DELTA for running DOPE
"""

K = 4
def h(x): # holding cost
    return x

def f(x): # revenue function
    if x > 0:
        return 8*x/100 # why devide by 100?
    return 0

def O(x): # purchase cost
    if x > 0:
        return K + 2*x # k is the fixed cost, 2 is the unit cost, x is the number of units purchased
    return 0

N_STATES = 6 # number of states, max capacity of inventory, should this actually be 7? This implementation is 0-5, but the paper mentions 0-6
actions = {} # actions available in each state

delta = 0.01 # bound


R = {} # dictionary of reward matrices
C = {} # dictionary of cost matrices
P = {} # dictionary of transition probability matrices

demand = [0.3,0.2,0.2,0.2,0.05,0.05] # probability of stochastic demand, dh in {0, 1, 2, 3, 4, 5, 6}, missing 1 probability!!

for s in range(N_STATES): # 0-5
    actions[s] = []
    for a in range(N_STATES - s): # shouldn't this be range(N_STATES - s + 1)?
        actions[s].append(a) # initialize available actions for each state

for s in range(N_STATES):
    l = len(actions[s])
    R[s] = np.zeros(l)
    C[s] = np.zeros(l)
    P[s] = {}
    for a in actions[s]:
        C[s][a] = O(a) + h(s+a) # cost of taking action a in state s = order cost + holding cost
        P[s][a] = np.zeros(N_STATES) # transition probability matrix
        for d in range(N_STATES): # Assume N_STATES = 7, d = 0,1,2,3,4,5,6
            s_ = s + a - d # next state
            if s_ < 0:
                s_ = 0
            elif s_ > N_STATES - 1:
                s_ = N_STATES - 1 # make sure next state is within bounds [0, N_STATES-1]
                
            P[s][a][s_] += demand[d] # assign transition probability based on demand probability
        R[s][a] = 0
        # for s_ in range(N_STATES):
        #     R[s][a] += P[s][a][s_]*f(s_)
        
for s in range(N_STATES):
    for a in actions[s]:        
      for d in range(N_STATES):
            s_ = min(max(0,s+a-d),N_STATES-1)
            if s + a - d >= 0:
                R[s][a] += P[s][a][s_]*f(d) # probability of demand d * revenue from demand d = expected revenue
            else:
                R[s][a] += 0
            


r_max = R[0][0]
c_max = C[0][0]
print("P")
pp.pprint(P)


for s in range(N_STATES):
    for a in actions[s]:
        if C[s][a] > c_max:
            c_max = C[s][a]
        if R[s][a] > r_max:
            r_max = R[s][a]

print("r_max =", r_max)
print("c_max =", c_max)

# normalize rewards and costs to be between 0 and 1
for s in range(N_STATES):
    for a in actions[s]:
        C[s][a] = C[s][a]/c_max
        R[s][a] = R[s][a]/r_max

EPISODE_LENGTH = 7

CONSTRAINT = EPISODE_LENGTH/2

C_b = CONSTRAINT/5  #Change this if you want different baseline policy. here is 0.2C

# NUMBER_EPISODES = 1e6
NUMBER_EPISODES = 3e5

NUMBER_SIMULATIONS = 1


EPS = 0.01 # not used
M = 0 # not used

util_methods_1 = utils(EPS, delta, M, P,R,C,EPISODE_LENGTH,N_STATES,actions,CONSTRAINT,C_b)
opt_policy_con, opt_value_LP_con, opt_cost_LP_con, opt_q_con = util_methods_1.compute_opt_LP_Constrained(0) # constrained MDP
opt_policy_uncon, opt_value_LP_uncon, opt_cost_LP_uncon, opt_q_uncon = util_methods_1.compute_opt_LP_Unconstrained(0) # unconstrained = standard MDP, not used in DOPE
f = open('solution-in.pckl', 'wb')
pickle.dump([opt_policy_con, opt_value_LP_con, opt_cost_LP_con, opt_q_con, opt_policy_uncon, opt_value_LP_uncon, opt_cost_LP_uncon, opt_q_uncon], f)
f.close()


util_methods_1 = utils(EPS, delta, M, P,R,C,EPISODE_LENGTH,N_STATES,actions,C_b,C_b)
policy_b, value_b, cost_b, q_b = util_methods_1.compute_opt_LP_Constrained(0)
f = open('base-in.pckl', 'wb')
pickle.dump([policy_b, value_b, cost_b, q_b], f)
f.close()


f = open('model-in.pckl', 'wb')
pickle.dump([NUMBER_SIMULATIONS, NUMBER_EPISODES, P, R, C, CONSTRAINT, N_STATES, actions, EPISODE_LENGTH, delta], f)
f.close()

print('\n*******')
print("opt_value_LP_uncon[0, 0] =",opt_value_LP_uncon[0, 0])
print("opt_value_LP_con[0, 0] =",opt_value_LP_con[0, 0])
print("value_b[0, 0] =",value_b[0, 0])