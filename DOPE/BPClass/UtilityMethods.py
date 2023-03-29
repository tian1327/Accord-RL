import numpy as np
import pulp as p
import time
import math
import sys

class utils:
    def __init__(self,eps, delta, M, P,R,C,EPISODE_LENGTH,N_STATES,ACTIONS,CONSTRAINT,Cb):
        self.P = P.copy()
        self.R = R.copy()
        self.C = C.copy()
        self.EPISODE_LENGTH = EPISODE_LENGTH
        self.N_STATES = N_STATES
        self.ACTIONS = ACTIONS
        self.eps = eps
        self.delta = delta
        self.M = M
        self.Cb = Cb
        #self.ENV_Q_VALUES = np.zeros((self.N_STATES,self.EPISODE_LENGTH,self.N_ACTIONS))
        
        self.P_hat = {}#np.zeros((self.N_STATES,self.N_ACTIONS,self.N_STATES))
        #self.P_tilde = np.zeros((self.N_STATES,self.N_ACTIONS,self.N_STATES))
        
        
        #self.R_hat = np.zeros((self.N_STATES,self.N_ACTIONS))
        #self.C_hat = np.zeros((self.N_STATES,self.N_ACTIONS))
        #self.R_tilde = np.zeros((self.N_STATES,self.N_ACTIONS))
        #self.C_tilde = np.zeros((self.N_STATES,self.N_ACTIONS))
        
        
        self.NUMBER_OF_OCCURANCES = {}#np.zeros((self.N_STATES,self.N_ACTIONS))
        self.NUMBER_OF_OCCURANCES_p = {}#np.zeros((self.N_STATES,self.N_ACTIONS,self.N_STATES))
        self.beta_prob = {}#np.zeros((self.N_STATES,self.N_ACTIONS,self.N_STATES))
        self.beta_prob_1 = {}#np.zeros((self.N_STATES,self.N_ACTIONS))
        self.beta_prob_2 = {}#np.zeros((self.N_STATES,self.N_ACTIONS))
        self.beta_prob_T = {}
        self.Psparse = [[[] for i in self.ACTIONS] for j in range(self.N_STATES)] # dict(), [s][a] --> list of s'
        
        self.mu = np.zeros(self.N_STATES) # an array indicating if the initial state is fixed
        self.mu[0] = 1.0 # initial state is fixed
        self.CONSTRAINT = CONSTRAINT
        
        self.R_Tao = {}
        for s in range(self.N_STATES):
            l = len(self.ACTIONS[s])
            self.R_Tao[s] = np.zeros(l)
            
        self.C_Tao = {}
        for s in range(self.N_STATES):
            l = len(self.ACTIONS[s])
            self.C_Tao[s] = np.zeros(l)
        
        for s in range(self.N_STATES):
            self.P_hat[s] = {} # estimated transition probabilities
            l = len(self.ACTIONS[s])
            self.NUMBER_OF_OCCURANCES[s] = np.zeros(l) # initialize the number of occurences of [s][a]
            self.beta_prob_1[s] = np.zeros(l)
            self.beta_prob_2[s] = np.zeros(l)
            self.beta_prob_T[s] = np.zeros(l)
            self.NUMBER_OF_OCCURANCES_p[s] = np.zeros((l, N_STATES)) # initialize the number of occurences of [s][a, s']
            self.beta_prob[s] = np.zeros((l, N_STATES)) # [s][a, s']
            
            for a in self.ACTIONS[s]:
                self.P_hat[s][a] = np.zeros(self.N_STATES) # initialize the estimated transition probabilities
                for s_1 in range(self.N_STATES):
                    if self.P[s][a][s_1] > 0:
                        self.Psparse[s][a].append(s_1) # collect list of s' for each P[s][a]
    

    def step(self,s, a, h):  # take a step in the environment
        # h is not used here
        probs = np.zeros((self.N_STATES))
        for next_s in range(self.N_STATES):
            probs[next_s] = self.P[s][a][next_s]
        next_state = int(np.random.choice(np.arange(self.N_STATES),1,replace=True,p=probs)) # find next_state based on the transition probabilities
        rew = self.R[s][a]
        cost = self.C[s][a]
        return next_state,rew, cost


    def setCounts(self,ep_count_p,ep_count): # add the counts of the current episode to the total counts
        for s in range(self.N_STATES):
            for a in self.ACTIONS[s]:
                self.NUMBER_OF_OCCURANCES[s][a] += ep_count[s, a]
                for s_ in range(self.N_STATES):
                    self.NUMBER_OF_OCCURANCES_p[s][a, s_] += ep_count_p[s, a, s_]


    def compute_confidence_intervals(self,ep, mode): # compute the confidence intervals beta for the transition probabilities
        for s in range(self.N_STATES):
            for a in self.ACTIONS[s]:
                if self.NUMBER_OF_OCCURANCES[s][a] == 0:
                    self.beta_prob[s][a, :] = np.ones(self.N_STATES)
                    self.beta_prob_T[s][a] = np.sqrt(ep/max(self.NUMBER_OF_OCCURANCES[s][a],1)) # not sure what is beta_prob_T used for? Used in other algorithms
                else:
                    if mode == 2:
                        self.beta_prob[s][a, :] = min(np.sqrt(ep/max(self.NUMBER_OF_OCCURANCES[s][a], 1)), 1)*np.ones(self.N_STATES)
                    elif mode == 3:
                        self.beta_prob_T[s][a] = np.sqrt(ep/max(self.NUMBER_OF_OCCURANCES[s][a],1))
                        
                    for s_1 in range(self.N_STATES):
                        if mode == 0:
                            # DOPE policy, which equation?
                            self.beta_prob[s][a,s_1] = min(np.sqrt(ep*self.P_hat[s][a][s_1]*(1-self.P_hat[s][a][s_1])/max(self.NUMBER_OF_OCCURANCES[s][a],1)) + ep/(max(self.NUMBER_OF_OCCURANCES[s][a],1)), ep/(max(np.sqrt(self.NUMBER_OF_OCCURANCES[s][a]),1)), 1)
                        
                        elif mode == 1:
                            # safe base policy 
                            # equation (5) in the paper to calculate the confidence interval for P
                            self.beta_prob[s][a, s_1] = min(2*np.sqrt(ep*self.P_hat[s][a][s_1]*(1-self.P_hat[s][a][s_1])/max(self.NUMBER_OF_OCCURANCES[s][a],1)) + 14*ep/(3*max(self.NUMBER_OF_OCCURANCES[s][a],1)), 1)
                        
                self.beta_prob_1[s][a] = max(self.beta_prob[s][a, :])
                self.beta_prob_2[s][a] = sum(self.beta_prob[s][a, :])


    def update_empirical_model(self,ep): # update the empirical/estimated model based on the counters every episode
        # ep is not used here

        for s in range(self.N_STATES):
            for a in self.ACTIONS[s]:
                if self.NUMBER_OF_OCCURANCES[s][a] == 0:
                    self.P_hat[s][a] = 1/self.N_STATES*np.ones(self.N_STATES) # uniform distribution for unvisited state-action pairs
                else:
                    for s_1 in range(self.N_STATES):
                        self.P_hat[s][a][s_1] = self.NUMBER_OF_OCCURANCES_p[s][a,s_1]/(max(self.NUMBER_OF_OCCURANCES[s][a],1)) #calculate the estimated/empirical probabilities
                    self.P_hat[s][a] /= np.sum(self.P_hat[s][a]) # normalize the probabilities

                if abs(sum(self.P_hat[s][a]) - 1)  >  0.001: # sanity check  after updating the probabilities
                    print("empirical is wrong")
                    print(self.P_hat)
                    
                    
    def update_costs(self):
        alpha_r = (self.N_STATES*self.EPISODE_LENGTH) + 4*self.EPISODE_LENGTH*(self.N_STATES*self.EPISODE_LENGTH)/(self.CONSTRAINT-self.Cb)
        for s in range(self.N_STATES):
            for a in self.ACTIONS[s]:
                self.R_Tao[s][a] = self.R[s][a] + alpha_r * self.beta_prob_T[s][a]
                self.C_Tao[s][a] = self.C[s][a] + (self.EPISODE_LENGTH * self.N_STATES)*self.beta_prob_T[s][a]





    def compute_opt_LP_Unconstrained(self, ep):

        print("\nComputing the optimal policy using LP_unconstrained ...")

        opt_policy = np.zeros((self.N_STATES,self.EPISODE_LENGTH,self.N_STATES)) #[s,h,a]
        opt_prob = p.LpProblem("OPT_LP_problem",p.LpMaximize)
        opt_q = np.zeros((self.EPISODE_LENGTH,self.N_STATES,self.N_STATES)) #[h,s,a], this is the solution container for decision variable w_h(s,a) in the paper
    
        #create problem variables
        q_keys = [(h,s,a) for h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in self.ACTIONS[s]]
        q = p.LpVariable.dicts("q",q_keys,lowBound=0,cat='Continuous') # q is the decision variable w_h(s,a) in the paper. Here uses a dict with keys (h,s,a)
        # define the lower bound of q as 0, so that the decision variable is non-negative, equation 17(e)
        
        #Objective function, equation 17(a)
        list_1 = [self.R[s][a] for s in range(self.N_STATES) for a in self.ACTIONS[s]] * self.EPISODE_LENGTH
        list_2 = [q[(h,s,a)] for  h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in self.ACTIONS[s]]

        opt_prob += p.lpDot(list_1,list_2) # this is dot product of two lists, objective function is the dot product of the reward vector and the decision variable w_h(s,a)
                  
        #opt_prob += p.lpSum([q[(h,s,a)]*self.R[s,a] for  h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in range(self.N_ACTIONS)])

        # this is unconstrained MDP, thus no constrained regret 
                  
        for h in range(1,self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                q_list = [q[(h,s,a)] for a in self.ACTIONS[s]]
                pq_list = [self.P[s_1][a_1][s]*q[(h-1,s_1,a_1)] for s_1 in range(self.N_STATES) for a_1 in self.ACTIONS[s_1]]
                opt_prob += p.lpSum(q_list) - p.lpSum(pq_list) == 0 # constraint 1, equation (17c)

        for s in range(self.N_STATES):
            q_list = [q[(0,s,a)] for a in self.ACTIONS[s]]
            opt_prob += p.lpSum(q_list) - self.mu[s] == 0 # constraint 2, equation (17d)
                
        status = opt_prob.solve(p.PULP_CBC_CMD(gapRel=0.001, msg = 0)) # solve the LP problem
        #print(p.LpStatus[status])   # The solution status
        #print(opt_prob)
        print("printing best value")
        print(p.value(opt_prob.objective))
        # for constraint in opt_prob.constraints:
        #     print(opt_prob.constraints[constraint].name, opt_prob.constraints[constraint].value() - opt_prob.constraints[constraint].constant)
                          
        for h in range(self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                for a in self.ACTIONS[s]:
                    opt_q[h,s,a] = q[(h,s,a)].varValue # fetch the solution of the decision variable w_h(s,a) from the LP problem

                # compute the optimal policy from the opt_q
                for a in self.ACTIONS[s]:
                    if np.sum(opt_q[h,s,:]) == 0:
                        opt_policy[s,h,a] = 1/len(self.ACTIONS[s])
                    else:
                        opt_policy[s,h,a] = opt_q[h,s,a]/np.sum(opt_q[h,s,:]) # equation (13), this is probability of take action a at state s at time h, which is the optimal policy
                    probs = opt_policy[s,h,:] # not used
                                                                  
        if ep != 0: # have not seen when the ep is not 0
            return opt_policy, 0, 0, 0

        # evaluate the optimal policy                                                                  
        q_policy, value_of_policy, cost_of_policy = self.FiniteHorizon_Policy_evaluation(self.P, opt_policy, self.R, self.C)
        

        val_policy = 0
        con_policy = 0
       
        for h in range(self.EPISODE_LENGTH):
         for s in range(self.N_STATES):
            for a in self.ACTIONS[s]:
                con_policy  += opt_q[h,s,a]*self.C[s][a] # use the occupancy of the state-action pair to compute the cost of the policy
                if opt_q[h,s,a] < 0:
                        opt_q[h,s,a] = 0
                elif opt_q[h,s,a] > 1:
                    opt_q[h,s,a] = 1.0
                
                val_policy += opt_q[h,s,a]*self.R[s][a]
                    
        print("value from the UnconLPsolver")
        print("value of policy", val_policy)
        print("cost of policy", con_policy)
                                                                          
        return opt_policy, value_of_policy, cost_of_policy, q_policy
                                                                                  
                                                                                  
    def compute_opt_LP_Constrained(self, ep):

        print("\nComputing optimal policy with constrained LP solver ...")

        opt_policy = np.zeros((self.N_STATES,self.EPISODE_LENGTH,self.N_STATES)) #[s,h,a], note here the action dimension is N_STATES
        opt_prob = p.LpProblem("OPT_LP_problem",p.LpMaximize)
        opt_q = np.zeros((self.EPISODE_LENGTH,self.N_STATES,self.N_STATES)) #[h,s,a], note here the action dimension is N_STATES
                                                                                  
        #create problem variables
        q_keys = [(h,s,a) for h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in self.ACTIONS[s]]
                                                                                          
        q = p.LpVariable.dicts("q",q_keys,lowBound=0,cat='Continuous')

        opt_prob += p.lpSum([q[(h,s,a)]*self.R[s][a] for  h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in self.ACTIONS[s]]) # objective function
            
        opt_prob += p.lpSum([q[(h,s,a)]*self.C[s][a] for h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in self.ACTIONS[s]]) - self.CONSTRAINT <= 0 # constrained !!!
            
        for h in range(1,self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                q_list = [q[(h,s,a)] for a in self.ACTIONS[s]]
                pq_list = [self.P[s_1][a_1][s]*q[(h-1,s_1,a_1)] for s_1 in range(self.N_STATES) for a_1 in self.ACTIONS[s_1]]
                opt_prob += p.lpSum(q_list) - p.lpSum(pq_list) == 0 # equation 17(c)

        for s in range(self.N_STATES):
            q_list = [q[(0,s,a)] for a in self.ACTIONS[s]]
            opt_prob += p.lpSum(q_list) - self.mu[s] == 0 # equation 17(d)

        status = opt_prob.solve(p.PULP_CBC_CMD(gapRel=0.001, msg = 0)) # solve the constrained LP problem
        #print(p.LpStatus[status])   # The solution status
        #print(opt_prob)
        print("printing best value constrained")
        print(p.value(opt_prob.objective))
                                                                                                                  
        # for constraint in opt_prob.constraints:
        #     print(opt_prob.constraints[constraint].name, opt_prob.constraints[constraint].value() - opt_prob.constraints[constraint].constant)
                                                                                                                  
        for h in range(self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                for a in self.ACTIONS[s]:
                    opt_q[h,s,a] = q[(h,s,a)].varValue
                for a in self.ACTIONS[s]: # for actions that are not available in the current state, their probability is 0
                    if np.sum(opt_q[h,s,:]) == 0:
                        opt_policy[s,h,a] = 1/len(self.ACTIONS[s])
                    else:
                        opt_policy[s,h,a] = opt_q[h,s,a]/np.sum(opt_q[h,s,:]) # calculate the optimal policy from the occupancy measures of the state-action pair

                        if math.isnan(opt_policy[s,h,a]):
                            opt_policy[s,h,a] = 1/len(self.ACTIONS[s])
                        elif opt_policy[s,h,a] > 1.0:
                            print("invalid value printing")
                            print("opt_policy[s,h,a]", opt_policy[s,h,a])
                #probs = opt_policy[s,h,:]
                #optimal_policy[s,h] = int(np.argmax(probs))
                                                                                                                                                                  
        if ep != 0:
            return opt_policy, 0, 0, 0
        
        # calculate the results to double check with the results obtained from self.FiniteHorizon_Policy_evaluation()
        val_policy = 0
        con_policy = 0
        for h in range(self.EPISODE_LENGTH):
         for s in range(self.N_STATES):
            for a in self.ACTIONS[s]:
                if opt_q[h,s,a] < 0:
                        opt_q[h,s,a] = 0
                elif opt_q[h,s,a] > 1:
                    opt_q[h,s,a] = 1.0
                    
                con_policy  += opt_q[h,s,a]*self.C[s][a]
                val_policy += opt_q[h,s,a]*self.R[s][a]
        print("value from the conLPsolver")
        print("value of policy", val_policy)
        print("cost of policy", con_policy)

        q_policy, value_of_policy, cost_of_policy = self.FiniteHorizon_Policy_evaluation(self.P, opt_policy, self.R, self.C) # evaluate the optimal policy using finite horizon policy evaluation
                                                                                                                                                                          
        return opt_policy, value_of_policy, cost_of_policy, q_policy
                                                                                                                                                                                  
                                                                                                                                                                                  
                                                                                                                                                                                  

    def compute_extended_LP(self,ep, cb):
        """
        - solve equation (10) CMDP using extended Linear Programming
        - optimal policy opt_policy[s,h,a] is the probability of taking action a at state s at time h
        - evaluate optimal policy using finite horizon policy evaluation, to get 
            - value_of_policy: expected cumulative value, [s,h] 
            - cost_of_policy: expected cumulative cost, [s,h]
            - q_policy: expected cumulative rewards for [s,h,a]
        """

        opt_policy = np.zeros((self.N_STATES,self.EPISODE_LENGTH,self.N_STATES)) #[s,h,a]
        opt_prob = p.LpProblem("OPT_LP_problem",p.LpMaximize)
        opt_z = np.zeros((self.EPISODE_LENGTH,self.N_STATES,self.N_STATES,self.N_STATES)) #[h,s,a,s_], decision variable, state-action-state occupancy measure
        #create problem variables
        
        z_keys = [(h,s,a,s_1) for h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in self.ACTIONS[s] for s_1 in self.Psparse[s][a]]
        z = p.LpVariable.dicts("z_var",z_keys,lowBound=0,upBound=1,cat='Continuous') # why the upperbound is 1? Because the Z is essentially probability, and the probability is between 0 and 1
        # lower bound is 0, because the occupancy measure is non-negative, constraint (18e) in the paper
            
        # r_k = {}
        # for s in range(self.N_STATES):
        #     l = len(self.ACTIONS[s])
        #     r_k[s] = np.zeros(l)
        #     for a in self.ACTIONS[s]:
        #         r_k[s][a] = self.R[s][a] + self.EPISODE_LENGTH**2/(self.CONSTRAINT - cb)* self.beta_prob_2[s][a]

        # objective function
        # why not adding the confidence bound to the objective function? as shown in equation (18a) in the papers
        opt_prob += p.lpSum([z[(h,s,a,s_1)]*self.R[s][a] for h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in self.ACTIONS[s] for s_1 in self.Psparse[s][a]])

        #Constraints equation 18(b)                                   # why plus sign here, not minus sign?
        opt_prob += p.lpSum([z[(h,s,a,s_1)]*(self.C[s][a] + self.EPISODE_LENGTH*self.beta_prob_2[s][a]) for h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in self.ACTIONS[s] for s_1 in self.Psparse[s][a]]) - self.CONSTRAINT <= 0

        for h in range(1,self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                z_list = [z[(h,s,a,s_1)] for a in self.ACTIONS[s] for s_1 in self.Psparse[s][a]]
                z_1_list = [z[(h-1,s_1,a_1,s)] for s_1 in range(self.N_STATES) for a_1 in self.ACTIONS[s_1] if s in self.Psparse[s_1][a_1]]
                opt_prob += p.lpSum(z_list) - p.lpSum(z_1_list) == 0 # constraint (18c) in the paper
                                                                                                                                                                                                              
        for s in range(self.N_STATES):
            q_list = [z[(0,s,a,s_1)] for a in self.ACTIONS[s] for s_1 in self.Psparse[s][a]]
            opt_prob += p.lpSum(q_list) - self.mu[s] == 0 # constraint (18d) in the paper
                                                                                                                                                                                                                      
                                                                                                                                                                                                                      #start_time = time.time()
        for h in range(self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                for a in self.ACTIONS[s]:
                    for s_1 in self.Psparse[s][a]:
                        opt_prob += z[(h,s,a,s_1)] - (self.P_hat[s][a][s_1] + self.beta_prob[s][a,s_1]) *  p.lpSum([z[(h,s,a,y)] for y in self.Psparse[s][a]]) <= 0  # equation (18f)
                        opt_prob += -z[(h,s,a,s_1)] + (self.P_hat[s][a][s_1] - self.beta_prob[s][a,s_1])* p.lpSum([z[(h,s,a,y)] for y in self.Psparse[s][a]]) <= 0 # equation (18g)
                                                                                                                                                                                                                                        
        status = opt_prob.solve(p.PULP_CBC_CMD(gapRel=0.01, msg = 0)) # solve the Extended LP problem
                                                                                                                                                                                                                                      
        if p.LpStatus[status] != 'Optimal':
            print(p.LpStatus[status])
            return np.zeros((self.N_STATES, self.EPISODE_LENGTH, self.N_STATES)), np.zeros((self.N_STATES, self.EPISODE_LENGTH)), np.zeros((self.N_STATES, self.EPISODE_LENGTH)), p.LpStatus[status], np.zeros((self.N_STATES, self.EPISODE_LENGTH, self.N_STATES))
                                                                                                                                                                                                                                                  
        for h in range(self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                for a in self.ACTIONS[s]:
                    for s_1 in self.Psparse[s][a]:
                        opt_z[h,s,a,s_1] = z[(h,s,a,s_1)].varValue # get the optimal z
                        if opt_z[h,s,a,s_1] < 0 and opt_z[h,s,a,s_1] > -0.001: # check the validity of the optimal z
                            opt_z[h,s,a,s_1] = 0
                        elif opt_z[h,s,a,s_1] <= -0.001:
                            print("invalid value")
                            sys.exit()

        # calculate the optimal policy based on the optimal z                                                                                                                                                                                                                                                                  
        den = np.sum(opt_z,axis=(2,3)) # [h,s] sum over a and s_1
        num = np.sum(opt_z,axis=3)     # [h,s,a] sum over s_1                                                                                                                                                                                                                                                                             
                                                                                                                                                                  
        for h in range(self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                sum_prob = 0

                for a in self.ACTIONS[s]:
                    if den[h,s] == 0:
                        # print("warning: denominator is zero")
                        opt_policy[s,h,a] = 1/len(self.ACTIONS[s]) # added code here to handle the 0 denominator cases at the beginning of the DOPE training
                    else:
                        opt_policy[s,h,a] = num[h,s,a]/den[h,s] # invalid value error used to be here, equation (19)
                    sum_prob += opt_policy[s,h,a]
                
                if abs(sum(num[h,s,:]) - den[h,s]) > 0.0001: # check if the values are matching
                    print("wrong values")
                    print(sum(num[h,s,:]),den[h,s])
                    sys.exit()

                if math.isnan(sum_prob): # this should not happen, bc the 0 denominator cases are handled above
                    for a in self.ACTIONS[s]:
                        opt_policy[s,h,a] = 1/len(self.ACTIONS[s])
                else:
                    for a in self.ACTIONS[s]:
                        opt_policy[s,h,a] = opt_policy[s,h,a]/sum_prob # normalize the policy to make sure the sum of the probabilities is 1

        q_policy, value_of_policy, cost_of_policy = self.FiniteHorizon_Policy_evaluation(self.P, opt_policy, self.R, self.C)
                                                                                                                                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                                                                                  
        return opt_policy, value_of_policy, cost_of_policy, p.LpStatus[status], q_policy
    
    
    def compute_LP_Tao(self, ep, cb):
        opt_policy = np.zeros((self.N_STATES,self.EPISODE_LENGTH,self.N_STATES)) #[s,h,a]
        opt_prob = p.LpProblem("OPT_LP_problem",p.LpMaximize)
        opt_q = np.zeros((self.EPISODE_LENGTH,self.N_STATES,self.N_STATES)) #[h,s,a]
                                                                                  
        #create problem variables
        q_keys = [(h,s,a) for h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in self.ACTIONS[s]]
                                                                                          
        q = p.LpVariable.dicts("q",q_keys,lowBound=0,cat='Continuous')
        
        
        
        
        # alpha_r = 1 + self.N_STATES*self.EPISODE_LENGTH + 4*self.EPISODE_LENGTH*(1+self.N_STATES*self.EPISODE_LENGTH)/(self.CONSTRAINT-cb)
        # for s in range(self.N_STATES):
        #     l = len(self.ACTIONS[s])
        #     self.R_Tao[s] = np.zeros(l)
        #     for a in self.ACTIONS[s]:
        #         self.R_Tao[s][a] = self.R[s][a] - alpha_r * self.beta_prob_T[s][a]
                
       
        
        # for s in range(self.N_STATES):
        #     l = len(self.ACTIONS[s])
        #     self.C_Tao[s] = np.zeros(l)
        #     for a in self.ACTIONS[s]:
        #         self.C_Tao[s][a] = self.C[s][a] + (1 + self.EPISODE_LENGTH * self.N_STATES)*self.beta_prob_T[s][a]
                
                
        
        #print(alpha_r)
        
        # for s in range(self.N_STATES):
        #     for a in range(self.N_ACTIONS):
        #         self.R_Tao[s][a] = self.R[s][a] - alpha_r * self.beta_prob_T[s][a]
        #         self.C_Tao[s][a] = self.C[s][a] + (self.EPISODE_LENGTH * self.N_STATES)*self.beta_prob_T[s][a]
        
        

        opt_prob += p.lpSum([q[(h,s,a)]*(self.R_Tao[s][a]) for  h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in self.ACTIONS[s]])
            
        opt_prob += p.lpSum([q[(h,s,a)]*(self.C_Tao[s][a]) for h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in self.ACTIONS[s]]) - self.CONSTRAINT <= 0
            
        for h in range(1,self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                q_list = [q[(h,s,a)] for a in self.ACTIONS[s]]
                pq_list = [self.P_hat[s_1][a_1][s]*q[(h-1,s_1,a_1)] for s_1 in range(self.N_STATES) for a_1 in self.ACTIONS[s_1]]
                opt_prob += p.lpSum(q_list) - p.lpSum(pq_list) == 0

        for s in range(self.N_STATES):
            q_list = [q[(0,s,a)] for a in self.ACTIONS[s]]
            opt_prob += p.lpSum(q_list) - self.mu[s] == 0

        status = opt_prob.solve(p.PULP_CBC_CMD(gapRel=0.001, msg = 0))
        #if p.LpStatus[status] != 'Optimal':
            #return np.zeros((self.N_STATES, self.EPISODE_LENGTH, self.N_STATES)), np.zeros((self.N_STATES, self.EPISODE_LENGTH)), np.zeros((self.N_STATES, self.EPISODE_LENGTH)), p.LpStatus[status]
        #print(p.LpStatus[status])   # The solution status
        #print(opt_prob)
        # print("printing best value constrained")
        # print(p.value(opt_prob.objective))
                                                                                                                  
        # for constraint in opt_prob.constraints:
        #     print(opt_prob.constraints[constraint].name, opt_prob.constraints[constraint].value() - opt_prob.constraints[constraint].constant)
                                                                                                                  
        for h in range(self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                for a in self.ACTIONS[s]:
                    opt_q[h,s,a] = q[(h,s,a)].varValue
                for a in self.ACTIONS[s]:
                    if np.sum(opt_q[h,s,:]) == 0:
                        opt_policy[s,h,a] = 1/len(self.ACTIONS[s])
                    else:
                        opt_policy[s,h,a] = opt_q[h,s,a]/np.sum(opt_q[h,s,:])
                        if math.isnan(opt_policy[s,h,a]):
                            opt_policy[s,h,a] = 1/len(self.ACTIONS[s])
                        elif opt_policy[s,h,a] > 1.0 and opt_policy[s,h,a]<1.1:
                            opt_policy[s,h,a] = 1.0
                        elif opt_policy[s,h,a]>1.1:
                            print("invalid value printing",opt_policy[s,h,a])
                            #print(opt_policy[s,h,a])
                #probs = opt_policy[s,h,:]
                #optimal_policy[s,h] = int(np.argmax(probs))
                                                                                                                                                                  
        if ep != 0:
            return opt_policy, 0, 0, 0
        
        
        for h in range(self.EPISODE_LENGTH):
         for s in range(self.N_STATES):
            for a in self.ACTIONS[s]:
                if opt_q[h,s,a] < 0:
                        opt_q[h,s,a] = 0
                elif opt_q[h,s,a] > 1:
                    opt_q[h,s,a] = 1.0
                    

        q_policy, value_of_policy, cost_of_policy = self.FiniteHorizon_Policy_evaluation(self.P, opt_policy, self.R, self.C)
                                                                                                                                                                          
        return opt_policy, value_of_policy, cost_of_policy, p.LpStatus[status]
                                                                                                                                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                                                                          
    def compute_extended_LP1(self,ep,alg):
        opt_policy = np.zeros((self.N_STATES,self.EPISODE_LENGTH,self.N_STATES)) #[s,h,a]
        opt_prob = p.LpProblem("OPT_LP_problem",p.LpMaximize)
        opt_z = np.zeros((self.EPISODE_LENGTH,self.N_STATES,self.N_STATES,self.N_STATES)) #[h,s,a,s_]

        z_keys = [(h,s,a,s_1) for h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in self.ACTIONS[s] for s_1 in self.Psparse[s][a]]
                                                                                                                                                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                                                                                                  
        z = p.LpVariable.dicts("z_var",z_keys,lowBound=0,upBound=1,cat='Continuous')
        opt_prob += p.lpSum([z[(h,s,a,s_1)]*self.R[s][a] for h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in self.ACTIONS[s] for s_1 in self.Psparse[s][a]])
                                                                                                                                                                                                                                                                                                                                                                      
        #Constraints
        if alg == 1:
            opt_prob += p.lpSum([z[(h,s,a,s_1)]*self.C[s][a] for h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in self.ACTIONS[s] for s_1 in self.Psparse[s][a]]) - self.CONSTRAINT <= 0
                                                                                                                                                                                                                                                                                                                                                                      
        for h in range(1,self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                z_list = [z[(h,s,a,s_1)] for a in self.ACTIONS[s] for s_1 in self.Psparse[s][a]]
                z_1_list = [z[(h-1,s_1,a_1,s)] for s_1 in range(self.N_STATES) for a_1 in self.ACTIONS[s_1] if s in self.Psparse[s_1][a_1]]
                opt_prob += p.lpSum(z_list) - p.lpSum(z_1_list) == 0
                                                                                                                                                                                                                                                                                                                                                                                      
        for s in range(self.N_STATES):
            q_list = [z[(0,s,a,s_1)] for a in self.ACTIONS[s] for s_1 in self.Psparse[s][a]]
            opt_prob += p.lpSum(q_list) - self.mu[s] == 0
                                                                                                                                                                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                                                                                                                                              #start_time = time.time()
        for h in range(self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                for a in self.ACTIONS[s]:
                    for s_1 in self.Psparse[s][a]:
                        opt_prob += z[(h,s,a,s_1)] - (self.P_hat[s][a][s_1] + self.beta_prob[s][a,s_1]) *  p.lpSum([z[(h,s,a,y)] for y in self.Psparse[s][a]]) <= 0
                        opt_prob += -z[(h,s,a,s_1)] + (self.P_hat[s][a][s_1] - self.beta_prob[s][a,s_1])* p.lpSum([z[(h,s,a,y)] for y in self.Psparse[s][a]]) <= 0

        status = opt_prob.solve(p.PULP_CBC_CMD(gapRel=0.01, msg = 0))
                                                                                                                                                                                                                                                                                                                                                                                                              
        if p.LpStatus[status] != 'Optimal':
            return np.zeros((self.N_STATES, self.EPISODE_LENGTH, self.N_STATES)), np.zeros((self.N_STATES, self.EPISODE_LENGTH)), np.zeros((self.N_STATES, self.EPISODE_LENGTH)), p.LpStatus[status], np.zeros((self.N_STATES, self.EPISODE_LENGTH, self.N_STATES))

        for h in range(self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                for a in self.ACTIONS[s]:
                    for s_1 in self.Psparse[s][a]:
                        opt_z[h,s,a,s_1] = z[(h,s,a,s_1)].varValue
                        if opt_z[h,s,a,s_1] < 0 and opt_z[h,s,a,s_1] > -0.001:
                            opt_z[h,s,a,s_1] = 0
                        elif opt_z[h,s,a,s_1] < -0.001:
                            print("invalid value")
                            sys.exit()

        den = np.sum(opt_z,axis=(2,3))
        num = np.sum(opt_z,axis=3)

        for h in range(self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                sum_prob = 0
                for a in self.ACTIONS[s]:
                    opt_policy[s,h,a] = num[h,s,a]/den[h,s]
                    sum_prob += opt_policy[s,h,a]
                if abs(sum(num[h,s,:]) - den[h,s]) > 0.0001:
                    print("wrong values")
                    print(sum(num[h,s,:]),den[h,s])
                    sys.exit()
                if math.isnan(sum_prob):
                    for a in self.ACTIONS[s]:
                        opt_policy[s,h,a] = 1/len(self.ACTIONS[s])
                else:
                    for a in self.ACTIONS[s]:
                        opt_policy[s,h,a] = opt_policy[s,h,a]/sum_prob

        q_policy, value_of_policy, cost_of_policy = self.FiniteHorizon_Policy_evaluation(self.P, opt_policy, self.R, self.C)

        return opt_policy, value_of_policy, cost_of_policy, p.LpStatus[status], q_policy

    def compute_extended_ucrl2(self,ep):
        opt_policy = np.zeros((self.N_STATES,self.EPISODE_LENGTH,self.N_STATES)) #[s,h,a]
        opt_prob = p.LpProblem("OPT_LP_problem",p.LpMaximize)
        opt_z = np.zeros((self.EPISODE_LENGTH,self.N_STATES,self.N_STATES,self.N_STATES)) #[h,s,a,s_]
        
        z_keys = [(h,s,a,s_1) for h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in self.ACTIONS[s] for s_1 in self.Psparse[s][a]]
        
        
        z = p.LpVariable.dicts("z_var",z_keys,lowBound=0,upBound=1,cat='Continuous')
        opt_prob += p.lpSum([z[(h,s,a,s_1)]*self.R[s][a] for h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in self.ACTIONS[s] for s_1 in self.Psparse[s][a]])
        
        #Constraints
        #opt_prob += p.lpSum([z[(h,s,a,s_1)]*self.C[s][a] for h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in self.ACTIONS[s] for s_1 in self.Psparse[s][a]]) - self.CONSTRAINT <= 0
        
        for h in range(1,self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                z_list = [z[(h,s,a,s_1)] for a in self.ACTIONS[s] for s_1 in self.Psparse[s][a]]
                z_1_list = [z[(h-1,s_1,a_1,s)] for s_1 in range(self.N_STATES) for a_1 in self.ACTIONS[s_1] if s in self.Psparse[s_1][a_1]]
                opt_prob += p.lpSum(z_list) - p.lpSum(z_1_list) == 0
        
        for s in range(self.N_STATES):
            q_list = [z[(0,s,a,s_1)] for a in self.ACTIONS[s] for s_1 in self.Psparse[s][a]]
            opt_prob += p.lpSum(q_list) - self.mu[s] == 0

        #start_time = time.time()
        for h in range(self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                for a in self.ACTIONS[s]:
                    for s_1 in self.Psparse[s][a]:
                        opt_prob += z[(h,s,a,s_1)] - (self.P_hat[s][a][s_1] + self.beta_prob[s][a,s_1]) *  p.lpSum([z[(h,s,a,y)] for y in self.Psparse[s][a]]) <= 0
                        opt_prob += -z[(h,s,a,s_1)] + (self.P_hat[s][a][s_1] - self.beta_prob[s][a,s_1])* p.lpSum([z[(h,s,a,y)] for y in self.Psparse[s][a]]) <= 0

        status = opt_prob.solve(p.PULP_CBC_CMD(gapRel=0.01, msg = 0))

        if p.LpStatus[status] != 'Optimal':
            return np.zeros((self.N_STATES, self.EPISODE_LENGTH, self.N_STATES)), np.zeros((self.N_STATES, self.EPISODE_LENGTH)), np.zeros((self.N_STATES, self.EPISODE_LENGTH)), p.LpStatus[status], np.zeros((self.N_STATES, self.EPISODE_LENGTH, self.N_STATES))
        
        for h in range(self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                for a in self.ACTIONS[s]:
                    for s_1 in self.Psparse[s][a]:
                        opt_z[h,s,a,s_1] = z[(h,s,a,s_1)].varValue
                        if opt_z[h,s,a,s_1] < 0 and opt_z[h,s,a,s_1] > -0.001:
                            opt_z[h,s,a,s_1] = 0
                        elif opt_z[h,s,a,s_1] < -0.001:
                            print("invalid value")
                            sys.exit()
    
        den = np.sum(opt_z,axis=(2,3))
        num = np.sum(opt_z,axis=3)
        
        for h in range(self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                sum_prob = 0
                for a in self.ACTIONS[s]:
                    opt_policy[s,h,a] = num[h,s,a]/den[h,s]
                    sum_prob += opt_policy[s,h,a]
                if abs(sum(num[h,s,:]) - den[h,s]) > 0.0001:
                    print("wrong values")
                    print(sum(num[h,s,:]),den[h,s])
                    sys.exit()
                if math.isnan(sum_prob):
                    for a in self.ACTIONS[s]:
                        opt_policy[s,h,a] = 1/len(self.ACTIONS[s])
                else:
                    for a in self.ACTIONS[s]:
                        opt_policy[s,h,a] = opt_policy[s,h,a]/sum_prob
        
        q_policy, value_of_policy, cost_of_policy = self.FiniteHorizon_Policy_evaluation(self.P, opt_policy)

        return opt_policy, value_of_policy, cost_of_policy, p.LpStatus[status], q_policy

    def FiniteHorizon_Policy_evaluation(self,Px,policy,R,C):
        
        # results to be returned
        q = np.zeros((self.N_STATES,self.EPISODE_LENGTH, self.N_STATES)) # q(s,h,a), q_policy, expected cumulative rewards
        v = np.zeros((self.N_STATES, self.EPISODE_LENGTH)) # v(s,h), expected cumulative value of the calculated optimal policy
        c = np.zeros((self.N_STATES,self.EPISODE_LENGTH)) # c(s,h), expected cumulative cost of the calculated optimal policy

        P_policy = np.zeros((self.N_STATES,self.EPISODE_LENGTH,self.N_STATES)) # P_policy(s,h,s_1), probability of being in state s_1 at time h+1 given that we are in state s at time h and we follow the optimal policy
        R_policy = np.zeros((self.N_STATES,self.EPISODE_LENGTH)) # R_policy(s,h), expected reward of being in state s at time h given that we follow the optimal policy
        C_policy = np.zeros((self.N_STATES,self.EPISODE_LENGTH)) # C_policy(s,h), expected cost of being in state s at time h given that we follow the optimal policy

        # initialize the last state for the value and cost, and q
        for s in range(self.N_STATES):
            x = 0
            for a in self.ACTIONS[s]:
                x += policy[s, self.EPISODE_LENGTH - 1, a]*C[s][a] # expected cost of the last state
            c[s,self.EPISODE_LENGTH-1] = x #np.dot(policy[s,self.EPISODE_LENGTH-1,:], self.C[s])

            for a in self.ACTIONS[s]:
                q[s, self.EPISODE_LENGTH-1, a] = R[s][a]
            v[s,self.EPISODE_LENGTH-1] = np.dot(q[s, self.EPISODE_LENGTH-1, :], policy[s, self.EPISODE_LENGTH-1, :]) # expected value of the last state under the policy

        # build R_policy, C_policy, P_policy
        for h in range(self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                x = 0
                y = 0
                for a in self.ACTIONS[s]:
                    x += policy[s,h,a]*R[s][a]
                    y += policy[s,h,a]*C[s][a]
                R_policy[s,h] = x # expected reward of the state s at time h under the policy
                C_policy[s,h] = y # expected cost of the state s at time h under the policy
                for s_1 in range(self.N_STATES):
                    z = 0
                    for a in self.ACTIONS[s]:
                        z += policy[s,h,a]*Px[s][a][s_1] # expected transition probability of taking action a in state s at time h and ending up in state s_1, following the policy 
                    P_policy[s,h,s_1] = z #np.dot(policy[s,h,:],Px[s,:,s_1])

        # going backwards in timesteps to calculate the cumulative value and cost of the policy
        for h in range(self.EPISODE_LENGTH-2,-1,-1):
            for s in range(self.N_STATES):
                c[s,h] = C_policy[s,h] + np.dot(P_policy[s,h,:],c[:,h+1]) # expected cumulative cost of the state s at time h under the policy = expected cost of the state s at time h under the policy + expected cumulative cost of the state s at time h+1 under the policy
                for a in self.ACTIONS[s]:
                    z = 0
                    for s_ in range(self.N_STATES):
                        z += Px[s][a][s_] * v[s_, h+1]
                    q[s, h, a] = R[s][a] + z # expected cumulative rewards = current reward of taking action a at state s + expected cumulative value of the state s_ at time h+1
                v[s,h] = np.dot(q[s, h, :],policy[s, h, :]) # expected cumulative value, regardless of the action taken  
        #print("evaluation",v)
                

        return q, v, c

    def compute_qVals_EVI(self, Rx):
        # Extended value iteration
        qVals = {}
        qMax = {}
        qMax[self.EPISODE_LENGTH] = np.zeros(self.N_STATES)
        p_tilde = {}
        for h in range(self.EPISODE_LENGTH):
            j = self.EPISODE_LENGTH - h - 1
            qMax[j] = np.zeros(self.N_STATES)
            for s in range(self.N_STATES):
                qVals[s, j] = np.zeros(len(self.ACTIONS[s]))
                p_tilde[s] = {}
                for a in self.ACTIONS[s]:
                    #rOpt = R[s, a] + R_slack[s, a]
                    p_tilde[s][a] = np.zeros(self.N_STATES)
                    # form pOpt by extended value iteration, pInd sorts the values
                    pInd = np.argsort(qMax[j + 1])
                    pOpt = self.P_hat[s][a].copy()
                    if pOpt[pInd[self.N_STATES - 1]] + self.beta_prob_1[s][a] * 0.5 > 1:
                        pOpt = np.zeros(self.N_STATES)
                        pOpt[pInd[self.N_STATES - 1]] = 1
                    else:
                        pOpt[pInd[self.N_STATES - 1]] += self.beta_prob_1[s][a] * 0.5

                    sLoop = 0
                    while np.sum(pOpt) > 1:
                        worst = pInd[sLoop]
                        pOpt[worst] = max(0, 1 - np.sum(pOpt) + pOpt[worst])
                        sLoop += 1

                    qVals[s, j][a] = Rx[s][a] + np.dot(pOpt, qMax[j + 1])
                    p_tilde[s][a] = pOpt.copy()

                qMax[j][s] = np.max(qVals[s, j])

        return qVals, qMax, p_tilde

    def  deterministic2stationary(self, policy):
        stationary_policy = np.zeros((self.N_STATES, self.EPISODE_LENGTH, self.N_STATES))
        for s in range(self.N_STATES):
            for h in range(self.EPISODE_LENGTH):
                a = int(policy[s, h])
                stationary_policy[s, h, a] = 1

        return stationary_policy

    def update_policy_from_EVI(self, Rx):
        qVals, qMax, p_tilde = self.compute_qVals_EVI(Rx)
        policy = np.zeros((self.N_STATES,self.EPISODE_LENGTH))

        for h in range(self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                Q = qVals[s,h]
                policy[s,h] = np.random.choice(np.where(Q==Q.max())[0])

        self.P_tilde = p_tilde.copy()

        policy = self.deterministic2stationary(policy)

        q, v, c = self.FiniteHorizon_Policy_evaluation(self.P, policy)

        return policy, v, c, q
