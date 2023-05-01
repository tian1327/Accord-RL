import numpy as np
import pulp as p
import time
import math
import sys
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error

class utils:
    def __init__(self, eps, delta, M, P, R_model, C_model, CONTEXT_VEC_LENGTH, ACTION_CODE_LENGTH, STATE_CODE_LENGTH,
                INIT_STATE_INDEX, state_index_to_code, action_index_to_code, EPISODE_LENGTH, N_STATES, N_ACTIONS, ACTIONS, CONSTRAINT, Cb, use_gurobi):

        self.use_gurobi = use_gurobi
        self.EPISODE_LENGTH = EPISODE_LENGTH
        self.N_STATES = N_STATES
        self.N_ACTIONS = N_ACTIONS
        self.ACTIONS = ACTIONS

        self.P = P.copy()
        self.R = np.zeros((self.N_STATES,self.N_ACTIONS))
        self.R_obs = np.zeros((self.N_STATES,self.N_ACTIONS)) # observed CVDRisk with noises added
        self.R_y_pred = np.zeros((self.N_STATES,self.N_ACTIONS)) # predicted value from CVDRisk regression model
        self.C = np.zeros((self.N_STATES,self.N_ACTIONS))
        self.R_model = R_model
        self.C_model = C_model
        self.CONTEXT_VECTOR = None
        self.state_index_to_code = state_index_to_code
        self.action_index_to_code = action_index_to_code

        self.X = None # observed context vectors from frist episode to current episode
        self.S = None # observed state vector from frist episode to current episode
        self.A = None # observed action vector from frist episode to current episode
        self.sbp_cont = None # observed sbp feedback from frist episode to current episode
        self.cvdrisk = None # observed cvdrisk feedback from frist episode to current episode

        self.cvdrisk_regr = None # regression models for cvdrisk feedback using upto current observations
        self.sbp_regr = None

        # for connfidence bounds
        self.episode = 0
        self.C1 = 10.0
        self.C2 = 0.02
        self.C3 = 0.25
        self.CONTEXT_VEC_LENGTH = CONTEXT_VEC_LENGTH
        self.STATE_CODE_LENGTH = CONTEXT_VEC_LENGTH
        self.ACTION_CODE_LENGTH = ACTION_CODE_LENGTH
        self.U_sbp =  np.zeros((CONTEXT_VEC_LENGTH+ACTION_CODE_LENGTH, CONTEXT_VEC_LENGTH+ACTION_CODE_LENGTH)) # the design matrix for SBP raduis calculation
        self.U_cvd = np.zeros((CONTEXT_VEC_LENGTH+STATE_CODE_LENGTH+ACTION_CODE_LENGTH, CONTEXT_VEC_LENGTH+STATE_CODE_LENGTH+ACTION_CODE_LENGTH)) # the design matrix for CVDRisk raduis calculation, 
        self.U_xsa_prod_dict = dict() # trade memory for speed

        # print('Actions: ', ACTIONS)
        # print('self.ACTIONS: ', self.ACTIONS)
        self.eps = eps
        self.delta = delta
        self.M = M
        self.Cb = Cb
        #self.ENV_Q_VALUES = np.zeros((self.N_STATES,self.EPISODE_LENGTH,self.N_ACTIONS))
        
        self.P_hat = {}#np.zeros((self.N_STATES,self.N_ACTIONS,self.N_STATES))
        #self.P_tilde = np.zeros((self.N_STATES,self.N_ACTIONS,self.N_STATES))
        
        self.R_hat = np.zeros((self.N_STATES,self.N_ACTIONS))
        self.C_hat = np.zeros((self.N_STATES,self.N_ACTIONS))
        self.Total_emp_reward = np.zeros((self.N_STATES,self.N_ACTIONS))
        self.Total_emp_cost = np.zeros((self.N_STATES,self.N_ACTIONS))
        #self.R_tilde = np.zeros((self.N_STATES,self.N_ACTIONS))
        #self.C_tilde = np.zeros((self.N_STATES,self.N_ACTIONS))
        
        self.alpha_p = 1.0
        self.alpha_r = 0.1
        self.alpha_c = 1.0

        self.NUMBER_OF_OCCURANCES = {}#np.zeros((self.N_STATES,self.N_ACTIONS))
        self.NUMBER_OF_OCCURANCES_p = {}#np.zeros((self.N_STATES,self.N_ACTIONS,self.N_STATES))
        self.beta_prob = {}#np.zeros((self.N_STATES,self.N_ACTIONS,self.N_STATES))
        self.beta_prob_1 = {}#np.zeros((self.N_STATES,self.N_ACTIONS))
        self.beta_prob_2 = {}#np.zeros((self.N_STATES,self.N_ACTIONS))
        self.beta_prob_T = {}
        self.sbp_confidence = {}
        self.cvdrisk_confidence = {}
        self.P_confidence = {}
        self.Psparse = [[[] for i in range(self.N_ACTIONS)] for j in range(self.N_STATES)] # dict(), [s][a] --> list of s'
        # print dimension of self.Psparse
        # print('self.Psparse dimension: ', len(self.Psparse), len(self.Psparse[0]))
            
        self.mu = np.zeros(self.N_STATES) # an array indicating if the initial state is fixed
        # self.mu[0] = 1.0 # initial state is fixed
        self.mu[INIT_STATE_INDEX] = 1.0 # initial state is fixed to most frequent BLR state 

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
            # print('s: ', s, 'l: ', l)
            self.NUMBER_OF_OCCURANCES[s] = np.zeros(l) # initialize the number of occurences of [s][a]
            self.beta_prob_1[s] = np.zeros(l)
            self.beta_prob_2[s] = np.zeros(l)
            self.beta_prob_T[s] = np.zeros(l)
            self.sbp_confidence[s] = np.zeros(l)
            self.cvdrisk_confidence[s] = np.zeros(l)
            self.P_confidence[s] = np.zeros(l)
            self.NUMBER_OF_OCCURANCES_p[s] = np.zeros((l, N_STATES)) # initialize the number of occurences of [s][a, s']
            self.beta_prob[s] = np.zeros((l, N_STATES)) # [s][a, s']
            
            # print('s: ', s, 'self.ACTIONS[s]: ', self.ACTIONS[s])

            for a in self.ACTIONS[s]:
                self.P_hat[s][a] = np.zeros(self.N_STATES) # initialize the estimated transition probabilities
                for s_1 in range(self.N_STATES):
                    # self.Psparse[s][a].append(s_1) # collect list of s' for each P[s][a]

                    # because we stored P in sparse format, we have to handle key error
                    if s in self.P: # to avoid key error  
                        if a in self.P[s]:
                            # if s_1 in self.P[s][a]:

                            if self.P[s][a][s_1] > 0:
                                self.Psparse[s][a].append(s_1) # collect list of s' for each P[s][a]
            
                # print(s, a, 'self.Psparse[s][a]: ', self.Psparse[s][a])
            
        # print('self.Psparse: ', self.Psparse)
    
    def update_episode(self, episode):
        self.episode = episode

    def set_context(self, CONTEXT_VECTOR):
        self.CONTEXT_VECTOR = CONTEXT_VECTOR

    # calculate the true reward and cost for each state-action pair using the context vector and offline R and C models
    def calculate_true_R_C(self, context_vec):
        # print('\ncontext_vec: ', context_vec)
        # print('self.state_index_to_code: ', self.state_index_to_code)
        # print('self.action_index_to_code: ', self.action_index_to_code)

        for s in range(self.N_STATES):
            for a in self.ACTIONS[s]:
                state_code = self.state_index_to_code[s]
                action_code = self.action_index_to_code[a]
                # convert state and action code digit by digit to a list of 0 and 1
                state_code_list = [int(x) for x in list(state_code)]
                action_code_list = [int(x) for x in list(action_code)]
            
                # concatenate state and action code to the back of context vector
                R_input = np.concatenate((context_vec, np.array(state_code_list), np.array(action_code_list)), axis=0)
                # print('\n---s: ', s, 'a: ', a)
                # print('---np.array(state_code_list):', np.array(state_code_list))
                # print('---np.array(action_code_list):', np.array(action_code_list))
                # print('---R_input.shape ', R_input.shape)
                # print('---R_input: ', R_input)

                y_pred = self.R_model.predict(R_input.reshape(1, -1))   
                # print('---y_pred: ', y_pred)
                reward = 1.0 /(1.0+np.exp(-y_pred))
                # print('---reward: ', reward)

                self.R[s][a] = reward

                self.R_y_pred[s][a] = y_pred
                # self.R_obs[s][a] = obs_reward

                C_input = np.concatenate((context_vec, np.array(action_code_list)), axis=0)
                self.C[s][a] = self.C_model.predict(C_input.reshape(1, -1))

                # print('s: ', s, 'a: ', a, 'state_code: ', state_code, 'action_code: ', action_code, 'R_input: ', R_input, 'C_input: ', C_input, 'self.R[s][a]: ', self.R[s][a], 'self.C[s][a]: ', self.C[s][a])

            

    def add_ep_rewards_costs(self, ep_context_vec, ep_state_code, ep_action_code, ep_sbp_cont, ep_cvdrisk):
        
        # return if no experience
        if len(ep_action_code) == 0:
            return
        
        assert len(ep_state_code) == len(ep_action_code)

        ep_length = len(ep_state_code)
        x_matrix = np.tile(ep_context_vec, (ep_length, 1)) # stack contect_vector vertically for numebr of steps in the episode

        # build the state matrix from the list of state code, if the state code is 1010, then the action matrix has a row of [1, 0, 1, 0]
        state_vector_list = []
        for i in range(ep_length):
            state_code = ep_state_code[i]
            state_vector = [int(x) for x in list(state_code)]
            state_vector_list.append(state_vector)
        
        state_matrix = np.array(state_vector_list)        

        # build the action matrix from the list of action code, if the action code is 1010, then the action matrix has a row of [1, 0, 1, 0]
        action_vector_list = []
        for i in range(ep_length):
            action_code = ep_action_code[i]
            action_vector = [int(x) for x in list(action_code)]
            action_vector_list.append(action_vector)
        
        action_matrix = np.array(action_vector_list)
        
        sbp_fb_cont =  np.array(ep_sbp_cont).reshape(-1, 1) # make a column vector
        cvd = np.array(ep_cvdrisk).reshape(-1, 1) # make a column vector

        if self.episode == 1: # only 1 episode so far
            
            self.X = x_matrix
            self.S = state_matrix
            self.A = action_matrix
            self.sbp_cont = sbp_fb_cont
            self.cvdrisk = cvd            

        else: # stack the new episode to the history
            self.X = np.concatenate((self.X, x_matrix), axis=0)
            self.S = np.concatenate((self.S, state_matrix), axis=0)
            self.A = np.concatenate((self.A, action_matrix), axis=0)
            self.sbp_cont = np.concatenate((self.sbp_cont, sbp_fb_cont), axis=0)
            self.cvdrisk = np.concatenate((self.cvdrisk, cvd), axis=0)
            
        # print('slef.X.shape: ', self.X.shape)
        # print('self.X: \n', self.X)
        # print('slef.S.shape: ', self.S.shape)
        # print('self.S: \n', self.S)
        # print('slef.A.shape: ', self.A.shape)
        # print('self.A: \n', self.A)
        # print('self.sbp_cont.shape: ', self.sbp_cont.shape)
        # print('self.sbp_cont: \n', self.sbp_cont)
        # print('self.cvdrisk.shape: ', self.cvdrisk.shape)
        # print('self.cvdrisk: \n', self.cvdrisk)
        

    def make_x_a_vector(self, context_vec, state_idx, action_idx):

        # reshape the context vector to a column vector
        context_vec = context_vec.reshape(-1, 1)
        # print("context_vec.shape =", context_vec.shape)
        # print("context_vec =", context_vec)

        # transform the state to a column vector
        state_code = self.state_index_to_code[state_idx]
        state_code_list = [int(x) for x in list(state_code)]
        state_vec = np.array(state_code_list).reshape(-1, 1)

        # transform the action to a column vector
        action_code = self.action_index_to_code[action_idx]
        action_code_list = [int(x) for x in list(action_code)]
        action_vec = np.array(action_code_list).reshape(-1, 1)
        # print('action_code =', action_code)
        # print('action_vec =', action_vec)
        # print("action_vec.shape =", action_vec.shape)

        # stack the context vector and action vector to form the design vector
        xa_vec = np.vstack((context_vec, action_vec))
        # print("sbp_xa_vec.shape =", sbp_xa_vec.shape)


        # print('context_vec= ', context_vec)
        # print('action_vec= ', action_vec)

        xsa_vec = np.vstack((context_vec, state_vec, action_vec))

        # print("cvd_xsa_vec.shape =", cvd_xsa_vec.shape)
        # print("cvd_xsa_vec =", cvd_xsa_vec)

        return xa_vec, xsa_vec



    def run_regression_rewards_costs(self, episode):

        # do nothing if no history yet
        if episode == 0:
            return (0, 0)
        
        #---------run logistic/linear regression to estimate the CVDRisk feedback, get \theta r
        x_train = np.concatenate((self.X, self.S, self.A), axis=1)
        print('----x_train.shape', x_train.shape)

        # save x_train and self.cvdrisk to pickle file
        # import pickle
        # with open('output/x_train.pkl', 'wb') as f:
        #     pickle.dump(x_train, f)
        # with open('output/cvdrisk.pkl', 'wb') as f:
        #     pickle.dump(self.cvdrisk, f)

        # replace the training data here with data used for offline training
        # x_train = np.load('output/X.npy')
        # self.cvdrisk = np.load('output/y_true.npy')
        # print('----x_train.shape', x_train.shape)

        y = self.cvdrisk[:, 0] 
        # transform the y which is self.cvdrisk using -np.log((1-y)/y) 
        y_train = -np.log((1.0-y)/y) 

        # create a linear regression to fit x_train and y_train
        cvd_regr = LinearRegression()
        cvd_regr.fit(x_train, y_train)
        cvd_regr.score(x_train, y_train)
        print('+++cvd_regr.score(x_train, y_train) =', cvd_regr.score(x_train, y_train))

        y_pred = cvd_regr.predict(x_train)
        y_pred_transformed = 1.0/(1.0+np.exp(-y_pred))
        mse = mean_squared_error(y, y_pred_transformed)
        cvd_rmse = np.sqrt(mse)        
        self.cvdrisk_regr = cvd_regr
        print('+++cvd_rmse =', cvd_rmse)
        

        #---------run linear regression to estimate the deviation from the SBP_feedback, get \theta c
        x_train = np.concatenate((self.X, self.A), axis=1)
        # print('---sbp x_train.shape', x_train.shape)
        y_train = self.sbp_cont[:, 0]
        # print('---sbp y_train.shape', y_train.shape)

        # x_train = np.load('output/X_sbp.npy')
        # print('---sbp x_train.shape', x_train.shape)
        # y_train = np.load('output/y_sbp.npy')
        # print('---sbp y_train.shape', y_train.shape)

        # create a linear regression to fit x_train and y_train
        sbp_regr = LinearRegression()
        sbp_regr.fit(x_train, y_train)
        sbp_regr.score(x_train, y_train)
        print('+++sbp_regr.score(x_train, y_train) =', sbp_regr.score(x_train, y_train))
        y_pred = sbp_regr.predict(x_train)
        mse = mean_squared_error(y_train, y_pred)
        sbp_rmse = np.sqrt(mse)
        print('+++SBP RMSE = ', sbp_rmse)
        self.sbp_regr = sbp_regr

        #------------ calculate the l2norm of the difference between the weights of true model and estimated model
        
        # get the weights of self.R_model, which is a linear regression model
        R_model_weights = list(self.R_model.coef_)
        # print('R_model_weights: ', R_model_weights)
        R_model_intercept = [self.R_model.intercept_]
        print('+++R_model_intercept: ', R_model_intercept)
        R_model_wt_vec = np.array(R_model_weights + R_model_intercept)
        print('+++R_model_wt_vec: ', R_model_wt_vec)
        #print('type(R_model_wt_vec): ', type(R_model_wt_vec))
        
        # get the weights of self.cvdrisk_regr, which is a linear regression model
        R_hat_weights = self.cvdrisk_regr.coef_.tolist()
        # print('+++R_hat_weights: ', R_hat_weights)
        #print('type(R_hat_weights): ', type(R_hat_weights))
        R_hat_intercept = [self.cvdrisk_regr.intercept_]
        print('+++R_hat_intercept: ', R_hat_intercept)
        #print('type(R_hat_intercept): ', type(R_hat_intercept))
        R_hat_wt_vec = np.array(R_hat_weights + R_hat_intercept)
        print('+++R_hat_wt_vec: ', R_hat_wt_vec)
        #print('type(R_hat_wt_vec): ', type(R_hat_wt_vec))

        # get the l2 norm of the difference between R_model_wt_vec and R_hat_wt_vec
        R_est_error = np.linalg.norm(R_model_wt_vec - R_hat_wt_vec)

        # get the weights of self.C_model, which is a linear regression model
        C_model_weights = list(self.C_model.coef_)
        C_model_intercept = [self.C_model.intercept_]
        print('+++C_model_intercept: ', C_model_intercept)
        C_model_wt_vec = np.array(C_model_weights + C_model_intercept)
        print('+++C_model_wt_vec: ', C_model_wt_vec)

        # get the weights of self.sbp_regr, which is a linear regression model
        C_hat_weights = self.sbp_regr.coef_.tolist()
        C_hat_intercept = [self.sbp_regr.intercept_]
        print('+++C_hat_intercept: ', C_hat_intercept)
        C_hat_wt_vec = np.array(C_hat_weights + C_hat_intercept)
        print('+++C_hat_wt_vec: ', C_hat_wt_vec)

        # get the l2 norm of the difference between C_model_wt_vec and C_hat_wt_vec
        C_est_error = np.linalg.norm(C_model_wt_vec - C_hat_wt_vec)

        print('cvd_rmse = ', round(cvd_rmse,4), 'sbp_rmse = ', round(sbp_rmse,4), 'R_est_error = ', round(R_est_error,4), 'C_est_error = ', round(C_est_error,4))        

        # if self.episode == 2:
        #     stop 

        #----------- use the cvd_regr to predict the cvdrisk and sbp_feedback for the whole state-action space, that's the self.R_hat and self.C_hat
        for s in range(self.N_STATES):
            for a in self.ACTIONS[s]:
                state_code = self.state_index_to_code[s]
                action_code = self.action_index_to_code[a]

                # convert state and action code digit by digit to a list of 0 and 1
                state_code_list = [int(x) for x in list(state_code)]
                action_code_list = [int(x) for x in list(action_code)]
                # concatenate state and action code to the back of context vector
                R_input = np.concatenate((self.CONTEXT_VECTOR, np.array(state_code_list), np.array(action_code_list)), axis=0)
                # print('R_input: ', R_input)
                # print('R_input.shape: ', R_input.shape)

                y_pred = self.cvdrisk_regr.predict(R_input.reshape(1, -1))[0]
                #print('y_pred: ', y_pred)

                reward = 1/(1+np.exp(-y_pred))
                #print('reward: ', reward)

                self.R_hat[s][a] = reward

                C_input = np.concatenate((self.CONTEXT_VECTOR, np.array(action_code_list)), axis=0)
                cost = self.sbp_regr.predict(C_input.reshape(1, -1))[0]
                self.C_hat[s][a] = cost
                # print('cost: ', cost)
                # print('cost.shape: ', cost.shape)
                # print('s: ', s, 'a: ', a, 'reward: ', reward, 'cost: ', self.C[s][a])
                # stop        

        return (R_est_error, C_est_error)


    def add_design_vector(self, sbp_xa_vec):
        
        #------- for self.U_sbp
        design_vector = sbp_xa_vec

        # miltiply design_vector by its transpose
        design_vector_transpose = np.transpose(design_vector)
        product = np.matmul(design_vector, design_vector_transpose)
        # print('product.shape: ', product.shape)
        
        # add the new product to the existing U_sbp
        self.U_sbp = self.U_sbp + product
        # print('self.U_sbp.shape: ', self.U_sbp.shape)

        #------- for self.U_cvd, not doing here for each timestep, because we don't have estimated regression model for the first k0 episode
        # instead we do this in the compute_confidence_interval function


    def update_CONSTRAINT(self, new_CONSTRAINT):
        self.CONSTRAINT = new_CONSTRAINT

    def step(self, s, a, h):  # take a step in the environment
        # h is not used here

        probs = np.zeros((self.N_STATES))
        for next_s in range(self.N_STATES):
            probs[next_s] = self.P[s][a][next_s]
        next_state = int(np.random.choice(np.arange(self.N_STATES),1,replace=True,p=probs)) # find next_state based on the transition probabilities

        # add some noise to the reward and cost
        # rew = self.R[s][a] + np.random.normal(0, 0.1) # CVDRisk_feedback

        y_pred = self.R_y_pred[s][a]
        noise = np.random.normal(0, 0.05)
        #noise = 0
        obs_reward = 1.0 /(1.0+np.exp(-y_pred + noise)) # with noises added
        rew = obs_reward
        # print('y_pred: ', y_pred, 'noise: ', noise, 'obs_reward: ', obs_reward)
        # print('s: ', s, 'a: ', a, 'rew: ', rew, 'self.R[s][a]: ', self.R[s][a])
        # print('self.R[s][a]: ', self.R[s][a])

        #cost = self.C[s][a]
        cost = self.C[s][a] + np.random.normal(0, 5) # this is the SBP feedback, not the deviation

        return next_state, rew, cost


    def update_mu(self, init_state):
        self.mu = np.zeros(self.N_STATES)
        self.mu[init_state] = 1.0
        
    def setCounts(self,ep_count_p, ep_count): # add the counts of the current episode to the total counts
        for s in range(self.N_STATES):
            for a in self.ACTIONS[s]:
                self.NUMBER_OF_OCCURANCES[s][a] += ep_count[s, a]
                for s_ in range(self.N_STATES):
                    self.NUMBER_OF_OCCURANCES_p[s][a, s_] += ep_count_p[s, a, s_]


    # compute the confidence intervals beta for the transition probabilities
    def compute_confidence_intervals_2(self, ep, L_prime, mode): 
                                         # ep = L
    
        for s in range(self.N_STATES):
            for a in self.ACTIONS[s]:
                if self.NUMBER_OF_OCCURANCES[s][a] == 0:
                    self.beta_prob[s][a, :] = np.ones(self.N_STATES)
                    # self.beta_prob_T[s][a] = np.sqrt(ep/max(self.NUMBER_OF_OCCURANCES[s][a],1)) 
                    # not sure what is beta_prob_T used for? Used in other algorithms

                else:
                        
                    for s_1 in range(self.N_STATES):
                        
                        # if mode == 1:
                        # equation (5) in the paper to calculate the confidence interval for P
                        self.beta_prob[s][a,s_1] = min(2*np.sqrt(ep*self.P_hat[s][a][s_1]*(1-self.P_hat[s][a][s_1])/max(self.NUMBER_OF_OCCURANCES[s][a],1)) + 
                                                            14*ep/(3*max(self.NUMBER_OF_OCCURANCES[s][a],1)), 1)                            
                        # print('self.beta_prob[s][a,s_1]: ', self.beta_prob[s][a,s_1])

                self.sbp_confidence[s][a] = 2
                self.cvdrisk_confidence[s][a] = 0.05


    # compute the confidence intervals beta for the transition probabilities
    def compute_confidence_intervals(self, L): 

        if self.episode == 0:
            return -1, -1

        # ----------------- get U_cvd_inverse -----------------
        # reset the self.U_cvd
        self.U_cvd = np.zeros((self.CONTEXT_VEC_LENGTH+1+self.ACTION_CODE_LENGTH, self.CONTEXT_VEC_LENGTH+1+self.ACTION_CODE_LENGTH))

        XSA = np.concatenate((self.X, self.S, self.A), axis=1)
        # print('XSA.shape: ', XSA.shape)

        Y_pred = self.cvdrisk_regr.predict(XSA)
        # print('Y_pred.shape: ', Y_pred.shape)
        y_pred = 1.0/(1.0+np.exp(-Y_pred))

        for i in range(self.X.shape[0]):

            if i in self.U_xsa_prod_dict:
                prod = self.U_xsa_prod_dict[i]
            else:
                xsa_vec = XSA[i].reshape(-1, 1)
                prod = np.matmul(xsa_vec, np.transpose(xsa_vec))
                self.U_xsa_prod_dict[i] = prod

            factor = y_pred[i]

            u_cvd = prod * factor**2 * (1.0-factor)**2
            self.U_cvd = self.U_cvd + u_cvd
        

        # add an identy matrix to self.U_cvd if cannot be inverted
        try:
            U_cvd_inverse = np.linalg.inv(self.U_cvd)

            eigenvalues_cvd = np.linalg.eigvals(self.U_cvd)
            min_eigenvalue_cvd = np.min(eigenvalues_cvd)
            print("Minimum eigenvalue of U_cvd:", min_eigenvalue_cvd)    

        except:
            print('cannot invert U_cvd, add an identity matrix to it')
            eigenvalues_cvd = np.linalg.eigvals(self.U_cvd)
            min_eigenvalue_cvd = np.min(eigenvalues_cvd)
            print("Before adding Identity Matrix - Minimum eigenvalue of U_cvd:", min_eigenvalue_cvd)

            self.U_cvd = self.U_cvd + np.identity(self.CONTEXT_VEC_LENGTH+1+self.ACTION_CODE_LENGTH)
            U_cvd_inverse = np.linalg.inv(self.U_cvd)

            eigenvalues_cvd = np.linalg.eigvals(self.U_cvd)
            min_eigenvalue_cvd = np.min(eigenvalues_cvd)
            print("After adding Identity Matrix - Minimum eigenvalue of U_cvd:", min_eigenvalue_cvd)            

        # calculate the end_term 4 *hr/sqrt(t)
        hr = self.C2* np.sqrt(9+1+4)
        end_term = 4 * hr / np.sqrt(self.episode)
        print('end_term: ', end_term) # 4

        # ----------------- get U_sbp_inverse -----------------

        # add an identy matrix to self.U_sbp if cannot be inverted
        try:
            U_sbp_inverse = np.linalg.inv(self.U_sbp)

            eigenvalues_sbp = np.linalg.eigvals(self.U_sbp)
            min_eigenvalue_sbp = np.min(eigenvalues_sbp)
            print("Minimum eigenvalue of U_sbp:", min_eigenvalue_sbp)                    
        except:
            print('cannot invert U_sbp, add an identity matrix to it')
            eigenvalues_sbp = np.linalg.eigvals(self.U_sbp)
            min_eigenvalue_sbp = np.min(eigenvalues_sbp)
            print("Before adding Identity Matrix - Minimum eigenvalue of U_sbp:", min_eigenvalue_sbp)   

            self.U_sbp = self.U_sbp + np.identity(self.CONTEXT_VEC_LENGTH+self.ACTION_CODE_LENGTH)
            U_sbp_inverse = np.linalg.inv(self.U_sbp)

            eigenvalues_sbp = np.linalg.eigvals(self.U_sbp)
            min_eigenvalue_sbp = np.min(eigenvalues_sbp)
            print("After adding Identity Matrix - Minimum eigenvalue of U_sbp:", min_eigenvalue_sbp)
    

        for s in range(self.N_STATES):
            for a in self.ACTIONS[s]:
                if self.NUMBER_OF_OCCURANCES[s][a] == 0:
                    self.beta_prob[s][a, :] = np.ones(self.N_STATES)
                else:
                        
                    for s_1 in range(self.N_STATES):
                        # equation (5) in the paper to calculate the confidence interval for P
                        self.beta_prob[s][a,s_1] = min(2*np.sqrt(L*self.P_hat[s][a][s_1]*(1-self.P_hat[s][a][s_1])/max(self.NUMBER_OF_OCCURANCES[s][a],1)) + 
                                                            14*L/(3*max(self.NUMBER_OF_OCCURANCES[s][a],1)), 1)                            
                        # print('self.beta_prob[s][a,s_1]: ', self.beta_prob[s][a,s_1])

                #---------- compute the confidence intervals for the SBP_feedback
                xa_vec, xsa_vec = self.make_x_a_vector(self.CONTEXT_VECTOR, s, a)

                xa_transpose = np.transpose(xa_vec)
               
                prod = np.matmul(xa_transpose, U_sbp_inverse)
                prod = np.matmul(prod, xa_vec)

                # print('prod: ', prod, 'sqrt(prod): ', np.sqrt(prod))
                self.sbp_confidence[s][a] = self.C1 * np.log(self.episode) * np.sqrt(prod)
                # print('self.sbp_confidence[s][a]: ', self.sbp_confidence[s][a])


                #---------- compute the confidence intervals for the CVDRisk_feedback
                xsa_vec.reshape(1, -1)
                # print('xsa_vec.shape: ', xsa_vec.shape)
                # print('xsa_vec: ', xsa_vec)
                y_pred = self.cvdrisk_regr.predict(xsa_vec.reshape(1, -1))[0]
                # print('y_pred: ', y_pred)
                factor = 1.0/(1.0+np.exp(-y_pred))
                # print('factor: ', factor)
                xsa = xsa_vec * factor**2 * (1-factor)**2
                xsa_transpose = np.transpose(xsa)
                prod = np.matmul(xsa_transpose, U_cvd_inverse)
                prod = np.matmul(prod, xsa_vec)
                # print('prod: ', prod, 'sqrt(prod): ', np.sqrt(prod))
                self.cvdrisk_confidence[s][a] = self.C3 * np.log(self.episode) * np.sqrt(prod) + end_term
                # print('self.cvdrisk_confidence[s][a]: ', self.cvdrisk_confidence[s][a])
        
        print('self.sbp_confidence[1][1]: ', self.sbp_confidence[1][1])
        print('self.cvdrisk_confidence[1][1]: ', self.cvdrisk_confidence[1][1])

        return min_eigenvalue_cvd, min_eigenvalue_sbp


    # update the empirical/estimated model based on the counters every episode
    def update_empirical_model(self, ep): 
        # ep is not used here

        for s in range(self.N_STATES):
            for a in self.ACTIONS[s]:
                if self.NUMBER_OF_OCCURANCES[s][a] == 0:
                    self.P_hat[s][a] = 1/self.N_STATES*np.ones(self.N_STATES) # uniform distribution for unvisited state-action pairs
                else:
                    for s_1 in range(self.N_STATES):
                        self.P_hat[s][a][s_1] = self.NUMBER_OF_OCCURANCES_p[s][a,s_1]/(max(self.NUMBER_OF_OCCURANCES[s][a],1)) 
                        #calculate the estimated/empirical probabilities

                    self.P_hat[s][a] /= np.sum(self.P_hat[s][a]) # normalize the transition probabilities

                if abs(sum(self.P_hat[s][a]) - 1)  >  0.001: # sanity check  after updating the probabilities
                    print("empirical is wrong")
                    print(self.P_hat)
                    
    def update_empirical_rewards_costs(self, ep_emp_reward, ep_emp_cost):

        for s in range(self.N_STATES):
            for a in self.ACTIONS[s]:
                self.Total_emp_reward[s][a] = self.Total_emp_reward[s][a] + ep_emp_reward[s][a]
                self.R_hat[s][a] = self.Total_emp_reward[s][a]/(max(self.NUMBER_OF_OCCURANCES[s][a] ,1))
                
                self.Total_emp_cost[s][a] = self.Total_emp_cost[s][a] + ep_emp_cost[s][a]
                self.C_hat[s][a] = self.Total_emp_cost[s][a]/(max(self.NUMBER_OF_OCCURANCES[s][a], 1))
                        


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
        q = p.LpVariable.dicts("q", q_keys, lowBound=0, cat='Continuous') # q is the decision variable w_h(s,a) in the paper. Here uses a dict with keys (h,s,a)
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
                                                                                  
    # ++++++ solve for the optimal policy with constrained LP solver ++++++
    # also used to solve the baseline policy                               
    def compute_opt_LP_Constrained(self, ep, msg):

        # print("\nComputing optimal policy with constrained LP solver ...")

        opt_policy = np.zeros((self.N_STATES,self.EPISODE_LENGTH,self.N_ACTIONS)) #[s, h, a]
        opt_prob = p.LpProblem("OPT_LP_problem",p.LpMinimize) # minimize the CVRisk
        opt_q = np.zeros((self.EPISODE_LENGTH, self.N_STATES, self.N_ACTIONS)) #[h, s, a]
                                                                                  
        # create problem variables
        q_keys = [(h, s, a) for h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in self.ACTIONS[s]]
                                                                                          
        q = p.LpVariable.dicts("q", q_keys, lowBound=0, cat='Continuous')

        # objective function
        opt_prob += p.lpSum([q[(h,s,a)]*self.R[s][a] 
                            for h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in self.ACTIONS[s]]) 
            
        # constraints
        # sbp within the range [110, 125] for all time steps
        opt_prob += p.lpSum([q[(h,s,a)] * (max(110-self.C[s][a], 0) + max(self.C[s][a]-125, 0)) 
                    for h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in self.ACTIONS[s]]) - self.CONSTRAINT <= 0 

        # opt_prob += p.lpSum([q[(h,s,a)] * self.C[s][a] for h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in self.ACTIONS[s]]) - self.CONSTRAINT <= 0 
            
        for h in range(1,self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                q_list = [q[(h,s,a)] for a in self.ACTIONS[s]]
                pq_list = [self.P[s_1][a_1][s]*q[(h-1,s_1,a_1)] for s_1 in range(self.N_STATES) for a_1 in self.ACTIONS[s_1]]
                opt_prob += p.lpSum(q_list) - p.lpSum(pq_list) == 0 # equation 17(c)

        for s in range(self.N_STATES):
            q_list = [q[(0,s,a)] for a in self.ACTIONS[s]]
            opt_prob += p.lpSum(q_list) - self.mu[s] == 0 # equation 17(d), initial state is fixed

        if self.use_gurobi:
            status = opt_prob.solve(p.GUROBI_CMD(msg = 0))
        else:
            status = opt_prob.solve(p.PULP_CBC_CMD(gapRel=0.001, msg = 0)) # solve the constrained LP problem


        #print(status)
        #print(p.LpStatus[status])   # The solution status
        if p.LpStatus[status] != 'Optimal':
            print("No optimal solution found!")
            return None, 0, 0, 0, p.LpStatus[status]

        #print(opt_prob)
        # print("printing best value constrained:", p.value(opt_prob.objective))
        # print(p.value(opt_prob.objective))
                                                                                                                  
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
                    
                # con_policy  += opt_q[h,s,a]*self.C[s][a]
                con_policy  += opt_q[h,s,a]*(max(0, 110-self.C[s][a]) + max(0, self.C[s][a]-125)) # since the cost here is the SBP feedback

                val_policy += opt_q[h,s,a]*self.R[s][a]

        # combine following 3 print statements to 1 line
        #print('opt_prob.objective:', p.value(opt_prob.objective))
        print(msg, p.LpStatus[status], 
                "- best value constrained:", round(p.value(opt_prob.objective),4), 
                ", value from the conLPsolver: value of policy =", round(val_policy,4), 
                ", cost of policy =", round(con_policy,4))

        # evaluate the optimal policy using finite horizon policy evaluation
        q_policy, value_of_policy, cost_of_policy = self.FiniteHorizon_Policy_evaluation(self.P, opt_policy, self.R, self.C) 
                                                                                                                                                                          
        return opt_policy, value_of_policy, cost_of_policy, q_policy, p.LpStatus[status]
                                                                                                                                                                                  
                                                                                                                                                                                  
                                                                                                                                                                                  
    # ++++ compute the optimal policy using the extended Linear Programming +++
    def compute_extended_LP(self,):
        """
        - solve equation (10) CMDP using extended Linear Programming
        - optimal policy opt_policy[s,h,a] is the probability of taking action a at state s at time h
        - evaluate optimal policy using finite horizon policy evaluation, to get 
            - value_of_policy: expected cumulative value, [s,h] 
            - cost_of_policy: expected cumulative cost, [s,h]
            - q_policy: expected cumulative rewards for [s,h,a]
        """

        opt_policy = np.zeros((self.N_STATES, self.EPISODE_LENGTH, self.N_ACTIONS)) #[s,h,a]
        opt_prob = p.LpProblem("OPT_LP_problem", p.LpMinimize) # minimize the expected cumulative CVDRisk
        opt_z = np.zeros((self.EPISODE_LENGTH, self.N_STATES, self.N_ACTIONS, self.N_STATES)) # [h,s,a,s_], decision variable, state-action-state occupancy measure
        #create problem variables
        
        z_keys = [(h,s,a,s_1) for h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in self.ACTIONS[s] for s_1 in self.Psparse[s][a]]
        z = p.LpVariable.dicts("z_var", z_keys, lowBound=0, upBound=1, cat='Continuous') 
        # why the upperbound is 1? Because the Z is essentially probability, and the probability is between 0 and 1
        # lower bound is 0, because the occupancy measure is non-negative, constraint (18e) in the paper
            
        r_k = {}
        c_k = {}
        for s in range(self.N_STATES):
            l = len(self.ACTIONS[s])
            r_k[s] = np.zeros(l)
            c_k[s] = np.zeros(l)

            for a in self.ACTIONS[s]:
                # r_k[s][a] = self.R_hat[s][a] - self.sbp_cvdrisk_confidence[s][a] # no need to times the self.episode_length since self.sbp_cvdrisk_confidence[s][a] is not visit dependent
                r_k[s][a] = max(0, self.R_hat[s][a] - self.alpha_r * self.cvdrisk_confidence[s][a])

                c_k[s][a] = max(0, 110-(self.C_hat[s][a] - self.alpha_c * self.sbp_confidence[s][a])) + max(0, (self.C_hat[s][a] + self.alpha_c * self.sbp_confidence[s][a]) - 125)

        # objective function
        # equation (18a) in the paper
        opt_prob += p.lpSum([z[(h,s,a,s_1)]*r_k[s][a] for h in range(self.EPISODE_LENGTH) 
                                                           for s in range(self.N_STATES) 
                                                           for a in self.ACTIONS[s] 
                                                           for s_1 in self.Psparse[s][a]])

        # Constraints equation 18(b)                                  
        opt_prob += p.lpSum([z[(h,s,a,s_1)]* (c_k[s][a]) 
                                            for h in range(self.EPISODE_LENGTH) 
                                            for s in range(self.N_STATES) 
                                            for a in self.ACTIONS[s] 
                                            for s_1 in self.Psparse[s][a]]) - self.CONSTRAINT <= 0
        
        for h in range(1,self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                z_list = [z[(h,s,a,s_1)] for a in self.ACTIONS[s] for s_1 in self.Psparse[s][a]]
                z_1_list = [z[(h-1,s_1,a_1,s)] for s_1 in range(self.N_STATES) for a_1 in self.ACTIONS[s_1] if s in self.Psparse[s_1][a_1]]
                opt_prob += p.lpSum(z_list) - p.lpSum(z_1_list) == 0 # constraint (18c) in the paper

                                                                                                                                                                                                             
        for s in range(self.N_STATES):
            q_list = [z[(0,s,a,s_1)] for a in self.ACTIONS[s] for s_1 in self.Psparse[s][a]]
            opt_prob += p.lpSum(q_list) - self.mu[s] == 0 # constraint (18d) in the paper
                                                                                                                                                                                                                      
        for h in range(self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                for a in self.ACTIONS[s]:
                    for s_1 in self.Psparse[s][a]:                
                        opt_prob += z[(h,s,a,s_1)] - (self.P_hat[s][a][s_1] + self.alpha_p * self.beta_prob[s][a,s_1]) *  p.lpSum([z[(h,s,a,y)] for y in self.Psparse[s][a]]) <= 0  # equation (18f)
                        opt_prob += -z[(h,s,a,s_1)] + (self.P_hat[s][a][s_1] - self.alpha_p * self.beta_prob[s][a,s_1])* p.lpSum([z[(h,s,a,y)] for y in self.Psparse[s][a]]) <= 0 # equation (18g)
                                                                                                                                                                                                                                     
        if self.use_gurobi:
            status = opt_prob.solve(p.GUROBI_CMD(gapRel=0.01, msg = 0)) # solve the Extended LP problem
        else:
            status = opt_prob.solve(p.PULP_CBC_CMD(gapRel=0.01, msg = 0)) # solve the Extended LP problem

                                                                                                                                                                                                                                      
        if p.LpStatus[status] != 'Optimal':
            # print(p.LpStatus[status])
            return np.zeros((self.N_STATES, self.EPISODE_LENGTH, self.N_ACTIONS)), np.zeros((self.N_STATES, self.EPISODE_LENGTH)), np.zeros((self.N_STATES, self.EPISODE_LENGTH)), p.LpStatus[status], np.zeros((self.N_STATES, self.EPISODE_LENGTH, self.N_ACTIONS))
                                                                                                                                                                                                                                                  
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


    def compute_extended_LP_random(self,):

        # assign uniform probability to opt_policy
        opt_policy = np.zeros((self.N_STATES, self.EPISODE_LENGTH, self.N_ACTIONS)) #[s,h,a]
        for s in range(self.N_STATES):
            for h in range(self.EPISODE_LENGTH):
                for a in self.ACTIONS[s]:
                    opt_policy[s,h,a] = 1/len(self.ACTIONS[s])

        q_policy, value_of_policy, cost_of_policy = self.FiniteHorizon_Policy_evaluation(self.P, opt_policy, self.R, self.C)
                                                                                                                                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                                                                                  
        return opt_policy, value_of_policy, cost_of_policy, 'Optimal', q_policy


    
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

    # ++++ Finite Horizon Policy Evaluation ++++
    def FiniteHorizon_Policy_evaluation(self, Px, policy, R, C):
        
        # results to be returned
        q = np.zeros((self.N_STATES, self.EPISODE_LENGTH, self.N_ACTIONS)) # q(s,h,a), q_policy, expected cumulative rewards
        v = np.zeros((self.N_STATES, self.EPISODE_LENGTH)) # v(s,h), expected cumulative value of the calculated optimal policy
        c = np.zeros((self.N_STATES, self.EPISODE_LENGTH)) # c(s,h), expected cumulative cost of the calculated optimal policy

        P_policy = np.zeros((self.N_STATES,self.EPISODE_LENGTH,self.N_STATES)) # P_policy(s,h,s_1), probability of being in state s_1 at time h+1 given that we are in state s at time h and we follow the optimal policy
        R_policy = np.zeros((self.N_STATES,self.EPISODE_LENGTH)) # R_policy(s,h), expected reward of being in state s at time h given that we follow the optimal policy
        C_policy = np.zeros((self.N_STATES,self.EPISODE_LENGTH)) # C_policy(s,h), expected cost of being in state s at time h given that we follow the optimal policy

        # initialize the last state for the value and cost, and q
        for s in range(self.N_STATES):
            x = 0
            for a in self.ACTIONS[s]:
                # x += policy[s, self.EPISODE_LENGTH - 1, a]*C[s][a] # expected cost of the last state
                x += policy[s, self.EPISODE_LENGTH - 1, a]* (max(0, 110-C[s][a]) + max(0, C[s][a]-125)) 
            c[s, self.EPISODE_LENGTH-1] = x #np.dot(policy[s,self.EPISODE_LENGTH-1,:], self.C[s])

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
                    # y += policy[s,h,a]*C[s][a]
                    y += policy[s,h,a]*(max(0, 110-C[s][a]) + max(0, C[s][a]-125))
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
                c[s,h] = C_policy[s,h] + np.dot(P_policy[s,h,:], c[:,h+1]) # expected cumulative cost of the state s at time h under the policy = expected cost of the state s at time h under the policy + expected cumulative cost of the state s at time h+1 under the policy
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
