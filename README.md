# Accord-RL
Safe RL for Healthcare

### To install environment
`conda env create -f environment_Linux.yml` or `conda env create -f environment_Mac.yml`


### Install Gurobi solver
`conda config --add channels https://conda.anaconda.org/gurobi`  
`conda install gurobi`   

Run the license get commandline after applying for academic license in your account.  

`conda remove gurobi`

To export current environment: `conda env export > environment.yml`


---
### BPClass

Codes located in`BPClass/` folder.

#### BPClass_Contextual

1. Data file used is `../data/ACCORD_BPClass_v2_merged_Contextual.csv`
   * `create_datasets_contextual.ipynb` is copied from the scripts when processing the data, it contains some steps to prepare the data file
   
2. Train true model for P, R, C using all data
   * `model_conntextual.ipynb`: 
     * discretize the context features
     * P is estimated the same way (empirical estimates) as DOPE, output model settings to `output/model_contextual.pkl`
  
   * `train_feedback_estimators.ipynb`：get the R and C offline estimators
     * R / CVDRisk: logistic regression with (context_vector, state, action)
     * C / SBP or Hba1c: linear regression with (context_vector, action)
     * Offline R and C models are saved to `output/CVDRisk_estimator.pkl`, `output/A1C_feedback_estimator.pkl`

3. Run `python Contextual.py` or `python Contextual.py 1` to run the main contextual algorithm. Use 1 to specify using GUROBI solver.
   
4. Run `python plot1.py output/CONTEXTUAL_opsrl15.pkl 30000` to plot all plots in the same figure, specify the filename and episodes to plot
   
5. `test.ipynb` is used to debug the code



#### BPClass_DOPE

1. Run `model.ipynb` to: 
   * data file used for each case:
     * BPClass: `../data/ACCORD_BPClass_v2.csv`
     * BGClass: `../data/ACCORD_BGClass_v2.csv`
     * BPBGClass: `../data/ACCORD_BPBGClass_v2.csv`
  
   * set up the state features and action features for state space and action space
   * get empriical estimates of the P R C based on the dataset, save it to `output/model.pkl`
   * solve the optimal policy, save it to `output/solution.pkl`
   * solve the baseline policy, save it to `output/base.pkl`

2. The `UtilityMethods.py` defines the `utils` class, which does:
   * linear programming solver
   * update the empirical estimate of P during exploration
  
3. Then run `python DOPE.py` to run the main DOPE algorithm, use `python DOPE.py 1` to specify using GUROBI solver.
   * Learns the objective regrets, and constraint regrets of the learned policy
   * save `DOPE_opsrl_RUNNUMBER.pkl` and `regrets_RUNNUMBER.pkl`

4. Plot the `Objective Regret` and `Constraint Regret`
   * run `python plot1.py output/DOPE_opsrl150.pkl 30000`
   * plots are in `output/` folder


#### BPClass_OptCMDP

1. Use the same model preparation scheme as in DOPE by running `model.ipynb`
2. Run `python OptCMDP.py` to run the OptCMDP algorithm
   * Similar to DOPE, but not running K0 episodes for baseline policy. Instead, it solves Extended LP directly.
   * Choose random policy for the first episode to get started
3. `python plot1.py output/OptCMDP_opsrl10.pkl 10000`
   * should expect to see non-zero Constraint Regret

#### BPClass_OptPessLP

1. Use the same model preparation scheme as in DOPE by running `model.ipynb`
2. Run `python OptPessLP.py` to run the algorithm
   * Similar to DOPE, but select to run baseline policy based on estimated cost of baseline policy
   * If estimated cost is too high, then run baseline, else solve the Extended LP
   * Radius is larger, and has no tunning parameter
3. `python plot1.py output/OptPessLP_opsrl10.pkl 10000`
   * should expect to see increasing linear Objective Regret with episodes, and 0 Connstraint Regret


#### BPClass Plot Regrets Comparison

* All models are run for 3e4 episodes, `python plot_all.py 30000` 


---
### BGClass

Codes located in`BGClass/` folder.

#### BGClass_Contextual



#### BGClass_DOPE

1. Run `model.ipynb` to: 
   * data file used is `../data/ACCORD_BGClass_v2.csv`
   * set up the state features and action features for state space and action space
   * the merged state levels are saved to `../data/ACCORD_BGClass_v2_merged.csv`, i.e. `hba1c_discrete` --> `hba1c_discrete_merged` 
   * get empriical estimates of the P R C based on the dataset, save it to `model.pkl`
   * solve the optimal policy, save it to `solution.pkl`
   * solve the baseline policy, save it to `base.pkl`

2. The `UtilityMethods.py` defines the `utils` class, which does:
   * linear programming solver
   * update the empirical estimate of P, R, C during exploration
  
3. `python DOPE.py` or `python DOPE.py 1` to specify using GUROBI solver.

4. `python plot1.py output/DOPE_opsrl20.pkl 1000`

#### BGClass_OptCMDP

1. Use the same model preparation scheme as in DOPE by running `model.ipynb`
2. Run `python OptCMDP.py` to run the OptCMDP algorithm
   * Similar to DOPE, but not running K0 episodes for baseline policy. Instead, it solves Extended LP directly.
   * The radius calculation is different from DOPE
   * Choose random policy for the first episode to get started
3. `python plot1.py output/OptCMDP_opsrl10.pkl 10000`
   * should expect to see better sublinear Objective Regret and non-zero Constraint Regret


#### BGClass_OptPessLP

1. Use the same model preparation scheme as in DOPE by running `model.ipynb`
2. Run `python OptPessLP.py` to run the algorithm
   * Similar to DOPE, but select to run baseline policy based on estimated cost of baseline policy
   * If estimated cost is too high, then run baseline, else solve the Extended LP
   * has higher pessimism (larger radius)
   * Radius is larger, and has no tunning parameter
3. `python plot1.py output/OptPessLP_opsrl10.pkl 10000`
   * should expect to see increasing linear Objective Regret with episodes, and 0 Connstraint Regret

---

#### BP+BGClass_DOPE

#### BP+BGClass_Contextual


#### BP+BGClass_OptCMDP


#### BP+BGClass_OptPessLP