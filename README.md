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


#### BPClass-DOPE

1. Run `model.ipynb` to: 
   * get empriical estimates of the P R C based on the dataset, save it to `model.pkl`
   * solve the optimal policy, save it to `solution.pkl`
   * solve the baseline policy, save it to `base.pkl`

2. The `UtilityMethods.py` defines the `utils` class, which does:
   * linear programming solver
   * update the empirical estimate of P during exploration
  
3. Then run `python DOPE.py` to run the main DOPE algorithm:
   * Learns the objective regrets, and constraint regrets of the learned policy
   * save `opsrl_RUNNUMBER.pkl` and `regrets_RUNNUMBER.pkl`

4. Run `python plot.py` to plot the `Objective Regret` and `Constraint Regret`
   * run `python plot.py 40000` to specify the plot the first 3000 episodes
   * plots are in `output/` folder


#### BPClass-Contextual

1. Data file used is `../data/ACCORD_BPClass_v2_merged_Contextual.csv`
   
2. Train true model for P, R, C using all data
   * P is estimated the same way (empirical estimates) as DOPE, using `model_conntextual.ipynb`
   * R / CVDRisk: logistic regression with (context_vector, state, action)
   * C / SBP: linear regression with (context_vector, state, action)
   * Get the R and C offline estimators by running `train_feedback_estimators.ipynb`
   * Offline R and C models are saved to `output/`

3. Run `python Contextual.py` to run the main contextual algorithm
   
4. Run `python plot1.py output/opsrl10.pkl 1000` to plot all plots in the same figure, specify the filename and episodes to plot
   
5. `test.ipynb` is used to debug the code


