# Accord-RL
Safe RL for Healthcare

### To install environment
`conda env create -f environment.yml`

To export current environment: `conda env export > environment.yml`

#### BPClass-DOPE

1. Run `model.ipynb` to: 
   * get empriical estimates of the P R C based on the dataset, save it to `model.pkl`
   * solve the optimal policy, save it to `solution.pkl`
   * solve the baseline policy, save it to `base.pkl`

2. The `UtilityMethods.py` defines the `utils` class, which does:
   * linear programming solver
   * update the empirical estimate of P during exploration
  
3. Then run `python DOPE.py` to run the main algorithm:
   * Learns the objective regrets, and constraint regrets of the learned policy
   * save `opsrl_RUNNUMBER.pkl` and `regrets_RUNNUMBER.pkl`

4. Run `python plot.py` to plot the `Objective Regret` and `Constraint Regret`
   * run `python plot.py 3000` to specify the plot the first 3000 episodes
   * plots are in `output/` folder