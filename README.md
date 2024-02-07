# Accord-RL
Safe RL for Healthcare

[Slides](https://docs.google.com/presentation/d/1FEpQrTIzbyv5q6Vs8AI1u_axD3H4F-SUG1MbFUCMjrk/edit#slide=id.g24666924b86_1_29)

### To install environment
`conda env create -f environment_linux.yml` or `conda env create -f environment_Mac.yml`

`conda activate tf`  

### Install Gurobi solver if desired

Using Gurobi solver is about 25% faster than Pulp solver in our experiment.

`conda config --add channels https://conda.anaconda.org/gurobi`  
`conda install gurobi`   

Run the license get commandline after applying for academic license in your account.  

`conda remove gurobi`

To export current environment: `conda env export > environment.yml`

---

In this work, we explored 3 cases: BPClass only, BGClass only, and BPClass+BGClass (Top2 BP+ Top2 BG, Top4 BP+ Top4 BG). Navigate to the corresponding folder to run the codes. The following commands should work for all 3 cases.

We compared the Contextual/COPS algorithm with DOPE, along with other two baselines, OptCMDP and OptPessLP.


#### Contextual/COPS

1. Data file used:
   * BPClass: `BPClass/data/ACCORD_BPClass_v2_merged_Contextual.csv`
   * BGClass: `BGClass/data/ACCORD_BGClass_v2_merged.csv`
   * BPClass+BGClass: `BPBGClass/data/ACCORD_BPBGClass_v2_merged.csv`
   * `create_datasets_contextual.ipynb` is copied from the scripts when preparing the ACCORD dataset, it contains some steps to prepare the data file (used in developing BPClass, not used for BG and BPBG)
   
2. Train true offline models for P, R, C using all data
   * Run `model_contextual.ipynb`: 
     * take above input datafile
     * discretize the context features
     * basic model settings, action space, state space etc.
     * P is estimated the same way (empirical estimates) as DOPE, output model settings to `output/model_contextual.pkl`
  
   * Run `train_feedback_estimators.ipynb`：get the R and C offline estimators
     * takes `../data/ACCORD_BPBGClass_v2_contextual.csv` as input
     * R / CVDRisk: logistic regression with (context_vector, state, action) 
     * C / SBP or Hba1c: linear regression with (context_vector, action) 
     * Offline R and C models are saved to `output/CVDRisk_estimator.pkl`, `output/A1C_feedback_estimator.pkl`, `output/SBP_feedback_estimator.pkl`

3. Run `python Contextual.py` or `python Contextual.py 1` to run the main contextual algorithm. Use 1 to specify using GUROBI solver.
   
4. Run `python plot1.py output/CONTEXTUAL_opsrl15.pkl 30000` to plot all plots in the same figure, specify the filename and episodes to plot
   
5. `test.ipynb` is used to debug the code.


#### DOPE

1. Run `model.ipynb` to: 
   * data file used for each case:
     * BPClass: `../data/ACCORD_BPClass_v2.csv`
     * BGClass: `../data/ACCORD_BGClass_v2.csv`
     * BPClass+BGClass: `../data/ACCORD_BPBGClass_v2.csv`
  
   * set up the state features and action features for state space and action space
   * get empriical estimates of the P R C based on the dataset, save it to `output/model.pkl`
   * set up CONSTRAINTS and baseline Constraint
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


#### OptCMDP

1. Use the same model preparation scheme as in DOPE by running `model.ipynb`. If you have run this for DOPE, no need to run this again.
2. Run `python OptCMDP.py 1` to run the OptCMDP algorithm
   * Similar to DOPE, but not running K0 episodes for baseline policy. Instead, it solves Extended LP directly.
   * Choose random policy for the first episode to get started
3. `python plot1.py output/OptCMDP_opsrl10.pkl 10000`
   * should expect to see non-zero Constraint Regret

#### OptPessLP

1. Use the same model preparation scheme as in DOPE by running `model.ipynb`. If you have run this for DOPE, no need to run again.
2. Run `python OptPessLP.py` to run the algorithm
   * Similar to DOPE, but select to run baseline policy based on estimated cost of baseline policy
   * If estimated cost is too high, then run baseline, else solve the Extended LP
   * Radius is larger, and has no tunning parameter
3. `python plot1.py output/OptPessLP_opsrl10.pkl 10000`
   * should expect to see increasing linear Objective Regret with episodes, and 0 Connstraint Regret


#### Plot Regrets Comparison

* All models are run for 3e4 episodes, navigate to each folder and run `python plot_all.py 30000` to compare all models.

* Final results for all 3 cases are stored in folders `output_final/`.


#### Compare COPS with Clinician actions and feedback

1. Navigate to `/Contextual` folder for each of the 3 cases, run the `model_contextual.ipynb` again to generate 
   * `CONTEXT_VECTOR_dict`: dict with each patient's MaskID as key, and values of `context_fea`, `init_state`, `state_index_recorded`, `action_index_recorded`
   * save the trained `knn_model` and `knn_model_label` and `scaler` to selecting the actions of Clinician with unmatching states

2. Run `python Contextual_test.py 1` to run the simulation of COPS and Clinician
   * The difference between solving constrained LP problem and solving the extended LP problem is that, in the extended LP problem, we need to add equations that relates to the confidence bound of the estimated P_hat. Here when testing, we still call the compute_extended_LP() function, but we do not add the confidence bound equations by settings the Inference to True.
   * the results are saved to `output_final/Contextual_test_BPClass.csv`

3. Copied the generated simulation output csv files to `NumericalResults/` folder
   
4. Run `Table_and_Plots_v3_LP_samepatient.ipynb` to get the plots and tables.

---
2023/08/17 Thursday

1. plot the regrets plots again by changing the blue background to white background
```bash
# navigate to each DOPE folder for each case, BPClass, BGClass, BPBGClass
python plot_all.py 30000

# for BPBGClass (2+2), plot the regrets separately
python plot_all_separate.py 30000

```

2023/08/23 Wednesday

1. Finished the scripts for running COPS-MM again with use SBP and A1C range, updated the range codes manually. Run the simulation test to get the simulation results.
   * `python Contextual.py 1`
   * `python Contextual_test.py 1`
2. Copy the simulation results to the `NumericalResults/` folder and run the analysis notebook


2024/01/31 Wednesday
1. Created the `Contextual_SBP90130_A1C7.9_shuffle` folder to run the COPS-MM with shuffled patients
2. The time cost is a big issue here, I did not implement the code to record a patient's history record but just randomly sample. I think this would be the same as randomly sample a patient from the dataset. And even if so, we only have like 3000 patients rather than the 3e4 episodes we used to have in the sequential case.
```bash
# use the old non-shuffled base models and pkl files
# run COP-MM with shuffled patients, DO NOT use GUROBI solver as it is making trouble, unsolved base policy, will keep skipping to next patients

# here I set the k0=200, which I could have used K0=-1
python Contextual.py 

# plot the regrets
python plot1.py output/CONTEXTUAL_opsrl100.pkl 2500

```

2024/02/06 Tuesday
1. Created the `Contextual_batch.py` for running the batch update of COPS-MM. For efficiency consideration (speed, RAM), I implemented the patients pool to define how many patients can interact. The batch update is conducted within the patients pool util the patients pool depletes. Then we re-sample the patients pool and continue the batch update. 
```bash
# in HPRC server
python Contextual_batch.py

# in LENSS server, use gurobi solver
python Contextual_batch.py 1
```
