<div align="center">
<h1>Safe Reinforcement Learning with Contextual<br>Information: Theory and Applications</h1>

[**Junyu Cao**](https://junyucao.com/)<sup>1*</sup> · [**Esmaeil Keyvanshokooh**](https://ekshokooh.github.io/)<sup>2*</sup> · [**Tian Liu**](https://tian1327.github.io/)<sup>2&dagger;</sup>

<sup>1</sup>University of Texas at Austin&emsp;&emsp;&emsp;<sup>2</sup>Texas A&M University
<br>
*equal contribution&emsp; &dagger;code implementation

<a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4583667"><img src='https://img.shields.io/badge/SSRN-Paper-red' alt='Paper PDF'></a>
</div>

This repository contains the implementation of the `Contextual Optimistic-Pessimistic Safe exploration (COPS)` reinforcement learning algorithm. We also implement baseline safe RL methods, including DOPE, OptCMDP, and OptPessLP. The implementation is tested on the ACCORD dataset for blood pressure and blood glucose management, as well as the inventory control problem.


<!-- ![teaser](assets/teaser_v7.png) -->

## News
- **2024-12-16:** COPS code released.
- **2023-09-25:** [ssrn paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4583667) released.


## Usage

### Install conda environment
```bash
# for Linux machine
conda env create -f environment_linux.yml

# or for Mac
conda env create -f environment_Mac.yml

# activate the environment
conda activate cops
```
You might need to install [Gurobi optimization solver](https://www.gurobi.com/academia/academic-program-and-licenses/) if desired. 
```bash
conda config --add channels https://conda.anaconda.org/gurobi
conda install gurobi

# run the license get commandline after applying for academic license in your account.

# to remove gurobi solver, use
conda remove gurobi
```

### Running Experiments

In this work, for ACCORD dataset, we explored 3 cases: BPClass only, BGClass only, and BPClass+BGClass (Top2 BP + Top2 BG, Top4 BP + Top4 BG). Navigate to the corresponding folder to run the codes. The following commands should work for all 3 cases.


#### Data Preparation
Create the corresponding datasets for each case by running the `create_datasets_contextual.ipynb` script in each folder. This will create the datasets for the corresponding case which are saved in the `data/` folder. 

---
#### Running COPS

1. Data file used:
   * BPClass: `BPClass/data/ACCORD_BPClass_v2_merged_Contextual.csv`
   * BGClass: `BGClass/data/ACCORD_BGClass_v2_merged.csv`
   * BPClass+BGClass: `BPBGClass/data/ACCORD_BPBGClass_v2_merged.csv`
   * `create_datasets_contextual.ipynb` is copied from the scripts when preparing the ACCORD dataset, it contains some steps to prepare the data file (used in developing BPClass, not used for BG and BPBG)
   
2. Train true offline models for P, R, C using all data
   * Run `model_contextual.ipynb`: 
     * take above input data files
     * discretize the context features
     * basic model settings, action space, state space etc.
     * P is estimated the same way (empirical estimates) as DOPE, output model settings to `output/model_contextual.pkl`
  
   * Run `train_feedback_estimators.ipynb`：get the R and C offline estimators
     * takes `../data/ACCORD_BPBGClass_v2_contextual.csv` as input
     * R (CVDRisk): logistic regression with (context_vector, state, action) 
     * C (SBP or Hba1c): linear regression with (context_vector, action) 
     * Offline R and C models are saved to `output/CVDRisk_estimator.pkl`, `output/A1C_feedback_estimator.pkl`, `output/SBP_feedback_estimator.pkl`

3. Run `python Contextual.py` or `python Contextual.py 1` to run the main contextual algorithm. Use 1 to specify using GUROBI solver.
   
4. Run `python plot1.py output/CONTEXTUAL_opsrl15.pkl 30000` to plot all plots in the same figure, specify the filename and episodes to plot
   
5. `test.ipynb` is used to debug the code.

---

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

---
#### OptCMDP

1. Use the same model preparation scheme as in DOPE by running `model.ipynb`. If you have run this for DOPE, no need to run this again.
2. Run `python OptCMDP.py 1` to run the OptCMDP algorithm
   * Similar to DOPE, but not running K0 episodes for baseline policy. Instead, it solves Extended LP directly.
   * Choose random policy for the first episode to get started
3. `python plot1.py output/OptCMDP_opsrl10.pkl 10000`
   * should expect to see non-zero Constraint Regret


---
#### OptPessLP

1. Use the same model preparation scheme as in DOPE by running `model.ipynb`. If you have run this for DOPE, no need to run again.
2. Run `python OptPessLP.py` to run the algorithm
   * Similar to DOPE, but select to run baseline policy based on estimated cost of baseline policy
   * If estimated cost is too high, then run baseline, else solve the Extended LP
   * Radius is larger, and has no tunning parameter
3. `python plot1.py output/OptPessLP_opsrl10.pkl 10000`
   * should expect to see increasing linear Objective Regret with episodes, and 0 Connstraint Regret

---
#### Plot Regrets Comparison

* All models are run for 3e4 episodes, navigate to each folder and run `python plot_all.py 30000` to compare all models.

* Final results for all 3 cases are stored in folders `output_final/`.

---
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
Refer to the file [tian_log.md](tian_log.md) for more detailed examples of running the code.


## Acknowledgment
This code base is developed based on the following project. We sincerely thank the authors for open-sourcing their project.

- [DOPE](https://github.com/archanabura/DOPE-DoublyOptimisticPessimisticExploration)

## Citation

If you find our project useful, please consider citing:

in latex format:
```bibtex
@article{cao2023safe,
  title={Safe Reinforcement Learning with Contextual Information: Theory and Applications},
  author={Cao, Junyu and Keyvanshokooh, Esmaeil and Liu, Tian},
  journal={SSRN: https://ssrn.com/abstract=4583667},
  year={2023}
}
```
or in word format:
```
Cao, Junyu and Keyvanshokooh, Esmaeil and Liu, Tian, Safe Reinforcement Learning with Contextual Information: Theory and Applications (September 25, 2023). Available at SSRN: https://ssrn.com/abstract=4583667 or http://dx.doi.org/10.2139/ssrn.4583667
```