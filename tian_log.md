2023/08/17 Thursday

1. plot the regrets plots again by changing the blue background to white background
```bash
# navigate to each DOPE folder for each case, BPClass, BGClass, BPBGClass
python plot_all.py 30000

# for BPBGClass (2+2), plot the regrets separately
python plot_all_separate.py 30000

```
---
2023/08/23 Wednesday

1. Finished the scripts for running COPS-MM again with use SBP and A1C range, updated the range codes manually. Run the simulation test to get the simulation results.
   * `python Contextual.py 1`
   * `python Contextual_test.py 1`
2. Copy the simulation results to the `NumericalResults/` folder and run the analysis notebook

---
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
---
2024/02/06 Tuesday
1. Created the `Contextual_batch.py` for running the batch update of COPS-MM. For efficiency consideration (speed, RAM), I implemented the patients pool to define how many patients can interact. The batch update is conducted within the patients pool util the patients pool depletes. Then we re-sample the patients pool and continue the batch update. 
```bash
# in HPRC server
cd BPBGClass_4+4/Contextual_SBP90130_A1C7.9_shuffle/
python Contextual_batch.py
python plot1.py output/CONTEXTUAL_opsrl100.pkl 2500

# running 4+4 in LENSS server, use gurobi solver
cd BPBGClass_4+4/Contextual_SBP90130_A1C7.9_shuffle_grobi/
python Contextual_batch.py 1
python plot1.py output/CONTEXTUAL_opsrl100.pkl 2500
```
2. Note that, very importantly, the regre plot we show in the paper is from BPBG Contextual 2+2 case, since it has better plots than 4+4 case. The regret plots for 4+4 is ugly and prof have decided to just use the one from the 2+2 case.

```bash
cd /scratch/user/ltmask/Accord-RL/BPBGClass/Contextual_shuffle
python Contextual_batch.py
```
---
2024/02/13 Tuesday
1. Implemented the overlapping batch update with fixed patient interval, use the BPBGClass 2+2 case
```bash
cd BPBGClass/Contextual_shuffle/
python Contextual_batch.py 20
python Contextual_batch.py 40
python Contextual_batch.py 100

# plot the comparison of batchsize
python plot_all_separate.py 3000
```
---
2024/02/14 Wednesday
1. Implemented the overlapping batch update with fixed patient interval, use the BPBGClass 2+2 case, define batchsize as # of finished patients
```bash
cd BPBGClass/Contextual_shuffle/
python Contextual_batch_numpatient.py 10
python Contextual_batch_numpatient.py 20
python Contextual_batch_numpatient.py 40
python Contextual_batch_numpatient.py 100

# plot the test results
python plot1.py output_bs10/CONTEXTUAL_opsrl100.pkl 40

# plot the comparison of batchsize
python plot_all_separate.py 3000
```
---
2024/02/16 Friday
1. Plot the batch update comparison COPS-MM vs. batch
```bash
cd Accord-RL/BPBGClass/Contextual_shuffle
python plot_all_separate.py 
```
---
2024/02/24-25 Sat- Sun
1. Setup the inventory control problem, run DOPE, OptCMDP, OptPessLP
```bash
cd Accord-RL/InventoryControl/DOPE

# setup the model
python model-in.py

# run different algorithms
python DOPE-in.py
python OptCMDP-in.py
python OptPessLP-in.py

# plots
python plot_all_separate.py 30000
```

2. Run the Contextual algorithm for the inventory control problem
```bash
cd Accord-RL/InventoryControl/Contextual

# setup the model
python model_contextual.py

# run the algorithm
python Contextual.py

# plots
python plot_all_separate.py 30000
```
---
20240301 Fri
1. Finalize the regret plots
```bash
# for inventory control problem
cd InventoryControl/DOPE
python plot_all_separate.py

# for BPBGClass batchsize comparison
cd BPBGClass/Contextual_shuffle
python plot_all_separate.py

# for BPBG comparison of COPS and DOPE, etc.
cd BPBGClass/DOPE
python plot_all_separate.py

```
