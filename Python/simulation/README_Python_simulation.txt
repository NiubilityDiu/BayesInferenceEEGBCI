## Data generation of simulation studies

## Imported Packages:
numpy
matplotlib
scipy
sklearn
csv

## Before Use:
We put all simulation-related python files under SMGP-Code/Python/simulation folder for consistency.
Move files under 'simulation' folder to the parent 'Python' folder before use. 
In addition, modify the directories of SMGP-Code/Python/self_py_fun/GeneralFun.py file at lines 50-51.
Create a new 'SIM_files' under the new parent directory
For simulation study, we generate the datasets for either single channel or multi-channel.

## Use
For single-channel, use 'SIM_generate_single.py' file.
For multi-channel, use 'SIM_generate_multi.py' file.
For example, 
In the python terminal, type
python3 -m SIM_generate_single T latency 5 1 1
python3 -m SIM_generate_multi T latency 5 1 multi_channel

## Parameters
We modify and tune the following parameters in the SMGP-Code/Python/self_py_fun/GlobalSIM.py file.
$local_use$: string of 'T' of 'F' indicating whether the code is running on local computers or the server.

$sim_type$: string options among 'latency', 'super_seq', and 'ML'. For data generation, use 'latency'; for SMGP method fit, use 'super_seq'; for other ML methods, use 'ML'.

$repet_num_fit$: integer of the sequence replication for each dataset.

$scenario_name$: integer among the following five scenarios for single-channel simulation study, including 1: TrueGen; 2: MisNoiseDist; 3: MisLatencyLen25; 4: MisLatencyLen35; 5: MisSignalDist.

$mean_fn_type$: integer or string option, including 1,2 (for single-channel simulation study); and 'multi_channel', 'multi_channel_2' for multi-channel simulation study).

$NUM_ELECTRODE$: integer of channel dimension. 1 for single-channel setting, and other positive integers for multi-channel setting.

[Optional]
s_x_sq: The variance parameter of observational-level noise.
rho: The auto-correlation parameter for temporal association.
t_df: The degree of freedom of the Student-t observational-level noise.
LETTERS: A list of true target characters.
NUM_REPETITION: The total number of sequence replications.
REPETITION_TRN, REPETITION_TEST: The total number of sequence replications for training and testing. Their sum should be NUM_REPETITION, and the values match $repet_num_fit$ during data generation process. In our case, we let REPETITION_TRN = REPETITION_TEST = 5.

The default optional values are
s_x_sq = 10, 20 for single-channel simulation study
s_x_sq = 20, 40 for multi-channel simulation study
rho = (0.5, 0)
t_df = 5
LETTERS = 'THE_QUICK_BROWN_FOX' 

## Output
Results are saved under the parent_directory/SIM_files/sim_x/sim_x_dataset_y directory.
For single-channel simulation study, relevant files include 
sim_dat_latency_TrueGen.mat,
sim_dat_super_seq_TrueGen_train.mat, sim_dat_super_seq_TrueGen_test.mat
sim_dat_ML_down_TrueGen_train.mat, sim_dat_ML_down_TrueGen_test.mat
We also obtain files under other scenarios if we replace 'TrueGen' with 'MisNoiseDist', 'MisLatencyLen25', 'MisLatencyLen35', and 'MisSignalDist'. 

For multi-channel simulation study, we have
sim_dat_latency_TrueGen.mat,
sim_dat_super_seq_TrueGen_train.mat, sim_dat_super_seq_TrueGen_test.mat
sim_dat_ML_down_TrueGen_train.mat, sim_dat_ML_down_TrueGen_test.mat



## SMGP model fit of simulation studies

## Imported Packages:
numpy
matplotlib
scipy
sklearn
csv

## Before Use:
We put all simulation-related python files under 'simulation' folder for consistency.
Move files under 'simulation' folder to the parent 'Python' folder before use. 
In addition, modify the directories of SMGP-Code/Python/self_py_fun/GeneralFun.py file at lines 50-51.
Create a new 'SIM_files' under the new parent directory

## Use
For single-channel, use 'SIM_bayes_super_seq_fit.py' file.
For multi-channel, use 'SIM_bayes_super_seq_fit_multi.py' file.
For example, 
In the python terminal, type
python3 -m SIM_bayes_super_seq_fit T super_seq 5 1 1 gamma_exp T 0 8
python3 -m SIM_bayes_super_seq_fit_multi T super_seq 5 1 multi_channel gamma_exp T 0 8
python3 -m SIM_bayes_super_seq_fit T super_seq 5 1 1 gamma_exp F 0.5 8
python3 -m SIM_bayes_super_seq_fit_multi T super_seq 5 1 multi_channel gamma_exp F 0.5 8

## Parameters
In addition to parameters in the SMGP-Code/Python/self_py_fun/GlobalSIM.py file as mentioned above, we tune the following parameters in the SMGP-Code/Python/simulation/SIM_bayes_super_seq_fit(_multi).py file.

$mean_fn_type$: integer or string option, including 1,2 (for single-channel simulation study); and 'multi_channel', 'multi_channel_2' for multi-channel simulation study).

$kernel_option$: string options of kernel type, including 'rbf', 'rbf_+_sine', 'gamma_exp', 'gamma_exp_+_sine', 'indep_rbf'. Usually, we consider 'rbf', and 'gamma_exp'.

$cont_fit_bool$: bool variable of continuous fitting or binary threshold.

$zeta_0$: float number of cutting-off threshold between 0 and 1. When $cont_fit_bool$=True, $zeta_0$=0 by default; otherwise, it takes values in (0, 1).

$level_num$: sample space of AR(q) parameter, 8 by default.

[Optional]
s_tar, s_ntar: scale hyper-parameters of covariance kernel, positive values.
var_tar, var_ntar: variance hyper-parameters of covariance kernel, positive values.
gamma_val: gamma value of gamma-exponential kernel, 0 < gamma_val < 2.
alpha_s, beta_s: shape and rate hyper-parameters of inverse-gamma distribution for sigma_x_sq.

The default optional values are
s_tar = s_ntar = 0.5
var_tar = 5.0, var_ntar = 0.5
gamma_val = 1.8
alpha_s = beta_s = 0.1

## Output
Results are saved under the parent_directory/SIM_files/sim_x/sim_x_dataset_y/BayesGenq2/scenario_name directory. Relevant files include
sim_x_dataset_y_super_seq_BayesGenq2_mcmc_trn_5_continuous_down_1_fit.mat
sim_x_dataset_y_super_seq_BayesGenq2_convol_seq_trn_5_continuous_down_1_fit.pdf
sim_x_dataset_y_super_seq_BayesGenq2_mcmc_trn_5_binary_down_1_fit_zeta_$zeta_0$.mat
sim_x_dataset_y_super_seq_BayesGenq2_mcmc_trn_5_binary_down_1_fit_zeta_$zeta_0$.pdf



## SMGP method prediction of simulation studies

## Imported Packages:
numpy
matplotlib
scipy
sklearn
csv

## Before Use:
We put all simulation-related python files under 'simulation' folder for consistency.
Move files under 'simulation' folder to the parent 'Python' folder before use. 
In addition, modify the directories of SMGP-Code/Python/self_py_fun/GeneralFun.py file at lines 50-51.
Create a new 'BayesGenq2Pred' under the new parent directory.

## Use
For single-channel, use 'SIM_bayes_super_seq_pred.py' file.
For multi-channel, use 'SIM_bayes_super_seq_pred_multi.py' file.
For example, 
In the python terminal, type
python3 -m SIM_bayes_super_seq_pred T super_seq 5 1 0.5 8
python3 -m SIM_bayes_super_seq_pred_multi T super_seq 5 1 multi_channel 0.5 8

## Parameters
In addition to parameters in the SMGP-Code/Python/self_py_fun/GlobalSIM.py file, we tune the following parameters in the SMGP-Code/Python/simulation/SIM_bayes_super_seq_pred(_multi).py file.

$zeta_0$: float number of cutting-off threshold between 0 and 1.

$level_num$: sample space of AR(q) parameter, 8 by default.

## Output
Results are saved under the parent_directory/BayesGenq2Pred/sim_x/sim_x_dataset_y/scenario_name directory. We only look at prediction accuracy on testing files including
sim_x_dataset_y_single_seq_test_train_5_pred_test_5_binary_down_1_fit_zeta_$zeta_0$.csv
sim_x_dataset_y_single_seq_test_train_5_pred_test_5_binary_down_1_fit_zeta_$zeta_0$.npz (which saves more details other than .csv)



## Other ML methods fit and prediction of simulation studies

## Imported Packages:
numpy
matplotlib
scipy
sklearn
csv

## Before Use
We put all simulation-related python files under 'simulation' folder for consistency.
Move files under 'simulation' folder to the parent 'Python' folder before use. 
In addition, modify the directories of SMGP-Code/Python/self_py_fun/GeneralFun.py file at lines 50-51.
Create new folder names for other ML methods under the new parent directory.

## Use
In the python terminal, type
python3 -m SIM_existML T ML 5 1 LR

## Parameters
In addition to parameters in the SMGP-Code/Python/self_py_fun/GlobalSIM.py file, we tune the following parameters in the SMGP-Code/Python/simulation/SIM_existML.py file.

$method_name$: names of existing ML methods, including
'LR' for logistic regression;
'SVC' for support vector machine;
'RF' for random forest;
'BAG' for bagging;
'ADA' for Adaptive Boosting.

## Output
Results are saved under the parent_directory/ML_name_folder/sim_x/sim_x_dataset_y/scenario_name directory. Relevant files include
sim_x_dataset_y_train_5_pred_test_5_down.csv
sim_x_dataset_y_test_5_pred_test_5_down.csv



## swLDA method prediction of simulation studies

## Imported Packages:
numpy
matplotlib
scipy
sklearn
csv

## Before Use
We put all simulation-related python files under 'simulation' folder for consistency.
Move files under 'simulation' folder to the parent 'Python' folder before use. 
Create a new folder name 'EEGswLDA' under the new parent directory.

## Use
In the python terminal, type
python3 -m SIM_swlda_pred T ML 5 1
python3 -m SIM_swlda_pred_multi T ML 5 1

## Parameters
$local_use$: string of 'T' of 'F' indicating whether the code is running on local computers or the server.

$sim_type$: string options among 'latency', 'super_seq', and 'ML'. For data generation, use 'latency'; for SMGP method fit, use 'super_seq'; for other ML methods, use 'ML'.

$repet_num_fit$: integer of the sequence replication for each dataset.

$scenario_name$: integer among the following five scenarios for single-channel simulation study, including 1: TrueGen; 2: MisNoiseDist; 3: MisLatencyLen25; 4: MisLatencyLen35; 5: MisSignalDist.

[Optional]
zeta_0: at most (1-zeta_0)*100% of features gets selected in swLDA method, 0.7 by default.
design_num: number of cases of simulation studies. It takes values among {0, 1, 4, 5, 10, 11}. 
0,1,4,5 are for the single-channel setting, while 10-11 are for the multi-channel setting.
subset_total_num or subset_num: integer of independent replications of each case, 100 by default.

## Output
Results are saved under the parent_directory/EEGswLDA/sim_x/sim_x_dataset_y/scenario_name directory. Relevant files include
sim_x_dataset_y_train_5_pred_test_5_down_zeta_0.7.csv
sim_x_dataset_y_test_5_pred_test_5_down_zeta_0.7.csv
