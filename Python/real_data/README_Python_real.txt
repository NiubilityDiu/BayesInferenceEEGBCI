## Real data pre-processing

## Imported Packages:
numpy
matplotlib
scipy
sklearn
csv

## Before Use:
We put all real-data analysis python files under 'real_data' folder for consistency.
Move files under 'real_data' folder to the parent 'Python' folder before use. 
In addition, modify the directories of SMGP-Code/Python/self_py_fun/GeneralFun.py file at lines 50-51.
Create a new 'TRN_files' under the new parent directory.
Datasets are only available upon request due to the confidentiality agreement. Contact the authors for more information.

## Use
In the python terminal, type
python3 -m EEG_pre T 114 8 6
python3 -m EEG_existML_pre T 114 8 6

## Parameters
We modify and tune the following parameters in the SMGP-Code/Python/self_py_fun/GlobalEEG.py.

$local_use$: string of 'T' of 'F' indicating whether the code is running on local computers or the server.

$K_num$: participant id.

$DEC_FACTOR$: decimation factor, 8 by default.

$bp_upp$: bandpass filter upper bound value, 6 by default.
Notice that $channel_id$ may differ depending on 16-channel model fitting or selected channel model fitting.

## Output
Sequence-based datasets are saved under the parent_directory/TRN_files/$K_num$ directory. Relevant files include
$K_num$_001_BCI_TRN_eeg_dat_down_8_from_raw_bp_0.5_6.mat;
Truncated datasets are saved under the same directory. Relevant files include
$K_num$_001_BCI_TRN_eeg_mat_ML_down_8_from_raw_bp_0.5_6.mat
$K_num$_001_BCI_TRN_eeg_mat_ML_down_8_from_raw_bp_0.5_6_odd.mat
$K_num$_001_BCI_TRN_eeg_mat_ML_down_8_from_raw_bp_0.5_6_even.mat



## SMGP model fit of real data

## Imported Packages:
numpy
matplotlib
scipy
sklearn
csv

## Before Use:
We put all real-data analysis python files under 'real_data' folder for consistency.
Move files under 'real_data' folder to the parent 'Python' folder before use. 
In addition, modify the directories of SMGP-Code/Python/self_py_fun/GeneralFun.py file at lines 50-51.
Create a new 'TRN_files' under the new parent directory.

## Use
In the python terminal, type
python3 -m EEG_bayes_seq_fit_TRN_multi T 114 8 6 gamma_exp T 0 8 0.5
python3 -m EEG_bayes_seq_fit_TRN_multi T 114 8 6 gamma_exp F 0.5 8 0.5

## Parameters
$bp_upp$: bandpass filter upper bound value, 6 by default.

$kernel_option$: string options of kernel type, including 'rbf', 'rbf_+_sine', 'gamma_exp', 'gamma_exp_+_sine', 'indep_rbf'. Usually, we consider 'rbf', and 'gamma_exp'.

$cont_fit_bool$: bool variable of continuous fitting or binary threshold.

$zeta_0$: float number of cutting-off threshold between 0 and 1. When $cont_fit_bool$=True, $zeta_0$=0 by default; otherwise, it takes values in (0, 1).

$level_num$: sample space of AR(q) parameter, 8 by default.

$bp_low$: bandpass filter lower bound value, 0.5 by default.

[Optional]
s_tar, s_ntar: scale hyper-parameters of covariance kernel, positive values.
var_tar, var_ntar: variance hyper-parameters of covariance kernel, positive values.
gamma_val: gamma value of gamma-exponential kernel, 0 < gamma_val < 2.
alpha_s, beta_s: shape and rate hyper-parameters of inverse-gamma distribution for sigma_x_sq.

The default optional values are
s_tar = s_ntar = 0.5
var_tar = 1.0, var_ntar = 1.0
gamma_val = 1.8
alpha_s = beta_s = 0.1

## Output
Results are saved under the parent_directory/TRN_files/$K_num$/BayesGenq2/channel_xxx/ directory. Relevant files include
$K_num$_001_BCI_TRN_super_seq_BayesGenq2_mcmc_trn_1_continuous_down_8_raw_bp_0.5_6.mat
$K_num$_001_BCI_TRN_super_seq_BayesGenq2_convol_seq_trn_1_continuous_down_8_raw_bp_0.5_6.pdf
$K_num$_001_BCI_TRN_super_seq_BayesGenq2_mcmc_trn_1_binary_down_8_raw_bp_0.5_6_zeta_$zeta_0$.mat
$K_num$_001_BCI_TRN_super_seq_BayesGenq2_convol_seq_trn_1_binary_down_8_raw_bp_0.5_6_zeta_$zeta_0$.pdf

## New Note on Sep 5th, 2021.
If values of $s_tar$, $s_ntar$, $gamma_val$ are assigned by "sys.argv[xxx]", then it will create additional folder "scale=$s_tar$, gamma=$gamma_val$" to distinguish the kernel hyper-parameters. This is to satisfy the requirement of sensitivity analysis.



## SMGP method prediction of real data

## Imported Packages:
numpy
matplotlib
scipy
sklearn
csv

## Before Use:
We put all real-data analysis python files under 'real_data' folder for consistency.
Move files under 'real_data' folder to the parent 'Python' folder before use. 
In addition, modify the directories of SMGP-Code/Python/self_py_fun/GeneralFun.py file at lines 50-51.
Create a new 'BayesGenq2Pred' under the new parent directory.

## Use
In the python terminal, type
python3 -m EEG_bayes_seq_pred_TRN_multi T 114 8 6 0.5 8 0.5

## Parameters
In addition to parameters in the SMGP-Code/Python/self_py_fun/GlobalEEG.py file, we tune the following parameters in the SMGP-Code/Python/real_data/EEG_bayes_seq_pred_TRN_multi.py file.

$zeta_0$: float number of cutting-off threshold between 0 and 1.

$level_num$: sample space of AR(q) parameter, 8 by default.

## Output
Results are saved under the parent_directory/BayesGenq2Pred/$K_num$/channel_xxx/ directory. We only look at prediction accuracy on testing files (even sequence replications) including
$K_num$_001_BCI_TRN_single_seq_test_train_7_pred_test_7_binary_down_8_raw_bp_0.5_6_zeta_$zeta_0$.csv
$K_num$_001_BCI_TRN_single_seq_test_train_7_pred_test_7_binary_down_8_raw_bp_0.5_6_zeta_$zeta_0$.npz (which saves more details)

## New Note on Sep 5th, 2021.
If values of $s_tar$, $s_ntar$, $gamma_val$ are assigned by "sys.argv[xxx]", then it will create additional folder "scale=$s_tar$, gamma=$gamma_val$" to distinguish the kernel hyper-parameters. This is to satisfy the requirement of sensitivity analysis.



## Other ML methods fit and prediction of simulation studies

## Imported Packages:
numpy
matplotlib
scipy
sklearn
csv

## Before Use
We put all real data analysis python files under 'real_data' folder for consistency.
Move files under 'real_data' folder to the parent 'Python' folder before use. 
In addition, modify the directories of SMGP-Code/Python/self_py_fun/GeneralFun.py file at lines 50-51.
Create new folder names for other ML methods under the new parent directory.

## Use
In the python terminal, type
python3 -m EEG_existML T 114 8 6 LR

## Parameters
In addition to parameters in the SMGP-Code/Python/self_py_fun/GlobalEEG.py file, we tune the following parameters in the SMGP-Code/Python/real_data/EEG_existML_pre.py and EEG_existML.py file.

$method_name$: names of existing ML methods, including
'LR' for logistic regression;
'SVC' for support vector machine;
'RF' for random forest;
'BAG' for bagging;
'ADA' for Adaptive Boosting.
'XGBoost' for XGBoost.

## Output
Prediction results are saved under the parent_directory/ML_name_folder/$K_num$/channel_xxx directory. Relevant files include
$K_num$_001_BCI_TRN_train_7_pred_test_7_ML_down_8_raw_bp_0.5_6.csv
$K_num$_001_BCI_TRN_train_7_pred_test_7_ML_down_8_raw_bp_0.5_6.csv



# swLDA method prediction of real data analysis

## Imported Packages:
numpy
matplotlib
scipy
sklearn
csv

## Before Use
We put all real data analysis python files under 'real_data' folder for consistency.
Move files under 'simulation' folder to the parent 'Python' folder before use. 
Create a new folder name 'EEGswLDA' under the new parent directory.

## Use
In the python terminal, type
python3 -m EEG_swlda_pred_multi T 114 8 6 0.7

## Parameters
In addition to parameters in the SMGP-Code/Python/self_py_fun/GlobalEEG.py file, we tune the following parameters in the SMGP-Code/Python/real_data/EEG_swlda_pred_multi.py file.

$zeta_0$: at most (1-zeta_0)*100% of features gets selected in swLDA method, 0.7 by default.

## Output
Results are saved under the parent_directory/EEGswLDA/$K_num$/channel_xxx/ directory. Relevant files include
$K_num$_001_BCI_TRN_train_7_pred_test_7_ML_down_8_from_raw_bp_0.5_6_zeta_0.7.csv
$K_num$_001_BCI_TRN_train_7_pred_test_7_ML_down_8_from_raw_bp_0.5_6_zeta_0.7.csv



# Perform channel selection based on our information criterion (added on Sep 5th, 2021)

## Imported Packages:
numpy
matplotlib
scipy
sklearn
csv

## Before Use
We put all real data analysis python files under 'real_data' folder for consistency.
Move files under 'simulation' folder to the parent 'Python' folder before use. 

## Use
In the python terminal, type
python3 -m EEG_bayes_channel_ranking T 114 8 6 0.5 8 0.5
python3 -m EEG_bayes_channel_ranking T 114 8 6 0.5 8 0.5 0.5 1.8

## Parameters
In addition to parameters in the SMGP-Code/Python/self_py_fun/GlobalEEG.py file, we tune the following parameters in the SMGP-Code/Python/real_data/EEG_bayes_channel_ranking.py file.

$bp_upp$: floating number of the bandpass filter upper bound, taking values in {5.5, 6, 6.5}.

$zeta_0$: float number of cutting-off threshold between 0 and 1.

$level_num$: integer number of the sample space of AR(q) parameter, 8 by default.

$bp_low$: floating number of the bandpass filter lower bound, taking values in {0.4, 0.5, 0.6}.

$scale_val$: floating number of the scale hyper-parameter in the gamma-exponential kernel, taking values in {0.4, 0.5, 0.6}. Theoretical range is (0, +infinity).

$gamma_val$: floating number of the gamma hyper-parameter in the gamma-exponential kernel, taking values in {1.7, 1.8, 1.9}. Theoretical range is (0, 2).

## Output
Results are saved under the parent directory as a csv format. Each combination of parameter input creates a numerical channel ranking, and the result appends to the existing csv file. Relevant files include
parent_directory/channel_selection.csv
parent_directory/channel_selection_2.csv

Two files have been renamed to "Bandpass Filters.xlsx" and "Kernel Hyper-parameters.xlsx", and they are saved under the directory of SMGP-Code/EEG_summary/Channel Ranking/.

## Note 
If we fix $scale_val$=0.5 and $gamma_val$=1.8 and change $bp_low$, $bp_upp$, it performs the channel ranking and selection based on the bandpass filter; if we fix $bp_low$=0.5 and $bp_upp$=6 and change $scale_val$, $gamma_val$, it performs the channel ranking and selection based on the kernel hyper-parameters. For each scenario, there are 10 * 9=90 rows in the csv file. Each row contains 19 columns. The first three columns are parameters to compare and subject name, while the remaining 16 columns are the channel ranking results. We label F3,  Fz,  F4,  T7,  C3,  Cz,  C4,  T8,  CP3,  CP4,  P3,  Pz,  P4, PO7, PO8, and Oz as 1, ..., 16. 
