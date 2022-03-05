## We use R to produce P300 response curves and prediction accuracy plots for both simulation studies and real-data analysis. We also use R to compute inference-based split (merge) window ratio estimates for simulation studies. Values of prediction accuracy is presented in the format of tables although plots are generated as well for visualization purposes.



## Single-channel simulation

## Before use
Modify the $parent_path$ and $r_fun_path$.
Sample simulation datasets can be found at SMGP-Code/SIM_summary/single_channel directory.

## Use
Open the R terminal, type
Rscript sim_single_prediction.R F test TrueGen 0.5
Rscript sim_single_inference.R F test TrueGen 0.5

## Parameters
$local_use$: string of 'T' of 'F' indicating whether the code is running on local computers or the server.
$data_type$: string of 'test' by default.
$design_num$: integer of simulation study type, takes values among {1, 2, 5, 6}.
$num_rep_fit$, $num_rep_pred$: sequence replications for training and testing sets, 5, 5 by default.
$dataset_num$: number of independent replications, 100 by default.
$scenario_name$: strings among the following five scenarios for single-channel simulation study, including TrueGen; MisNoiseDist; MisLatencyLen25; MisLatencyLen35; MisSignalDist.
$zeta$: floating number between 0 and 1 indicating the cut-off threshold for the SMGP method.

## Output
The file 'sim_single_prediction.R' produces tables of prediction accuracy (mean estimates and standard errors among 100 replications) of the SMGP method and other ML methods.
The file 'sim_single_inference.R' produces ERP function estimates and prediction accuracy plots under five scenarios.



## Multi-channel simulation

## Before use
Modify the $parent_path$ and $r_fun_path$.
Sample simulation datasets can be found at SMGP-Code/SIM_summary/multi_channel directory.

## Use
Open the R terminal, type
Rscript sim_multi_prediction.R F test TrueGen 0.5 0.7
Rscript sim_multi_inference.R F test TrueGen 0.5 0.7

## Parameters
$local_use$: string of 'T' of 'F' indicating whether the code is running on local computers or the server.
$data_type$: string of 'test' by default.
$design_num$: integer of simulation study type, takes values among {11, 12}.
$num_rep_fit$, $num_rep_pred$: sequence replications for training and testing sets, 5, 5 by default.
$dataset_num$: number of independent replications, 100 by default.
$scenario_name$: strings among the following five scenarios for single-channel simulation study, including TrueGen; MisNoiseDist; MisLatencyLen25; MisLatencyLen35; MisSignalDist.
$zeta_bayes$: floating number between 0 and 1 indicating the cut-off threshold for the SMGP method.
$zeta_swlda$: floating number between 0 and 1 indicating at most (1-zeta_swlda)*100% features are selected for swLDA method.

## Output
The file 'sim_multi_prediction.R' produces the table of prediction accuracy (mean estimates and standard errors among 100 replications) of the SMGP method and other ML methods.
The file 'sim_multi_inference.R' produces ERP function estimates under TrueGen scenario.



## Simulation study SMGP prior inference

## Before use
Modify the $parent_path$ and $r_fun_path$.
Sample simulation datasets can be found at SMGP-Code/SIM_summary/multi_channel directory.

## Use
Open the R terminal, type
Rscript sim_single_smgp_prior_select.R F test TrueGen 0.5
Rscript sim_multi_smgp_prior_select.R F test TrueGen 0.5

## Parameters
$local_use$: string of 'T' of 'F' indicating whether the code is running on local computers or the server.
$data_type$: string of 'test' by default.
$design_num$: integer of simulation study type, takes values among {11, 12}.
$num_rep_fit$, $num_rep_pred$: sequence replications for training and testing sets, 5, 5 by default.
$dataset_num$: number of independent replications, 100 by default.
$scenario_name$: strings among the following five scenarios for single-channel simulation study, including TrueGen; MisNoiseDist; MisLatencyLen25; MisLatencyLen35; MisSignalDist.
$zeta_0$: floating number between 0 and 1 indicating the cut-off threshold for the SMGP method.

## Output
It produces the mean and standard error of ISWR, IMWR of the SMGP method and ESWR, EEWR of the swLDA method under specified scenario with varying true parameters.



## Real-data analysis

## Before use
Modify the $parent_path$ and $r_fun_path$.
Datasets are only available upon request due to the confidentiality agreement. Contact the authors for more information.

## Use
Open the R terminal, type
Rscript eeg_multi_inference.R F test 114 7 7 8 6 0.5
Rscript eeg_multi_sensitivity.R F test 114 7 7 8 6 0.5

## Parameters
$local_use$: string of 'T' of 'F' indicating whether the code is running on local computers or the server.
$data_type$: string of 'test' by default.
$design_num$: participant ID.
$num_rep_fit$, $num_rep_pred$: sequence replications for training and testing sets, 7, 7 by default.
$dec_factor$: decimation factor, 8 by default.
$bp_upp$: bandpass filter upper bound, 6 by default.
$zeta_0$: floating number between 0 and 1 indicating the cut-off threshold for the SMGP method.

## Output
The file 'eeg_multi_inference.R' produces tables of prediction accuracy comparing the SMGP method to other ML methods (Tables 1 & S5) and P300 response function estimates and 95% credible bands (Figures 3 & S2-S5).
The file 'eeg_multi_sensitivity.R' produces the plot and the table of the sensitivity analysis (Figure S1 and Table S4).



## Review R Code Update (2021-09-14)
The code in eeg_summary_review_sensitivity.R answers the questions from the first-round review, including the sensitivity analysis on different bandpass filters, the sensitivity analysis on different kernel hyper-parameters, ERP function estimates and split-and-merge time window (SMTW) plots for channels Cz and PO8, and updated prediction accuracy for Table 1.