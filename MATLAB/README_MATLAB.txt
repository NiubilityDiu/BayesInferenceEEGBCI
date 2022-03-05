## Real data pre-processing
Since the de-identified file is saved in MATLAB, we extract EEG signals and other useful information in MATLAB first.

## Before use
Modify the $folder_dir$ (working directory) and addpath if running on the server.
Create a new 'TRN_files' under the new parent directory.
Datasets are only available upon request due to the confidentiality agreement. Contact the author for more information.

## Use
module load matlab
matlab -nodisplay -r "addpath(genpath('parent_directory')); EEG_bandpass_filter; exit"

## Parameters
$K_num$: participant ID
$dec_factor$: decimation factor, 8 by default
$band_low$: bandpass filter lower bound, 0.5 by default
$band_upp$: bandpass filter upper bound, 6 by default

## Output
The extracted .mat file is saved under parent_directory/TRN_files/$K_num$ directory. Relevant file includes $K_num$_001_BCI_TRN_raw_bp_0.5_6.mat



## swLDA method fit
Since there is a built-in function in MATLAB for swLDA, inference function is implemented in MATLAB to save time.

## Before use
Modify the $folder_dir$ (working directory) and 'addpath' if running on the server.
Create a new 'TRN_files' under the new parent directory.

## Use
module load matlab
matlab -nodisplay -r "addpath(genpath('parent_directory')); SIM_train_swlda_matlab; exit"
matlab -nodisplay -r "addpath(genpath('parent_directory')); SIM_train_swlda_matlab_multi; exit"
matlab -nodisplay -r "addpath(genpath('parent_directory')); EEG_train_swlda_matlab_all; exit"
matlab -nodisplay -r "addpath(genpath('parent_directory')); EEG_train_swlda_matlab_multi; exit"

## Parameters
$K_num$: participant $K_num$.
$dec_factor$: decimation factor, 8 by default.
$band_low$: bandpass filter lower bound, 0.5 by default.
$band_upp$: bandpass filter upper bound, 6 by default.
$channel_ids$: 1d-array of channel ids, only needed for selected channel fit in the real data analysis.
$scenario_name$: a string of scenario name, only needed for single-channel simulation study.

SIM_train_swlda_matlab.m is for single-channel simulation study, and we need to modify $scenario_name$.
SIM_train_swlda_matlab_multi.m is for multi-channel simulation study, and only 'TrueGen' is provided.
EEG_train_swlda_matlab_multi.m is for selected-channel real data analysis, and we need to modify $channel_ids$.
EEG_train_swlda_matlab_all.m is for 16-channel real data analysis.

## Output
For simulation study, inference files are saved under the parent_directory/SIM_files/sim_x/sim_x_dataset_y/swLDA/scenario_name directory, like sim_swlda_wts_train_5_down_zeta_0.7.mat
For real data analysis, inference files are saved under the parent_directory/TRN_files/$K_num$/swLDA/channel_xxx directory, like $K_num$_001_BCI_TRN_swlda_wts_train_down_8_all_channels_odd_raw_bp_0.5_6_zeta_0.7.mat
