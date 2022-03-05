% cd
% change my current folder if it is not set

num_id = str2num(getenv('SLURM_ARRAY_TASK_ID')); 
% num_id = 1001;
design_num = floor(num_id / 100);
subset_num = rem(num_id, 100);
N_MULTIPLE_FIT = 6;
FLASH_PAUSE_LENGTH = 5;
n_length_fit = N_MULTIPLE_FIT * FLASH_PAUSE_LENGTH;
scenario_name = 'TrueGen';

num_letter = 19;
num_repetition = 5;
num_rep = 12;
% The total number of features change here:
num_electrode = 3;
n_length = FLASH_PAUSE_LENGTH * N_MULTIPLE_FIT;
seq_length = num_electrode * n_length;
zeta = 0.3;
data_type = 'SIM';
file_subscript = 'down';
% folder_dir = Modify the directory here!
sim_dir = strcat(folder_dir, '/', data_type, '_files/sim_', ...
    int2str(design_num + 1), '/sim_', ...
    int2str(design_num + 1), '_dataset_', int2str(subset_num + 1));
data_name = strcat(sim_dir, '/sim_dat_ML_', file_subscript, '_', scenario_name, '_train.mat');
disp(data_name);
eeg_dat = load(data_name);
signal = eeg_dat.eeg_signals;
signal = double(signal);
label = eeg_dat.eeg_type';


% check the dimension of label and the truncated matrix
mkdir(strcat(sim_dir, '/swLDA')); % create subfolder to save weights
mkdir(strcat(sim_dir, '/swLDA/', scenario_name));
[sample_size, ~] = size(label);
prop_num = round(zeta * seq_length);

[b, se, pval, inmodel, stats] = train_SWLDAmatlab(signal, label, prop_num);

% save swlda wts
file_name_zeta = sprintf('sim_swlda_wts_train_%d_%s_zeta_', num_repetition, file_subscript);
file_name_zeta = strcat(file_name_zeta, num2str(1-zeta), '.mat');
save(sprintf('%s/swLDA/%s/%s', sim_dir, scenario_name, file_name_zeta), ...
    'b', 'se', 'pval', 'inmodel', 'stats');
