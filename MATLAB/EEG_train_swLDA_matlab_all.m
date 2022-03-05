%cd
% change my current folder if it is not set

K_num = 114;
dec_factor = 8;
bp_upp = 6;
zeta_binary_threshold = 0.3;
serial_num = '001';
exp_name = 'BCI';
data_type = 'TRN';
file_subscript = 'down';

if bp_upp < 0
    eeg_file_subscript = 'raw';
else
    eeg_file_subscript = strcat('raw_bp_0.5_', num2str(bp_upp));
end

% folder_dir = Modify the directory here!;
cd(strcat(folder_dir, '/', data_type, '_files/K', int2str(K_num)));

% some global constants:
num_letter = 19;
if K_num == 154 || K_num == 190
    num_repetition = 19; % remove the first odd sequence
    odd_seq_len = 9;
    even_seq_len = 10;
else
    % remove the first odd sequence for training
    num_repetition = 14;
    odd_seq_len = 7;  
    even_seq_len = 7;
end
num_rep = 12;
if dec_factor == 4
    n_length = 50;
else
    n_length = 25;
end
num_electrode = 16;
seq_length = n_length * num_electrode;
prop_num = round(0.3 * seq_length);

data_file_name = strcat('K', int2str(K_num), '_', serial_num, '_', ...
    exp_name, '_', data_type, '_eeg_dat_ML_', file_subscript, '_', ...
    int2str(dec_factor), '_from_', eeg_file_subscript);
eeg_dat = load(strcat(data_file_name, '_odd.mat'));
label = eeg_dat.eeg_type';
signal = eeg_dat.eeg_signals;
signal = double(signal);
dir1 = strcat(folder_dir, '/', data_type, '_files/K', int2str(K_num));


mkdir(dir1, '/swLDA')
% Permute the array, reshape the array
signal = permute(signal, [3, 2, 1]);
signal = reshape(signal, [seq_length, num_letter * odd_seq_len * num_rep]);
% label = reshape(label, [num_letter, num_repetition, num_rep]);
mkdir(dir1, strcat('/swLDA/all_channels'));

[b, se, pval, inmodel, stats] = train_SWLDAmatlab(signal.', label, ...
    prop_num);

% save swlda wts
dir2 = strcat('/K', int2str(K_num), '_', serial_num, '_', ...
    exp_name, '_', data_type, '_swlda_wts_train_', file_subscript, ...
    '_', int2str(dec_factor), '_all_channels_', ...
    'odd_', eeg_file_subscript, '_zeta_', num2str(1-zeta_binary_threshold), '.mat');
save(strcat(dir1, '/swLDA/all_channels', dir2), ...
    'b', 'se', 'pval', 'inmodel', 'stats');

