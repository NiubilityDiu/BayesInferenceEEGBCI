% cd
% change my current folder if it is not set

K_num = 114;
dec_factor = 8;
bp_upp = 6;
zeta_binary_threshold = 0.3;
serial_num = '001';
exp_name = 'BCI';
data_type = 'TRN';
file_subscript = 'down';

%%
channel_ids_ls = struct('K114', [15, 6, 7, 12, 1], ...
    'K117', [14, 16, 15, 8, 10], ...
    'K121', [15, 14, 8, 7, 6], ...
    'K146', [15, 14, 1, 5, 7], ...
    'K151', [15, 6, 1, 12, 7], ...
    'K158', [16, 3, 2, 1, 6], ...
    'K171', [16, 6, 7, 2, 15], ...
    'K172', [16, 1, 2, 12, 6], ...
    'K177', [15, 10, 7, 8, 12], ...
    'K183', [6, 2, 3, 1, 7]);
channel_ids_k = channel_ids_ls.(strcat('K', num2str(K_num)));
channel_ids = channel_ids_k(1:str2num(getenv('SLURM_ARRAY_TASK_ID')));

if bp_upp < 0
    eeg_file_subscript = 'raw';
else
    eeg_file_subscript = strcat('raw_bp_0.5_', num2str(bp_upp));
end

% folder_dir = Modify the directory here!
cd(strcat(folder_dir, '/', data_type, '_files/K', int2str(K_num)));

% some global constants:
num_letter = 19;
% remove the first odd sequence
if K_num == 154 || K_num == 190
    num_repetition = 19;
    odd_seq_len = 9;
    even_seq_len = 10;
else
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
seq_length = n_length * 1;

data_file_name = strcat('K', int2str(K_num), '_', serial_num, '_', ...
    exp_name, '_', data_type, '_eeg_dat_ML_', file_subscript, '_', ...
    int2str(dec_factor), '_from_', eeg_file_subscript);
disp(data_file_name);
eeg_dat_odd = load(strcat(data_file_name, '_odd.mat'));
label_odd = eeg_dat_odd.eeg_type';
signal_odd = eeg_dat_odd.eeg_signals;
% signal_odd = double(signal_odd);
dir1 = strcat(folder_dir, '/', data_type, '_files/K', int2str(K_num));

mkdir(dir1, '/swLDA');
% Reshape the array first, take the subset
% Here, we look at the prediction effect of each channel.
% label_odd = reshape(label_odd, [num_letter * odd_seq_len * num_rep, 1]);
[sample_size_odd, ~] = size(label_odd);


mkdir(dir1, strcat('/swLDA/channel', sprintf('_%d', channel_ids)));
signal_odd_multi = squeeze(signal_odd(:, channel_ids, :));
signal_odd_multi = permute(signal_odd_multi, [3, 2, 1]);
feature_length = n_length * length(channel_ids);
signal_odd_multi = double(reshape(signal_odd_multi, ...
    [feature_length, sample_size_odd]));
prop_num = round(zeta_binary_threshold * feature_length);

[b, se, pval, inmodel, stats] = train_SWLDAmatlab(signal_odd_multi.', label_odd, prop_num);


% save swlda wts
dir2 = strcat('/K', int2str(K_num), '_', serial_num, '_', ...
    exp_name, '_', data_type, '_swlda_wts_train_', file_subscript, ...
    '_', int2str(dec_factor), '_channel_', sprintf('%d_', channel_ids), ...
    'odd_', eeg_file_subscript, '_zeta_', num2str(1-zeta_binary_threshold), '.mat');
save(strcat(dir1, '/swLDA/channel', sprintf('_%d', channel_ids), dir2), ...
    'b', 'se', 'pval', 'inmodel', 'stats');
