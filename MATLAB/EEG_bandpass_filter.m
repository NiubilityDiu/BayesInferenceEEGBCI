%cd
% change my current folder if it is not set

K_num = 114;
disp(K_num);
num_electrode = 16;
dec_factor = 8;
serial_num = '001';
exp_name = 'BCI';
data_type = 'TRN';
% folder_dir = Modify the directory here!
cd(strcat(folder_dir, '/', data_type, '_files/K', int2str(K_num)));
trn_file_name = strcat('K', int2str(K_num), '_', serial_num, '_', ...
    exp_name, '_', data_type);
disp(trn_file_name);
load(strcat(trn_file_name, '.mat'))
dir1 = strcat(folder_dir, data_type, '_files/K', int2str(K_num));


% Raw signals
RawData = Data.RawData;
signal = RawData.signal;
states = RawData.states;
parameters = RawData.parameters;
N_sample = RawData.total_samples;

% State variables
Stim_Code = states.StimulusCode;
Stim_Type = states.StimulusType;
Stim_Begin = states.StimulusBegin;
Stim_PIS = states.PhaseInSequence;

fs = 256;
mkdir(dir1, '/psd_plots')

for i = 1:num_electrode
    disp(strcat('channel_', int2str(i)));
    signal_channel_i = signal(:, i);
    h = figure;
    pspectrum(signal_channel_i, fs);
    saveas(h, sprintf('psd_plots/K%d_channel_%d_power_spectral_density.png', K_num, i));
end

% Bandpass filter for the raw signal
band_low = 0.5;
band_upp = 6;
 
signal_bp = bandpass(signal, [band_low, band_upp], fs);

% optional: spatial filter common average reference (CAR)
signal_bp_mean = mean(signal_bp, 2);
signal_bp = signal_bp - signal_bp_mean;
% disp(size(signal_bp_mean))
% optional: spatial filter Laplacian filter (partial channel)


signal_bp_down = downsample(signal_bp, dec_factor);
Stim_Begin_down = downsample(Stim_Begin, dec_factor);
Stim_Code_down = downsample(Stim_Code, dec_factor);
Stim_PIS_down = downsample(Stim_PIS, dec_factor);
Stim_Type_down = downsample(Stim_Type, dec_factor);

raw_name = strcat('/', trn_file_name, '_raw.mat');
raw_bp_name = strcat('/', trn_file_name, '_raw_bp_', num2str(band_low), '_', num2str(band_upp), '.mat');
down_bp_name = strcat('/', trn_file_name, '_down_', num2str(dec_factor), '_bp_', num2str(band_low), '_', num2str(band_upp), '.mat');

%%
if strcmp(data_type, 'FRT')
    % Weights from LS method
    ls_weights = parameters.Classifier.NumericValue;
    % True predict letter
    frt_letters = GUIspecific.goalText;
    save(strcat(dir1, raw_name), ...
    'signal', ...
    'Stim_Begin', 'Stim_Code', 'Stim_PIS', 'Stim_Type', ...
    'ls_weights', 'frt_letters')

    save(strcat(dir1, raw_bp_name), ...
    'signal_bp', ...
    'Stim_Begin', 'Stim_Code', 'Stim_PIS', 'Stim_Type', ...
    'ls_weights', 'frt_letters')

%     save(strcat(dir1, down_bp_name), ...
%     'signal_bp_down', ...
%     'Stim_Begin_down', 'Stim_Code_down', 'Stim_PIS_down', 'Stim_Type_down', ...
%     'ls_weights', 'frt_letters')

else
   save(strcat(dir1, raw_name), ...
    'signal', ...
    'Stim_Begin', 'Stim_Code', 'Stim_PIS', 'Stim_Type')

   save(strcat(dir1, raw_bp_name), ...
    'signal_bp', ...
    'Stim_Begin', 'Stim_Code', 'Stim_PIS', 'Stim_Type')

%     save(strcat(dir1, down_bp_name), ...
%     'signal_bp_down', ...
%     'Stim_Begin_down', 'Stim_Code_down', 'Stim_PIS_down', 'Stim_Type_down')
end
