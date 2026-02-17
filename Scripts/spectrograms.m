% 1_generate_spectrograms_FINAL.m
% REVISION: Added 'audioinfo' safety check to prevent M1 Hangs.

clc; clear; close all;

% --- PATHS ---
BASE_PATH = '/Users/ahero1/Downloads/SonicID';
DATASET_PATH = fullfile(BASE_PATH, 'UrbanSound8K');
OUTPUT_PATH = fullfile(BASE_PATH, 'data', 'spectrograms');
CSV_FILE = fullfile(DATASET_PATH, 'UrbanSound8K.csv');

% --- SETTINGS ---
TARGET_FS = 22050;        
DURATION = 4;             
NUM_SAMPLES = TARGET_FS * DURATION;

% Filter Design (One-time setup)
Nyquist = TARGET_FS / 2;
[b, a] = butter(6, [50, 10000] / Nyquist, 'bandpass');

if ~exist(OUTPUT_PATH, 'dir'); mkdir(OUTPUT_PATH); end

% --- MAIN LOOP ---
opts = detectImportOptions(CSV_FILE);
opts.SelectedVariableNames = {'slice_file_name', 'fold', 'classID', 'class'};
metadata = readtable(CSV_FILE, opts);

disp(['Processing ', num2str(height(metadata)), ' files...']);
h = waitbar(0, 'Generating Spectrograms...');

for i = 1:height(metadata)
    try
        filename = metadata.slice_file_name{i};
        fold = metadata.fold(i);
        className = metadata.class{i};
        
        % Check for hidden files
        if startsWith(filename, '._'); continue; end
        
        audioPath = fullfile(DATASET_PATH, ['fold', num2str(fold)], filename);
        
        if ~isfile(audioPath); continue; end

        % --- SAFETY CHECK (Prevents Hangs) ---
        % 'audioinfo' is fast. If this fails, we skip 'audioread' completely.
        try
            info = audioinfo(audioPath);
        catch
            continue; % Silently skip bad files
        end
        % -------------------------------------

        [y, fs] = audioread(audioPath);
        
        % Processing Pipeline
        if size(y, 2) > 1; y = mean(y, 2); end
        if fs ~= TARGET_FS; y = resample(y, TARGET_FS, fs); end
        
        if length(y) < NUM_SAMPLES
            y = [y; zeros(NUM_SAMPLES - length(y), 1)];
        else
            y = y(1:NUM_SAMPLES);
        end
        
        y = filtfilt(b, a, y);
        
        % Spectrogram
        S = melSpectrogram(y, TARGET_FS, ...
            'WindowLength', 1024, ...
            'OverlapLength', 512, ...
            'NumBands', 64, ...
            'Range', [50, 10000]);
        
        S_dB = 10 * log10(S + eps);
        S_norm = (S_dB - min(S_dB(:))) / (max(S_dB(:)) - min(S_dB(:)));
        
        % Save
        classFolder = fullfile(OUTPUT_PATH, className);
        if ~exist(classFolder, 'dir'); mkdir(classFolder); end
        
        imgName = strrep(filename, '.wav', '.png');
        savePath = fullfile(classFolder, imgName);
        imwrite(S_norm, savePath);
        
    catch ME
        % Skip
    end
    
    if mod(i, 100) == 0
        waitbar(i/height(metadata), h, sprintf('Processed %d / %d', i, height(metadata)));
    end
end
close(h);
disp('Done.');