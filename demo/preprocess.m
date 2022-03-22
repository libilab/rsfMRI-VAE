%% 
% preprocess fMRI data for the VAE

%% Configuration
addpath('./lib');
addpath('./CIFTI_read_save');
cii_input_filepath = './data/rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii';
cii_output_filepath = './data/rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean_preprocessed';

%% Preprocess
% sampling frequency of HCP fMRI data
Fs = 1/0.72; 

% read in original data with fieldtrip toolbox
% loaded in as a struc
cii = ft_read_cifti(cii_input_filepath);

% extract time-series data from left and right cortex (regions 1,2)
cortex_dtseries = cii.dtseries((cii.brainstructure == 1 | cii.brainstructure == 2), :);
cortex_nonan_dtseries = cortex_dtseries(~isnan(cortex_dtseries(:,1)), :); % 59412 dimensional

% detrend and filter the data
Normalized_cortex_nonan_dtseries = Detrend_Filter(cortex_nonan_dtseries,Fs);

% fill the normalized data into the correct index of the cifti data
cortex_dtseries(~isnan(cortex_dtseries(:,1)), :) = Normalized_cortex_nonan_dtseries;
cii.dtseries((cii.brainstructure == 1 | cii.brainstructure == 2), :) = cortex_dtseries;

% save the preprocessed data
ft_write_cifti(cii_output_filepath, cii, 'parameter', 'dtseries');
