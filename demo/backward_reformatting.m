%% 
% backward projection from the VAE reconstruction to the cortex

%% Configuration
batchsize = 120;
addpath('./CIFTI_read_save');
recon_path = './result/recon/';
inverse_transformation_path = './result/';
cii_template_filepath = './data/rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii';
cii_output_filepath = './data/rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean_reconstruction';

%% load the inverse transformation matrix

% load the inverse transformation matrix of geometric reformatting, as in JH's Recon_fMRI_Generator.m
load([inverse_transformation_path 'Left_fMRI2Grid_192_by_192_NN.mat'], 'inverse_transformation');
Left_inverse_transformation = inverse_transformation;
load([inverse_transformation_path 'Right_fMRI2Grid_192_by_192_NN.mat'], 'inverse_transformation');
Right_inverse_transformation = inverse_transformation;

%% backward projection
recon_dtseries = zeros(59412, 1200);
for idx = 1:1:10
    load(['./result/recon/img' num2str(idx-1) '.mat'], 'recon_L', 'recon_R');
    corticalrecon_L = Left_inverse_transformation * double(reshape(permute(recon_L,[1,2,4,3]),batchsize, [])');
    corticalrecon_R = Right_inverse_transformation * double(reshape(permute(recon_R,[1,2,4,3]),batchsize, [])');
    recon_dtseries(:, (idx-1)*batchsize+1:idx*batchsize) = [corticalrecon_L; corticalrecon_R];
end

%% save the reconstruction back into cifti file
% read in original data with fieldtrip toolbox
% loaded in as a struc
cii = ft_read_cifti(cii_template_filepath);

% extract time-series data from left and right cortex (regions 1,2)
cortex_dtseries = cii.dtseries((cii.brainstructure == 1 | cii.brainstructure == 2), :);

% fill the normalized data into the correct index of the cifti data
cortex_dtseries(~isnan(cortex_dtseries(:,1)), :) = recon_dtseries;
cii.dtseries((cii.brainstructure == 1 | cii.brainstructure == 2), :) = cortex_dtseries;

% save the preprocessed data
ft_write_cifti(cii_output_filepath, cii, 'parameter', 'dtseries');
