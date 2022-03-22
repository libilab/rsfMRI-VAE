VAE Data Preparation README

Steps for running the VAE data preparation code:

1. Included for this demo is sample input data.  This data is stored in:
     /demo/data/rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii

2. The pretrained model is stored in: /demo/checkpoint/checkpoint99.pth.tar

3. There are some MATLAB software toolboxes required for running the code: fieldtrip and gifti.
    Run these commands in the MATLAB command window to install:
    
    fieldtrip:  addpath(genpath('/code/matlab/0libi/hgwen/Matlab/toolbox/fieldtrip/'))
    gifti: addpath(genpath('/code/matlab/0libi/khlu/CIFTI_read_save'))

4. Add the lib folder within demo (selected folders and subfolders) to the path to be able to call
    functions located here.

5. Run these scripts in the following order:
    VAE_Data_Preparation.m
    data_prep.py
    Get_data.py


6. The following mat files are saved while running VAE_Data_Preparation:

    /data/fMRI.mat
        matrix that holds normalized fMRI data for each time point
        size is (num voxels in visual cortex) x (number time points)
        includes only voxels in valid regions (gets rid of any nan values)

    /result/Transformed_Grid.mat
        six fields: L_az, L_el, R_az, R_el, Left_fMRI, Right_fMRI
        these correspond to the coordinates in terms of azimuth and elevation angles of the voxels
        in the visual cortex on the spherical template and the corresponding fMRI data values for the
        first time point, for both the left and right hemispheres

    /result/MSE_Mask.mat
        two fields: Regular_Grid_Right_Mask, Regular_Grid_Left_Mask
        these hold im_size x im_size 2D masks telling whether that voxel is valid, or has an nan
        value (meaning that data point is not useful)

    /result/Left_fMRI2Grid_192_by_192_NN.mat
        two fields: grid_mapping_L, inverse_transformation_L
    /result/Right_fMRI2Grid_192_by_192_NN.mat
        two fields: grid_mapping_R, inverse_transformation_R
        the grid mapping (for each of L/R) is size (im_size x im_size) x num voxels (without nan) 
        and will be multiplied by the voxel data for each time point to map them to the 2D grid
        the inverse transformation (for each of L/R) maps the data in the 2D grid back to the
        voxel space

