% Geometric_Reformatting_fMRI2Grid_NN.m
% Creates the grid map for transforming fMRI data to 2D grid
% The goal is to spread the voxels out over the 2D grid

function [grid_mapping, inverse_transformation, transformed_gridmap, Loss_Rate] = Geometric_Reformatting_fMRI2Grid_NN(im_size, T_az, T_el, X,Y, Sample_Data_no_nan)

% Inputs
% im_size: size of the output image
% T_az: transformed azimuth angles at points in spherical model
% T_el: transformed elevation angles at points in spherical model
% X: values for x in 2D grid
% Y: values for y in 2D grid
% Sample_Data_no_nan: voxel data for first time point without nan values

% Outputs
% grid_mapping: matrix to map voxel data to points in 2D grid
% inverse_transformation: matrix to map points in 2D grid to voxel space
% transformed_gridmap: voxel data from first time point mapped to 2D grid
% Loss_Rate: rate of data loss after transform from voxel->2D grid and back

% initialize grid mapping and inverse transformation matrices
grid_mapping = sparse(im_size*im_size,length(Sample_Data_no_nan));
inverse_transformation = sparse(length(Sample_Data_no_nan),im_size*im_size);

% initialize vector with num voxels elements, where each value is the index
% these idx values number the voxels
Idx = 1:length(Sample_Data_no_nan);

% use the griddata function to interpolate the Idx vector at the points
% defined by the azimuth angle and the sine of the elevation angle at the
% query points in X and Y
% this creates an idx grid for where the voxels are located on the 2D grid

Grid_Idx = griddata(T_az,T_el,Idx,X,Y,'nearest');

% iterate through im_size x im_size (each entry in the 2D grid)
% in each row, put a 1 in the col corresponding to the voxel that is in
% that position in the 2D grid - this assigns the closest voxel to each
% entry in the grid
% the output will be a mapping telling which voxels correspond to which
% points on the 2D grid
% multiplying grid_mapping by a vector of voxel values will spread those
% voxel values out on the 2D grid

for i=1:size(grid_mapping,1)
    grid_mapping(i,Grid_Idx(i)) = 1;
end

% initialize the inverse grid map index vector
Inverse_Idx = 1:size(grid_mapping,1);

% use the griddata function to interpolate the Inverse_idx vector at the
% points defined by X and Y at the query points in the transformed angle
% grids
Inverse_Grid_Idx = griddata(X,Y,Inverse_Idx,T_az,T_el,'nearest');

% iterate through the number of voxels and in each row, put a 1 in the col
% corresponding to the position in the 2D im_size x im_size grid
% the output will be a mapping telling which points on the 2D grid
% correspond to which voxels

for i=1:size(inverse_transformation,1)
    inverse_transformation(i,Inverse_Grid_Idx(i)) = 1;
end

% now do calculations specifically for data from the first time point

% map the voxel data to the new 2D grid and do the inverse calculation
transformed_gridmap = grid_mapping*Sample_Data_no_nan; 
inversed_fmrimap = inverse_transformation*transformed_gridmap;

% calculate the loss rate after transformation/inverse transformation
Loss_Rate = (sum((inversed_fmrimap - Sample_Data_no_nan).^2)./sum(Sample_Data_no_nan.^2))*100;


end
