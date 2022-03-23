% Mask_Generation.m
% Create mask of 2D image grid

% Inputs
% im_size: size of the output image
% Left_Mask: mask for the left cortex (if voxel valid/invalid)
% Right_Mask: mask for the right cortex (if voxel valid/invalid)
% T_L_az: transformed azimuth angles at points in spherical model of L hemi
% T_L_el: transformed elevation angles at points in spherical model of L hemi
% T_R_az: transformed azimuth angles at points in spherical model of R hemi
% T_R_el: transformed elevation angles at points in spherical model of R hemi
% X: values for x in 2D grid
% Y: values for y in 2D grid

% Outputs
% Regular_Grid_Left_Mask: im_size x im_size 2D mask for L hemisphere
% Regular_Grid_Right_Mask: im_size x im_size 2D mask for R hemisphere

function [Regular_Grid_Left_Mask, Regular_Grid_Right_Mask] = Mask_Generation(im_size, Left_Mask, Right_Mask, T_L_az, T_L_el, T_R_az, T_R_el, X, Y)

% use the griddata function to interpolate the mask at the points
% defined by the azimuth angle and the sine of the elevation angle at the
% query points in X and Y
% the output will be the mask value telling if that point in the new grid
% set by X and Y is valid or has an nan data value
% then reshape the interpolated mask values into shape im_size x im_size

Grid_Idx = griddata(T_L_az,T_L_el,Left_Mask,X,Y,'nearest');
Regular_Grid_Left_Mask = reshape(Grid_Idx,im_size,im_size)';

Grid_Idx = griddata(T_R_az,T_R_el,Right_Mask,X,Y,'nearest');
Regular_Grid_Right_Mask = reshape(Grid_Idx,im_size,im_size)';

end



