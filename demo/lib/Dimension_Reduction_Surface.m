% Dimension_Reduction_Surface.m
% Record azimuth and elevation angle for each voxel on the surface

% Inputs
% voxel_valid: logical vector denoting which voxels to use in calculations
% lb: left hemisphere spherical template
% rb: right hemisphere spherical template

% Outputs
% L_az: azimuth angle coordinates of voxels in left cortex
% L_el: elevation angle coordinates of voxels in left cortex
% R_az: azimuth angle coordinates of voxels in right cortex
% R_el: elevation angle coordinates of voxels in right cortex

function [L_az, L_el, R_az, R_el] = Dimension_Reduction_Surface(voxel_valid,lb,rb) 

% separate brain_idx into left/right visual cortex
Left_Idx = voxel_valid(1:length(voxel_valid)/2);
Right_Idx = voxel_valid((length(voxel_valid)/2 + 1):end);

% vertices field of lb/rb has (x,y,z) coordinates of the spherical template
% each column of vertices corresponds to one of (x,y,z)
% column length is number of voxels - same length as Left_Idx/Right_Idx

% estimate azimuth/elevation angles given spherical model of L hemisphere
[azimuth,elevation,~] = cart2sph(lb.vertices(:,1),lb.vertices(:,2),lb.vertices(:,3));

% extract data corresponding to the Left_Idx values
Left_azimuth = azimuth(Left_Idx);
Left_elevation = elevation(Left_Idx);

% estimate azimuth/elevation angles given spherical model of R hemisphere
% to mirror images from hemispheres, flip the x-direction of R hemisphere
rb.vertices(:,1) = -rb.vertices(:,1); 
[azimuth,elevation,~] = cart2sph(rb.vertices(:,1),rb.vertices(:,2),rb.vertices(:,3));

% extract data corresponding to the Right_Idx values
Right_azimuth = azimuth(Right_Idx);
Right_elevation = elevation(Right_Idx);

% saved to function output names
L_az = Left_azimuth;
L_el = Left_elevation;
R_az = Right_azimuth;
R_el = Right_elevation;

end













