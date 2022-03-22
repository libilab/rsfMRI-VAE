% Create_Grid.m
% Transform angles and create image grid

function [T_L_az, T_L_el, T_R_az, T_R_el, X, Y] = Create_Grid(im_size, L_az, L_el, R_az, R_el)

% Inputs
% im_size: size of the output image
% Left_Mask: mask for the left cortex (if voxel valid/invalid)
% Right_Mask: mask for the right cortex (if voxel valid/invalid)
% L_az: azimuth angles at points in spherical model of L hemisphere
% L_el: elevation angles at points in spherical model of L hemisphere
% R_az: azimuth angles at points in spherical model of R hemisphere
% R_el: elevation angles at points in spherical model of R hemisphere

% Outputs
% T_L_az: transformed azimuth angles at points in spherical model of L hemi
% T_L_el: transformed elevation angles at points in spherical model of L hemi
% T_R_az: transformed azimuth angles at points in spherical model of R hemi
% T_R_el: transformed elevation angles at points in spherical model of R hemi
% X: values for x in 2D grid
% Y: values for y in 2D grid

% create vector of length im_size with values evenly spaced from -1 to 1
space = -1:2/(im_size-1):1;

% initialize empty vectors X,Y that form the 2D grid
X = [];
Y = [];

% iterate over each element in space and create the query points that will
% be passed into the griddata function
for i = space
    X = [X;ones(im_size,1)*i];
    Y = [Y;space'];
end


% calculate the sine of the elevation angle
Transformed_L_el = sin(L_el);
Transformed_R_el = sin(R_el);

% normalize the sine of left elevation angles and azimuth angles to [-1,1]
Transformed_L_el = double(2*Transformed_L_el./(max(Transformed_L_el)-min(Transformed_L_el)));
Transformed_L_az = double(2*L_az./(max(L_az)-min(L_az)));

Transformed_R_el = double(2*Transformed_R_el./(max(Transformed_R_el)-min(Transformed_R_el)));
Transformed_R_az = double(2*R_az./(max(R_az)-min(R_az)));

% set the output variables
T_L_el = Transformed_L_el;
T_L_az = Transformed_L_az;
T_R_el = Transformed_R_el;
T_R_az = Transformed_R_az;