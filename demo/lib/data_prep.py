import matplotlib.pyplot as plt
import argparse
import numpy as np
import scipy.io as sio
import h5py
import os

parser = argparse.ArgumentParser(description='VAE for fMRI generation')
parser.add_argument('--time-points', type=int, default=1200, help='number of time points')
parser.add_argument('--img-size', type=int, default=192, help='size of geometric-reformatted image')
parser.add_argument('--fmri-path', default='./data', type=str, help='path of the input fMRI data')
parser.add_argument('--trans-path', default='./result', type=str, help='path of the geometric reformatting transformation')
parser.add_argument('--output-path', default='./data', type=str, help='path of the output data for VAE inference')
print('here above')

# Transform the data
def GenerateData(args, left_trans_mat, right_trans_mat):
    LeftSurfData = np.zeros([args.time_points,1,args.img_size,args.img_size])
    RightSurfData = np.zeros([args.time_points,1,args.img_size,args.img_size])

    fmri_file = os.path.join(args.fmri_path , 'fMRI.mat')
    fmri_data = sio.loadmat(fmri_file)['Normalized_fMRI']
    left_data = fmri_data[0:29696,:]
    right_data = fmri_data[29696:59412,:]
    print(f'Loading data the size of the left hemisphere is {left_data.shape}; the size of the right hemisphere is {right_data.shape}')
    # left
    LeftSurfData = np.expand_dims(left_trans_mat.dot(left_data).T.reshape((-1,args.img_size,  args.img_size)), axis=1)
    print(LeftSurfData.shape)
    # right
    RightSurfData = np.expand_dims(right_trans_mat.dot(right_data).T.reshape((-1, args.img_size, args.img_size)), axis=1)
    print(RightSurfData.shape)   
    print('here in generate data')

    return LeftSurfData, RightSurfData

# Save the training data as hdf5 file
def SaveData(LeftSurfData, RightSurfData, LeftMask, RightMask, file_path):
    print(LeftSurfData.shape)
    if os.path.isfile(file_path):
        print('Output Data Exists. Will delete it and generate a new one')
        os.system('rm -r ' + file_path) 
    H5File = h5py.File(file_path, 'w')  
    H5File['LeftData'] = LeftSurfData.astype('float32')
    H5File['RightData'] = RightSurfData.astype('float32') 
    H5File['LeftMask'] = LeftMask.astype('float32')
    H5File['RightMask'] = RightMask.astype('float32')
    H5File.close() 
    print('here in save_data')                                                

if __name__ == "__main__":
    args = parser.parse_args()
    # check data availability of the target folder
    if os.path.isdir(args.output_path):
        print('Target directory exists: ' + args.output_path)
    else:
        os.system('mkdir ' + args.output_path)
        print('Target directory does not exist and is created: ' + args.output_path)
    
    # Loading transformation data
    left_trans_mat = sio.loadmat(os.path.join(args.trans_path,'Left_fMRI2Grid_192_by_192_NN.mat'))['grid_mapping']
    print(f'The shape of the loaded left-transoformation file is: {left_trans_mat.shape}')
    right_trans_mat = sio.loadmat(os.path.join(args.trans_path,'Right_fMRI2Grid_192_by_192_NN.mat'))['grid_mapping']
    print(f'The shape of the loaded right-transoformation file is: {right_trans_mat.shape}')
    
    # Loading Brain Mask
    LeftMask = sio.loadmat(os.path.join(args.trans_path,'MSE_Mask.mat'))['Regular_Grid_Left_Mask']
    RightMask = sio.loadmat(os.path.join(args.trans_path,'MSE_Mask.mat'))['Regular_Grid_Right_Mask']
    print('here in main')
    # Generate the Left and Right Data
    file_path = os.path.join(args.output_path,'demo_data.h5')
    LeftSurfData, RightSurfData = GenerateData(args, left_trans_mat, right_trans_mat)
    SaveData(LeftSurfData, RightSurfData, LeftMask, RightMask, file_path)
	                                     

