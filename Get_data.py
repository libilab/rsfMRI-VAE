import torch
import scipy.io as io
import numpy as np
import argparse
from utils import *
from fMRIVAE_Model import *
import torch.utils.data
import os



parser = argparse.ArgumentParser(description='VAE for fMRI generation')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                            help='input batch size for training (default: 128)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
parser.add_argument('--zdim', type=int, default=256, metavar='N',
                            help='dimension of latent variables')
parser.add_argument('--data-path', default='./H5_Format_Data/Subject1', type=str, metavar='DIR',
                            help='path to dataset, which should be concatenated with either _train.h5 or _val.h5 to yield training or validation datasets')

parser.add_argument('--z-path', type=str, default='./Testing_Data_Z/Sess1/Sub1', 
                            help='path to saved z files. Only Z files must be in this path, not other files.')
parser.add_argument('--resume', type=str, default='../Trained_VAE/Checkpoint/checkpoint99.pth.tar',
                            help='checkpoint file name of saved model parameters to load')
parser.add_argument('--img-path', type=str, default='./Rec_Testing_Data/Sess1/Sub1', 
                            help='path to save reconstructed images')
parser.add_argument('--mode', type=str, default='encode', 
                            help='Mode to get data. Choose one of [encode, decode]')

args = parser.parse_args()
torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BetaVAE(z_dim=args.zdim, nc=1).to(device)
if os.path.isfile(args.resume):
    print("==> Loading checkpoint: ", args.resume)
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])
else:
    print('[ERROR] Checkpoint not found: ', args.resume)
    raise RuntimeError




if args.mode.lower() == 'encode':
    '''
    Encoder mode. 
    Image --> z
    '''
    test_loader = load_dataset_test(args.data_path, args.batch_size)

    print('Mode: Encode \n Distribution of Z will be saved at: ' + args.z_path)
    if not os.path.isdir(args.z_path):
        os.system('mkdir '+args.z_path)

    for batch_idx, (xL, xR) in enumerate(test_loader):
        xL = xL.to(device)
        xR = xR.to(device)
        z_distribution = model._encode(xL, xR)

        save_data = {}
        save_data['z_distribution'] = z_distribution.detach().cpu().numpy()
        io.savemat(args.z_path + 'save_z{}.mat'.format(batch_idx), save_data)


elif args.mode.lower() == 'decode':
    '''
    Decoder mode.
    z --> reconstructed image
    '''
    print('Mode: Decode \n Reconstructed images will be saved at: ' + args.img_path)
    if not os.path.isdir(args.z_path):
        os.system('mkdir '+args.z_path)
        print('[ERROR] Dir does not exist: ' + args.z_path)
        raise RuntimeError
    if not os.path.isdir(args.img_path):
        os.system('mkdir '+args.img_path)
    
    filelist = [f for f in os.listdir(args.z_path) if f.split('_')[0] == 'save']
    for batch_idx, filename in enumerate(filelist):

        # z_dist = io.loadmat(os.path.join(args.z_path, filename))
        z_dist = io.loadmat(args.z_path + 'save_z{}.mat'.format(batch_idx))
        z_dist = z_dist['z_distribution']
        mu=z_dist[:, :args.zdim]
        logvar = z_dist[:, args.zdim:]

        #z = model.reparametrize(torch.tensor(mu).to(device), torch.tensor(logvar).to(device))
        z = torch.tensor(mu).to(device)

        #z = model.reparametrize(1, 1)
        #eps_z = z
        #save_data = {}
        #save_data['eps'] = eps_z.detach().cpu().numpy()
        #io.savemat(args.img_path + 'save_z_Test{}.mat'.format(batch_idx), save_data)

        # test_save_Z_mat(z,batch_idx,args.z_path)
        x_recon_L, x_recon_R = model._decode(z)
        #x_recon_L, x_recon_R = model._decode(torch.tensor(mu).to(device))
        save_image_mat(x_recon_R, x_recon_L, args.img_path, batch_idx)

else:
    print('[ERROR] Selected mode: ' +  args.mode + ' is not valid. \n Choose either [encode, decode]')

