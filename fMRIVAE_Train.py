from __future__ import print_function
import argparse
import torch
from utils import *
from fMRIVAE_Model import *
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import os
import scipy.io as sio
import torch.optim as optim


parser = argparse.ArgumentParser(description='VAE for fMRI data')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--zdim', type=int, default=256, metavar='N',
                    help='dimension of latent variables (default: 256)')
parser.add_argument('--vae-beta', default=10, type=float, 
                    help='beta parameter for KL-term (default: 10)')
parser.add_argument('--lr', default=1e-4, type=float, 
                    help='learning rate (default : 1e-4)')
parser.add_argument('--beta1', default=0.9, type=float, 
                    help='Adam optimizer beta1')
parser.add_argument('--beta2', default=0.999, type=float, 
                    help='Adam optimizer beta2')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--data-path', default='./Training_Set/Subject150_RS_Filtered', type=str, metavar='DIR',
                    help='path to dataset, which should be concatenated with either _train.h5 or _val.h5 to yield training or validation datasets')
parser.add_argument('--apply-mask', default=True, type=bool,
                    help='Whether apply a mask of the crtical surface to the MSe loss function')
parser.add_argument('--Output-path',default='./Output_Temp/', type=str,
                    help='Path to save results')
parser.add_argument('--mother-path', default='./VAE_Model/', type=str,
                    help='Path to mother folder')





def save_image_mat(img_r, img_lu, result_path):

    save_data = {}

    save_data['recon_L'] = img_l.detach().cpu().numpy()

    save_data['recon_R'] = img_r.detach().cpu().numpy()

    sio.savemat(result_path+'save_img_mat.mat', save_data)

    print('image saved as mat')

# initialization
args = parser.parse_args()
start_epoch = 0
torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# path
result_path = args.Output_path
log_path = args.mother_path + '/Log/'
checkpoint_path = args.mother_path + '/Checkpoint/'
figure_path = args.mother_path + '/Figure/'
# create folder
if not os.path.isdir(args.mother_path):
    os.system('mkdir '+args.mother_path)
if not os.path.isdir(result_path):
    os.system('mkdir '+result_path)
if not os.path.isdir(log_path):
    os.system('mkdir '+log_path)
if not os.path.isdir(checkpoint_path):
    os.system('mkdir '+checkpoint_path)
if not os.path.isdir(figure_path):
    os.system('mkdir '+figure_path)
# create log name
rep = 0
stat_name = f'Zdim_{args.zdim}_Vae-beta_{args.vae_beta}_Lr_{args.lr}_Batch-size_{args.batch_size}_'+'your_extra_notation'
while(os.path.isfile(log_path+stat_name+f'_Rep_{rep}.txt')):
    rep += 1
log_name = log_path+stat_name+f'_Rep_{rep}.txt'


# dataloader 
kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
train_dir = args.data_path + '_train.h5'
val_dir = args.data_path + '_val.h5'
train_set = H5Dataset(train_dir)
val_set = H5Dataset(val_dir)
train_loader = torch.utils.data.DataLoader(train_set,batch_size=args.batch_size, shuffle=True, **kwargs)
val_loader = torch.utils.data.DataLoader(val_set,batch_size=args.batch_size, shuffle=False, **kwargs)

# load the mask form the data loader
if args.apply_mask:
    print('Will apply a mask to the loss function') 
    left_mask = torch.from_numpy(val_set.LeftMask).to(device)
    right_mask = torch.from_numpy(val_set.RightMask).to(device)
else:
    print('Will not apply a mask to the loss function')
    left_mask = None
    right_mask = None

# model
model = BetaVAE(z_dim=args.zdim, nc=1).to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)


# optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# resume
if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

# loss function
def loss_function(xL, xR, x_recon_L, x_recon_R, mu, logvar, beta, left_mask, right_mask):

    Image_Size=xL.size(3)

    beta/=Image_Size**2
    
    # print('====> Image_Size: {} Beta: {:.8f}'.format(Image_Size, beta))

    # R_batch_size=xR.size(0)
    # Tutorial on VAE Page-14
    # log[P(X|z)] = C - \frac{1}{2} ||X-f(z)||^2 // \sigma^2 
    #             = C - \frac{1}{2} \sum_{i=1}^{N} ||X^{(i)}-f(z^{(i)}||^2 // \sigma^2
    #             = C - \farc{1}{2} N * F.mse_loss(Xhat-Xtrue) // \sigma^2
    # log[P(X|z)]-C = - \frac{1}{2}*2*192*192//\sigma^2 * F.mse_loss
    # Therefore, vae_beta = \frac{1}{36864//\sigma^2}
    if left_mask is not None:
        MSE_L = F.mse_loss(x_recon_L * left_mask.detach(), xL * left_mask.detach(), size_average=True)
        MSE_R = F.mse_loss(x_recon_R * right_mask.detach(), xR * right_mask.detach(), size_average=True)  
    else: # left and right masks are None
        MSE_L = F.mse_loss(x_recon_L, xL, size_average=True)
        MSE_R = F.mse_loss(x_recon_R, xR, size_average=True)

       # MSE_L = F.mse_loss(x_recon_L, xL, size_average=False).div(L_batch_size)
       # MSE_R = F.mse_loss(x_recon_R, xR, size_average=False).div(R_batch_size)

    # KLD is averaged across batch-samples
    KLD = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean()

    return KLD * beta + MSE_L + MSE_R

def train_save_image_mat(img_r, img_l,recon_r,recon_l,loss,recon_loss,N_Epoch, result_path):

    save_data = {}

    save_data['recon_L'] = recon_l.detach().cpu().numpy()

    save_data['recon_R'] = recon_r.detach().cpu().numpy()

    save_data['img_R'] = img_r.detach().cpu().numpy()

    save_data['img_L'] = img_l.detach().cpu().numpy()

    save_data['Loss'] = loss

    save_data['Recon_Loss'] = recon_loss

    sio.savemat(result_path + '/train_save_img_mat' + str(N_Epoch) + '.mat', save_data)

    print('train image saved as mat')

def test_save_image_mat(img_r, img_l,recon_r,recon_l,loss,recon_loss,N_Epoch, result_path):

    save_data = {}

    save_data['recon_L'] = recon_l.detach().cpu().numpy()

    save_data['recon_R'] = recon_r.detach().cpu().numpy()

    save_data['img_R'] = img_r.detach().cpu().numpy()

    save_data['img_L'] = img_l.detach().cpu().numpy()

    save_data['Loss'] = loss

    save_data['Recon_Loss'] = recon_loss

    sio.savemat(result_path + '/test_save_img_mat' + str(N_Epoch) + '.mat', save_data)

    print('test image saved as mat')

def train(epoch):

    model.train()
    train_loss = 0
    recon_loss = 0

    for batch_idx, (xL, xR) in enumerate(train_loader):
        xL = xL.to(device)
        xR = xR.to(device)
        optimizer.zero_grad()
        x_recon_L, x_recon_R, mu, logvar = model(xL, xR)
        
        Recon_Error = loss_function(xL, xR, x_recon_L, x_recon_R, mu, logvar, 0, left_mask, right_mask)
        
        recon_loss +=Recon_Error.item()

        loss = loss_function(xL, xR, x_recon_L, x_recon_R, mu, logvar, args.vae_beta, left_mask, right_mask)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % args.log_interval == 0:

           print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\trecon_Loss: {:.6f}\tKLD:{:.6f}'.format(epoch, batch_idx * len(xL), len(train_loader.dataset),100. * batch_idx / len(train_loader),loss.item() / len(xL),Recon_Error.item() / len(xL), xL.size(3)**2 * (loss.item() - Recon_Error.item())/(args.vae_beta * len(xL))))
 
        #if batch_idx == 0:

           # train_save_image_mat(xR, xL,x_recon_R,x_recon_L,loss.item()/len(xL),Recon_Error.item()/len(xL),epoch,result_path)

    stat_file = open(log_name,'a+')
    stat_file.write('Epoch:{} Average training loss: {:.8f} Average reconstruction loss: {:.8f}'.format(epoch, train_loss / batch_idx, recon_loss / batch_idx))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / batch_idx))

def test(epoch):
    model.eval()
    test_loss = 0
    recon_loss = 0
    with torch.no_grad():
        for i, (xL, xR) in enumerate(val_loader):
            xL = xL.to(device)
            xR = xR.to(device)
            x_recon_L, x_recon_R, mu, logvar = model(xL, xR)
            test_loss += loss_function(xL, xR, x_recon_L, x_recon_R, mu, logvar, args.vae_beta, left_mask, right_mask).item()
            recon_loss += loss_function(xL, xR, x_recon_L, x_recon_R, mu, logvar, 0, left_mask, right_mask).item()
            #if i == 0:
                #n = min(xL.size(0), 8)
                #img_len = xL.size(2)
                # left
                #comparisonL = torch.cat([xL[:n],x_recon_L.view(args.batch_size, 1, img_len, img_len)[:n]])
                #save_image(comparisonL.cpu(),figure_path+'reconstruction_left_epoch_' + str(epoch) + '.png', nrow=n)
                # right
                #comparisonR = torch.cat([xR[:n],x_recon_R.view(args.batch_size, 1, img_len, img_len)[:n]])
                #save_image(comparisonR.cpu(),figure_path+'reconstruction_right_epoch_' + str(epoch) + '.png', nrow=n)
                #test_save_image_mat(xR,xL,x_recon_R,x_recon_L,test_loss,recon_loss,epoch,result_path)

    test_loss /= i
    stat_file = open(log_name,'a+')
    stat_file.write('Epoch:{} Average validation loss: {:.8f}'.format(epoch, test_loss))
    print('====> Test set loss: {:.4f}'.format(test_loss))

def save_checkpoint(state, filename):
    torch.save(state, filename)

if __name__ == "__main__":
    test(0)
    for epoch in range(start_epoch+1, args.epochs):
        train(epoch)
        test(epoch)  
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict()
}, checkpoint_path+'/checkpoint'  + str(epoch) + '.pth.tar')
        scheduler.step()
