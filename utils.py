import h5py
import torch
import torch.utils.data as data
import torch.multiprocessing
import scipy.io as sio
# torch.multiprocessing.set_start_method('spawn')

class H5Dataset(data.Dataset):
    def __init__(self, H5Path):
        super(H5Dataset, self).__init__()
        self.H5File = h5py.File(H5Path,'r')       
        self.LeftData = self.H5File['LeftData']
        self.RightData = self.H5File['RightData']
        self.LeftMask = self.H5File['LeftMask'][:]
        self.RightMask = self.H5File['RightMask'][:]

    def __getitem__(self, index):
        return (torch.from_numpy(self.LeftData[index,:,:,:]).float(),
                torch.from_numpy(self.RightData[index,:,:,:]).float())
 
    def __len__(self):
        return self.LeftData.shape[0]



def save_image_mat(img_r, img_l, result_path, idx):
    save_data = {}
    save_data['recon_L'] = img_l.detach().cpu().numpy()
    save_data['recon_R'] = img_r.detach().cpu().numpy()
    sio.savemat(result_path+'img{}.mat'.format(idx), save_data)

def load_dataset(data_path, batch_size):
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_dir = data_path + '_train.h5'
    val_dir = data_path + '_val.h5'
    train_set = H5Dataset(train_dir)
    val_set = H5Dataset(val_dir)
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size, shuffle=False, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_set,batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, val_loader

def load_dataset_test(data_path, batch_size):
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    test_dir = data_path + '.h5'
    test_set = H5Dataset(test_dir)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size, shuffle=False, **kwargs)
    return test_loader