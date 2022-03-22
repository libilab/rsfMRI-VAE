"""model.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from collections import Iterable
from torch.autograd import Variable

class BetaVAE(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, z_dim=64, nc=1, cirpad_dire=(False, True)):
        super(BetaVAE, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.cirpad_dire = cirpad_dire
   
        self.ocs = [64, 128, 128, 256, 256]
        self.nLays = len(self.ocs)
        self.topW = int(192/2**self.nLays)

        # encoder
        self.ConvL = nn.Conv2d(1,int(self.ocs[0]/2),8,2,0)  # pad=3, only in forward
        self.ConvR = nn.Conv2d(1,int(self.ocs[0]/2),8,2,0)  # pad=3, only in forward # B, 128, 96, 96
        self.EncConvs = nn.ModuleList([nn.Conv2d(self.ocs[i-1], self.ocs[i], 4, 2, 0) for i in range(1, self.nLays)]) # pad=1 only in forward     
        self.fc1 = nn.Linear(self.ocs[-1]*self.topW**2, z_dim*2)

        # decoder
        self.fc2 = nn.Linear(z_dim, self.ocs[-1]*self.topW**2)
        self.DecConvs = nn.ModuleList([nn.ConvTranspose2d(self.ocs[i], self.ocs[i-1], 4, 2, 3) for i in range(4,0,-1)]) # pad=1; dilation * (kernel_size - 1) - padding = 6  (later in forward)
        self.tConvL = nn.ConvTranspose2d(int(self.ocs[0]/2), nc, 8, 2, 9) # pad=3 later; dilation * (kernel_size - 1) - padding = 4  (later in forward)
        self.tConvR = nn.ConvTranspose2d(int(self.ocs[0]/2), nc, 8, 2, 9) # pad=3 later

        self.relu = nn.ReLU(inplace=True)

        self.weight_init()

    def cirpad(self, x, padding, cirpad_dire):
        # x            is    input
        # padding      is    the size of pading
        # cirpad_dire  is    (last_dim_pad, second_to_last_dim_pad)
        
        # >>> t4d = torch.empty(3, 3, 4, 2)
        # >>> p2d = (1, 1, 2, 2) # pad last dim by (1, 1) and 2nd to last by (2, 2)
        # >>> out = F.pad(t4d, p2d, "constant", 0)
        # >>> print(out.size())
        # torch.Size([3, 3, 8, 4])

        # last dim
        if cirpad_dire[0] is True:
            x = F.pad(x, (padding, padding, 0, 0), 'circular')
        else:
            x = F.pad(x, (padding, padding, 0, 0), "constant", 0)
        
        # second last dim
        if cirpad_dire[1] is True:
            x = F.pad(x, (0, 0, padding, padding), 'circular')
        else:
            x = F.pad(x, (0, 0, padding, padding), "constant", 0)
            
        return x


    def weight_init(self):
        for block in self._modules:
            if isinstance(self._modules[block], Iterable):
                for m in self._modules[block]:
                    m.apply(kaiming_init)
            else:
                self._modules[block].apply(kaiming_init)

    def _encode(self, xL, xR):
        xL = self.cirpad(xL, 3, self.cirpad_dire) 
        xR = self.cirpad(xR, 3, self.cirpad_dire) 
        x = torch.cat((self.ConvL(xL), self.ConvR(xR)), 1)
        x = self.relu(x)
        for lay in range(self.nLays-1):
            x = self.cirpad(x, 1, self.cirpad_dire) 
            x = self.relu(self.EncConvs[lay](x))
        x = x.view(-1, self.ocs[-1]*self.topW*self.topW)
        x = self.fc1(x)
        return x

    def _decode(self, z):
        x = self.relu(self.fc2(z).view(-1 , self.ocs[-1], self.topW, self.topW))

        #x.size()
        #print(x.size())
        for lay in range(self.nLays-1):
            #print(x.shape)
            x = self.cirpad(x, 1, self.cirpad_dire) 
            x = self.relu(self.DecConvs[lay](x))
            #print(x.size())


        xL, xR = torch.chunk(x, 2, dim=1)

        xrL = self.tConvL(self.cirpad(xL, 3, self.cirpad_dire))

        xrR = self.tConvR(self.cirpad(xR, 3, self.cirpad_dire))
        return xrL, xrR

    def reparametrize(self, mu, logvar):
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std*eps


    def forward(self, xL, xR):
        distributions = self._encode(xL, xR)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = self.reparametrize(mu, logvar)
        x_recon_L, x_recon_R = self._decode(z)
        return x_recon_L, x_recon_R, mu, logvar

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)): # Shall we apply init to ConvTranspose2d?
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()

#if __name__ == "__main__":
#    m = BetaVAE_H()
#    a=torch.ones(1,1,192,192)
#    out1, out2, _, _ = m(a,a)
#    print(out1.size())




