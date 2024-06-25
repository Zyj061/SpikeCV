import torch
import torch.nn as nn

from .representation import Representation
from .noise_estimator import NoiseEstimator
from .color_prior import ColorPrior


class SJDDNet(nn.Module):
    def __init__(self, n=39, wsize=15, blocks=5, iter_num=3, init_mu=0.5):
        super(SJDDNet, self).__init__() 
        self.repre = Representation(wsize, blocks)
        self.estimator = NoiseEstimator(n)
        self.prior = ColorPrior()
        self.iter_num = iter_num
        # init alpha
        self.mus = nn.ParameterList([torch.nn.Parameter(
            torch.FloatTensor(1), requires_grad=True) for _ in range(iter_num)])
        for mu in self.mus:
            mu.data.fill_(init_mu)

    def forward(self, spk, mask):
        y = self.repre(spk, mask)
        x = y # init
        noise_map = self.estimator(spk, mask)
        mask = torch.cat((torch.cat((
            mask[:,0:1,:,:] , mask[:,1:2,:,:] + mask[:,2:3,:,:]), axis=1), mask[:,3:4,:,:]), axis=1)
        for i in range(self.iter_num):
            z = self.prior(x, noise_map)
            x = (y + self.mus[i] * z) / (y.shape[1] + self.mus[i])
        return x, noise_map
