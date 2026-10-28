import torch
import torch.nn as nn

import numpy as np

from empbnn.torch import ResidualBlock, BasicBlock, WideResidualBlock
from empbnn.torch import bnn_to_mp, LinearMomentProp, Conv2dMomentProp, ReLUMomentProp, BnnSampler

# Model definitions

class HYPER(nn.Module):
    def __init__(self, num_in, num_out):
        super().__init__()
        self.num_in = num_in
        self.num_out = num_out

        self.l = torch.nn.Sequential(
            nn.Linear(num_in, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_out),
            nn.Softmax(dim=1)
        )

    def classic_from_bnn(self, bnn_model):
        self.l = torch.nn.Sequential(
            nn.Linear(self.num_in, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, self.num_out),
            nn.Softmax(dim=1)
        )

        for i in [0, 2, 4]:
            self.l[i].weight = nn.Parameter(bnn_model.l[i].mu_weight.detach())
            self.l[i].bias = nn.Parameter(bnn_model.l[i].mu_bias.detach())

    def forward(self, x):
        x = self.l(x)
        return x


class SPLIT_HYPER(nn.Module):
    def __init__(self, original, num_mc):
        super().__init__()

        self.tail = torch.nn.Sequential(*original.l[:-2])
        bnn_to_mp(self.tail, num_mc=num_mc, use_simple=True, use_large=True)
        
        head = torch.nn.Sequential(original.l[-2:])
        self.head = BnnSampler(head, num_mc=num_mc)

    def forward(self, x):
        x = self.tail(x)
        x = self.head(x)
        return x


class B2N2(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Sequential(
            BasicBlock(nn.Conv2d(1, 32, 3, 1, "same"), "relu"),
            BasicBlock(nn.Conv2d(32, 32, 3, 1, "same"), "relu", pool=nn.MaxPool2d(2)),
            BasicBlock(nn.Conv2d(32, 64, 3, 1, "same"), "relu"),
            BasicBlock(nn.Conv2d(64, 64, 3, 1, "same"), "relu", pool=nn.MaxPool2d(2)),
            BasicBlock(nn.Conv2d(64, 128, 3, 1, "same"), "relu"),
            BasicBlock(nn.Conv2d(128, 128, 3, 1, "same"), "relu", pool=nn.MaxPool2d(2)),
            BasicBlock(nn.Linear(2048, 10), "softmax", flatten=True)
        )
    
    def forward(self, x):
        x = self.l(x)
        return x

class SPLIT_B2N2(nn.Module):
    def __init__(self, original, num_mc):
        super().__init__()

        self.tail = torch.nn.Sequential(*original.l[:-1])
        bnn_to_mp(self.tail, num_mc=num_mc, use_simple=True, use_large=True)
        
        head = torch.nn.Sequential(original.l[-1:])
        self.head = BnnSampler(head, num_mc=num_mc)

    def forward(self, x):
        x = self.tail(x)
        x = self.head(x)
        return x



class LENET(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Sequential(
            BasicBlock(nn.Conv2d(1, 32, 3, 1), "relu"),
            BasicBlock(nn.Conv2d(32, 64, 3, 1), "relu", pool=nn.MaxPool2d(2)),
            BasicBlock(nn.Linear(12544, 128), "relu", flatten=True),
            BasicBlock(nn.Linear(128, 10), "softmax")
        )

    def forward(self, x):
        x = self.l(x)
        return x


class SPLIT_LENET(nn.Module):
    def __init__(self, original, num_mc):
        super().__init__()

        self.tail = torch.nn.Sequential(*original.l[:-1])
        bnn_to_mp(self.tail, num_mc=num_mc, use_simple=True, use_large=True)
        
        head = torch.nn.Sequential(original.l[-1:])
        self.head = BnnSampler(head, num_mc=num_mc)

    def forward(self, x):
        x = self.tail(x)
        x = self.head(x)
        return x


class RESNET_TINY(nn.Module):
    def __init__(self):
        super().__init__()

        self.l1 = torch.nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, "same"),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            ResidualBlock(16, 16, 1),
            ResidualBlock(16, 32, 2),
            ResidualBlock(32, 64, 2),
            nn.AvgPool2d(8)
        )

        self.l2 = torch.nn.Sequential(
            nn.Identity(),
            nn.Linear(64, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.l1(x)
        
        if len(x.shape) == 4:
            x = x.permute((0,2,3,1)).flatten(1)
        elif len(x.shape) == 5:
            x = x.permute((0,1,3,4,2)).flatten(2)

        x = self.l2(x)
        return x


class SPLIT_RESNET_TINY(nn.Module):
    def __init__(self, original, num_mc):
        super().__init__()

        self.tail = torch.nn.Sequential(original.l1)
        bnn_to_mp(self.tail, num_mc=num_mc, use_simple=True, use_large=True)
        
        head = torch.nn.Sequential(original.l2)
        self.head = BnnSampler(head, num_mc=num_mc)

    def forward(self, x):
        x = self.tail(x)
        x = x.permute((0,1,3,4,2)).flatten(2)
        x = self.head(x)
        return x


class WIDE_RESNET(nn.Module):
    def __init__(self, n, k, in_channels, num_classes):
        super().__init__()
        # Number of layers for the Residual Blocks
        nrb = (n - 4) / 6
        # Channels using the widen factor k
        nchannels = np.array([16, 32, 64]) * k

        self.l0 = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, 1, "same"),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.l1 = nn.Sequential(
            WideResidualBlock(nrb, 16, nchannels[0], 1),
            WideResidualBlock(nrb, nchannels[0], nchannels[1], 2),
            WideResidualBlock(nrb, nchannels[1], nchannels[2], 2),
            nn.AvgPool2d(8)
        )
        self.l2 = nn.Sequential(
            nn.Identity(),
            nn.Linear(nchannels[2], num_classes)
        )

    def forward(self, x):
        x = self.l0(x)
        x = self.l1(x)
        
        if len(x.shape) == 4:
            x = x.permute((0,2,3,1)).flatten(1)
        elif len(x.shape) == 5:
            x = x.permute((0,1,3,4,2)).flatten(2)

        x = self.l2(x)
        return x

class WIDE_RESNET_INF(nn.Module):
    def __init__(self, trained_model):
        super().__init__()
        self.tm = trained_model
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.tm(x)
        x = self.softmax(x)
        return x


class WIDE_RESNET_SPLIT(nn.Module):
    def __init__(self, original, num_mc):
        super().__init__()
        
        self.tail = torch.nn.Sequential(original.tm.l0, original.tm.l1)
        bnn_to_mp(self.tail, num_mc=num_mc, use_simple=True, use_large=True)
    
        head = torch.nn.Sequential(original.tm.l2, original.softmax)
        self.head = BnnSampler(head, num_mc=num_mc)

    def forward(self, x):
        x = self.tail(x)
        x = x.permute((0,1,3,4,2)).flatten(2)
        x = self.head(x)
        return x
