import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F

from torchvision import transforms as T

from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint


# helper functions

def flatten(t):
    return t.reshape(t.shape[0], -1)

def get_module_device(module):
    return next(module.parameters()).device


# loss fn

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def normalized_mse_loss(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

def vic_reg_loss(x, y, mse_l, std_l, cov_l):
    (B, D) = x.shape
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    loss_mse = 2 - 2 * (x * y).sum(dim=-1)
    loss_mse = mse_l * loss_mse

    std_x = torch.sqrt(x.var(dim=0) + 0.0001)
    std_y = torch.sqrt(y.var(dim=0) + 0.0001)
    loss_std = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2
    loss_std = std_l * loss_std

    cov_x = (x.T @ x) / (B - 1)
    cov_y = (y.T @ y) / (B - 1)
    loss_cov = off_diagonal(cov_x).pow_(2).sum().div(D) + off_diagonal(cov_y).pow_(2).sum().div(D)
    loss_cov = cov_l * loss_cov

    total_loss = loss_mse + loss_std + loss_cov

    return total_loss

def vic_reg_nonorm_loss(x, y, mse_l, std_l, cov_l, sub=False): #same as paper
    (B, D) = x.shape
    
    loss_mse = F.mse_loss(x, y)
    loss_mse = mse_l * loss_mse # 25

    # applied in VICReg
    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)

    if sub:
        x=x[:,torch.randperm(D)]
        y=y[:,torch.randperm(D)]
        D = int(D/3)
        x=x[:,:D]
        y=y[:,:D]

    std_x = torch.sqrt(x.var(dim=0) + 0.0001)
    std_y = torch.sqrt(y.var(dim=0) + 0.0001)
    loss_std = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2
    loss_std = std_l * loss_std # 25

    cov_x = (x.T @ x) / (B - 1)
    cov_y = (y.T @ y) / (B - 1)
    loss_cov = off_diagonal(cov_x).pow_(2).sum().div(D) + off_diagonal(cov_y).pow_(2).sum().div(D)
    loss_cov = cov_l * loss_cov # 1

    total_loss = loss_mse + loss_std + loss_cov

    return total_loss

def mse_loss(x, y):
    return F.mse_loss(x, y)

def std_loss(x, lam=0.0001):
    x = x - x.mean(dim=0)
    std_x = torch.sqrt(x.var(dim=0) + lam)
    return torch.mean(F.relu(1 - std_x)) 

def cov_loss(x):
    (B, D) = x.shape
    x = x - x.mean(dim=0)
    cov_x = (x.T @ x) / (B - 1)
    return off_diagonal(cov_x).pow_(2).sum().div(D)


def infoNCE(nn, p, temperature=0.1):
    nn = F.normalize(nn, dim=1)
    p = F.normalize(p, dim=1)
    # nn = gather_from_all(nn)
    # p = gather_from_all(p)
    logits = nn @ p.T
    logits /= temperature
    logits = logits.to(device='cuda')
    n = p.shape[0]
    labels = torch.arange(0, n, dtype=torch.long).to(device='cuda')
    loss = F.cross_entropy(logits, labels)
    return loss


# helper models

# MLP class for projector and predictor

def MLP_sub(dim, projection_size, hidden_size=4096, num_layer=2):
    if num_layer == 1:
        return nn.Sequential(
            nn.Linear(dim, projection_size)
        )
    elif num_layer == 2:
        return nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )
    else:
        return nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )

class MLP(nn.Module):
    def __init__(
        self,
        dim, 
        projection_size, 
        hidden_size=4096, 
        num_layer=2, 
        bn_last=False, 
        relu_last=False
    ):
        super().__init__()
        self.net = MLP_sub(dim, projection_size, hidden_size, num_layer)
        if bn_last:
            self.net.add_module("last_bn", nn.BatchNorm1d(projection_size))
        self.relu_last = relu_last
        
    def forward(self, x, forward = True):
        out = self.net(x)
        if self.relu_last:
            out = self.relu(out)
        return out

# residual MLP where relu could as the last layer
class Res_MLP(nn.Module):
    def __init__(
        self,
        dim, 
        projection_size, 
        hidden_size=4096, 
        num_layer=2, 
        bn_last=False,
        relu_last=False
    ):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.net = MLP_sub(dim, projection_size, hidden_size, num_layer)
        if bn_last:
            self.net.add_module("last_bn", nn.BatchNorm1d(projection_size))
        self.relu_last = relu_last
        
    def forward(self, x, forward = True):
        out = self.net(x)
        out += x
        if self.relu_last:
            out = self.relu(out)
        return out

# For the kind of predictor that predict backward or forward 
class Res_MLP_D(nn.Module):
    def __init__(
        self,
        dim, # input
        projection_size, # output
        hidden_size=4096, # hidden
        num_layer=2, 
        bn_last=False,
        relu_last=False
    ):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.net = MLP_sub(dim, projection_size, hidden_size, num_layer)
        if bn_last:
            self.net.add_module("last_bn", nn.BatchNorm1d(projection_size))
        self.relu_last = relu_last
        
    def forward(self, x, forward = True):
        out = self.net(x)
        if forward:
            out += x
        else:
            out -= x
        if self.relu_last:
            out = self.relu(out)
        return out

# ODE classes for predictor
def ODE_sub(dim, projection_size, hidden_size=4096, num_layer=2):
    if num_layer == 1:
        return nn.Sequential(
            nn.Linear(dim, projection_size),
        )
    elif num_layer == 2:
        return nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, projection_size),
        )
    else:
        return nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, projection_size),
        )

class LatentODEfunc(nn.Module):
    def __init__(
            self, 
            dim, 
            projection_size=512, 
            hidden_size=4096, 
            num_layer=2, 
            bn_last=False):
        super(LatentODEfunc, self).__init__()
        self.func = ODE_sub(dim, projection_size, hidden_size, num_layer)
        if bn_last:
            self.func.add_module("last_bn", nn.BatchNorm1d(projection_size))

    def forward(self, t, x): # to use odeint, the given nn.Module as ODE func need to take t as input as well?
        out = self.func(x)
        return out

class LatentODEblock(nn.Module):
    def __init__(
            self, 
            dim, 
            projection_size=512, 
            hidden_size=4096, 
            num_layer=2,
            bn_last: bool = False,
            relu_last: bool = False,
            odefunc=LatentODEfunc, 
            solver: str = 'dopri5',
            rtol: float = 1e-4, 
            atol: float = 1e-4, 
            adjoint: bool = False
            ):
        super().__init__()
        self.odefunc = odefunc(dim=dim, projection_size=projection_size, hidden_size=hidden_size, num_layer=num_layer, bn_last=bn_last)
        self.rtol = rtol
        self.atol = atol
        self.solver = solver
        self.use_adjoint = adjoint
        self.ode_method = odeint_adjoint if adjoint else odeint
        self.integration_time_f = torch.tensor([0, 1.0], dtype=torch.float32)
        self.integration_time_b = torch.tensor([1.0, 0], dtype=torch.float32)

    def forward(self, x: torch.Tensor, forward=True, integration_time=None):
        if integration_time is None:
            if forward:
                integration_time = self.integration_time_f
            else:
                integration_time = self.integration_time_b
        integration_time = integration_time.to(x.device)

        out = self.ode_method(
            self.odefunc, x, integration_time, rtol=self.rtol,
            atol=self.atol, method=self.solver)
        # print(out.size())
        return out[1:] # omit the first output


