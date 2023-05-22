import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F

from torchvision import transforms as T

from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint


# Flatten output
def flatten(t):
    return t.reshape(t.shape[0], -1)

# Set gradients of a certain model
def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

# Normalized MSE Loss
def loss_mse(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

# Register hook wrapper for encoder as nn.Module
class ForwardHookWrapper(nn.Module):
    def __init__(self, net, layer = -2, flatten=False):
        super().__init__()
        self.net = net
        self.layer = layer
        self.flatten = flatten
        self.hidden = {}
        self._register_hook()
        
    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, input, output):
        device = input[0].device
        if self.flatten:
            self.hidden[device] = flatten(output)
        else:
            self.hidden[device] = output

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)

    def forward(self, x):
        # original forward pass
        _ = self.net(x)
        # get hooked output after forward pass
        hidden = self.hidden.pop(x.device)
        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden


class LatentODEfunc(nn.Module):
    def __init__(self, latent_dim=512, nhidden=4096, odenorm=None, num_layer=2):
        super(LatentODEfunc, self).__init__()
        
        self.elu = nn.ELU(inplace=True)
        
        if num_layer == 1:
            self.func = nn.Linear(latent_dim, latent_dim)
        elif num_layer == 2:
            self.func = nn.Sequential(
                nn.Linear(latent_dim, nhidden),
                nn.BatchNorm1d(nhidden),
                nn.ELU(inplace=True),
                nn.Linear(nhidden, latent_dim)
            )
        else:
            self.func = nn.Sequential(
                nn.Linear(latent_dim, nhidden),
                nn.BatchNorm1d(nhidden),
                nn.ELU(inplace=True),
                nn.Linear(nhidden, nhidden),
                nn.BatchNorm1d(nhidden),
                nn.ELU(inplace=True),
                nn.Linear(nhidden, latent_dim)
            )
        

    def forward(self, t, x):
        out = self.func(x)
        return out

# MLP class for projector and predictor
def MLP(dim, projection_size, hidden_size=4096, num_layer=2):
    if num_layer == 1:
        return nn.Linear(dim, projection_size)
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

def SimSiamMLP(dim, projection_size, hidden_size=4096):
    return nn.Sequential(
        nn.Linear(dim, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size, bias=False),
        nn.BatchNorm1d(projection_size, affine=False)
    )
    
# Neural ODE class for projector and predictor
class LatentODEblock(nn.Module):
    def __init__(self, odefunc: nn.Module = LatentODEfunc, solver: str = 'dopri5',
                 latent_dim=512, nhidden=4096,
                 rtol: float = 1e-4, atol: float = 1e-4, adjoint: bool = False,
                 odenorm = None, num_layer=2):
        super().__init__()
        self.odefunc = odefunc(latent_dim=latent_dim, nhidden=nhidden, odenorm=odenorm, num_layer=num_layer)
        self.rtol = rtol
        self.atol = atol
        self.solver = solver
        self.use_adjoint = adjoint
        self.ode_method = odeint_adjoint if adjoint else odeint
        self.integration_time_f = torch.tensor([0, 1], dtype=torch.float32)
        self.integration_time_b = torch.tensor([1, 0], dtype=torch.float32)
        self.odenorm = odenorm

    def forward(self, x: torch.Tensor, integforward=True, integration_time=None):
        if integration_time is None:
            if integforward:
                integration_time = self.integration_time_f
            else:
                integration_time = self.integration_time_b
        integration_time = integration_time.to(x.device)

        out = self.ode_method(
            self.odefunc, x, integration_time, rtol=self.rtol,
            atol=self.atol, method=self.solver)
        # print(out.size())
        return out

# 3D PC Net class
class PCNet_3d(nn.Module):
    def __init__(self, 
                encoder: nn.Module, 
                use_autoregressor: bool,
                use_projector: bool, 
                projection_size: int, 
                projection_hidden_size: int, 
                loss_mode: int, closed_loop: bool,
                use_momentum: bool, use_stop_grad: bool,
                pred_step: int, num_clip: int, clip_len: int,
                num_layer, use_ode: bool, 
                rtol: float = 1e-4, 
                atol: float = 1e-4, 
                adjoint: bool = False):
        super().__init__()
        self.encoder = encoder
        self.projector = select_projector(use_projector, projection_size, projection_hidden_size)
        self.predictor = select_predictor(num_layer, use_ode, rtol, atol, adjoint)

    def forward(self, x:torch.Tensor):
        images = x[:, :, :self.clip_len, :, :]
        images2_list = []
        for i in range(args.num_seq-1):
            # index = i*args.seq_len
            index2 = (i+1)*args.seq_len
            index3 = (i+2)*args.seq_len
            # images = video[:, :, index:index2, :, :]
            images2 = video[:, :, index2:index3, :, :]
            # images = images.to(cuda)
            images2 = images2.to(cuda)
            images2_list.append(images2)

        
