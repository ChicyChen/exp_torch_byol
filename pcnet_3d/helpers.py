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

def default(val, def_val):
    return def_val if val is None else val

def flatten(t):
    return t.reshape(t.shape[0], -1)

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

def get_module_device(module):
    return next(module.parameters()).device

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

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
    # loss_std = torch.mean(F.relu(1 - std_x)) + torch.mean(F.relu(1 - std_y))
    loss_std = std_l * loss_std # 25

    cov_x = (x.T @ x) / (B - 1)
    cov_y = (y.T @ y) / (B - 1)
    loss_cov = off_diagonal(cov_x).pow_(2).sum().div(D) + off_diagonal(cov_y).pow_(2).sum().div(D)
    # loss_cov = off_diagonal(cov_x).pow_(2).sum().div(D) / 2 + off_diagonal(cov_y).pow_(2).sum().div(D) / 2
    loss_cov = cov_l * loss_cov # 1
    # print(loss_cov)

    total_loss = loss_mse + loss_std + loss_cov

    return total_loss

def reg_mse_loss(x, y):
    # x2 = F.normalize(x, dim=-1, p=2)
    # y2 = F.normalize(y, dim=-1, p=2)
    x2 = F.normalize(x, dim=0, p=2)
    y2 = F.normalize(y, dim=0, p=2)
    return F.mse_loss(x2, y2)

def mse_loss(x, y):
    loss = F.mse_loss(x, y)
    print("mse loss:", loss)
    return loss

def std_loss(x, lam=0.0001):
    y = x - x.mean(dim=0)
    std_x = torch.sqrt(y.var(dim=0) + lam)
    loss_std = torch.mean(F.relu(1 - std_x)) / 2

    print("std loss:", loss_std)
    return loss_std

# def std_n_loss(x, lam=0.0001):
#     y = x - x.mean(dim=0)
#     std_x = torch.sqrt(y.var(dim=0) + lam)
#     return torch.mean(F.relu(std_x - 1)) / 2

def std_2_loss(x, lam=0.0001):
    (B, D) = x.shape
    y = x - x.mean(dim=0)
    # y = F.normalize(x, dim=-1, p=2)
    # y = F.normalize(x, dim=0, p=2)

    std_x = y.var(dim=0)
    loss_std = (std_x - 1).pow_(2).sum().div(D)
    print("std:", std_x.mean())
    print("std loss:", loss_std)

    return loss_std

    # std_x = torch.sqrt(x.var(dim=0) + lam)
    # return torch.mean(F.relu(1 - std_x)) / 2

def std_4_loss(x, lam=0.0001):
    (B, D) = x.shape
    y = x - x.mean(dim=0)
    # y = F.normalize(x, dim=-1, p=2)
    # y = F.normalize(x, dim=0, p=2)

    std_x = y.var(dim=0)
    loss_std = (std_x - 1).pow_(4).sum().div(D)
    print("std loss:", loss_std)

    return loss_std

def std_3_loss(x, lam=0.0001):
    (B, D) = x.shape
    y1 = F.relu(x)
    y2 = F.relu(-x)
    # y = F.normalize(x, dim=-1, p=2)
    # y = F.normalize(x, dim=0, p=2)

    std_x1 = y1.var(dim=0)
    std_x2 = y2.var(dim=0)
    loss_std = (std_x1 - 1).pow_(2).sum().div(D) + (std_x2 - 1).pow_(2).sum().div(D)
    print("std loss:", loss_std)

    return loss_std

def cov_loss(x):
    (B, D) = x.shape
    y = x - x.mean(dim=0)
    # y = F.normalize(x, dim=-1, p=2)
    # y = F.normalize(x, dim=0, p=2)

    cov_x = (y.T @ y) / (B - 1)
    loss_std = off_diagonal(cov_x).pow_(2).sum().div(D)
    # print(loss_std)

    print("cov loss:", loss_std)
    return loss_std

def cov_2_loss(x):
    (B, D) = x.shape
    y = x - x.mean(dim=0)
    # y = F.normalize(x, dim=-1, p=2)
    # y = F.normalize(x, dim=0, p=2)

    cov_x = (y.T @ y) / (B - 1)
    loss_std = off_diagonal(cov_x).pow_(2).sum().div(D*D/2048)
    # print(loss_std)

    print("cov loss:", loss_std)
    return loss_std

def cov_4_loss(x):
    (B, D) = x.shape
    y = x - x.mean(dim=0)
    # y = F.normalize(x, dim=-1, p=2)
    # y = F.normalize(x, dim=0, p=2)

    cov_x = (y.T @ y) / (B - 1)
    loss_std = off_diagonal(cov_x).pow_(4).sum().div(D)
    # print(loss_std)

    print("cov loss:", loss_std)
    return loss_std

def sparse_loss(x):
    B = x.shape[0]
    D = x.shape[-1]

    # print(B, D)

    # demean, norm
    # y = x - x.mean(dim=0)
    # y = F.normalize(y, dim=-1, p=2)

    # normalize each difference
    y = F.normalize(x, dim=-1, p=2)
    print("Vector L2 norm", torch.norm(y[0,0,:], p=2))
    print("Vector L1 norm", torch.norm(y[0,0,:], p=1))

    # # nothing
    # y = x

    # form 1A
    loss = torch.norm(y, p=1) / B / D
    # # form 1B
    # loss = torch.norm(x, p=1) / B 

    # # potential 1
    # l1 = torch.norm(y, p=1) / B / D
    # loss = torch.sqrt(l1.pow_(2) + 0.0001)

    # # potential 2
    # loss = torch.sqrt(y.pow_(2) + 0.0001).sum() / B / D 

    # # potential 2 2
    # loss = torch.sqrt(y.pow_(2) + 0.000001).sum() / B / D 

    return loss



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



# exponential moving average

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# helper models

# MLP class for projector and predictor
def SimSiamMLP(dim, projection_size, hidden_size=4096, num_layer=3, bn_last=True):
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

# def MLP(dim, projection_size, hidden_size=4096, num_layer=2, bn_last=False, relu_last=False):
#     sub_module = MLP_sub(dim, projection_size, hidden_size, num_layer)
#     if bn_last:
#         sub_module.add_module("last_bn", nn.BatchNorm1d(projection_size))
#     return sub_module

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
            rtol: float = 1e-1, 
            atol: float = 1e-1, 
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


# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets

class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, layer = -2, use_simsiam_mlp = False, use_projector = True, num_layer = 2, bn_last = False):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size
        self.num_layer = num_layer

        self.use_simsiam_mlp = use_simsiam_mlp
        self.use_projector = use_projector
        self.bn_last = bn_last

        self.hidden = {}
        self.hook_registered = False

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
        self.hidden[device] = flatten(output)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape # input dimension
        if self.num_layer > 0:
            create_mlp_fn = MLP if not self.use_simsiam_mlp else SimSiamMLP
            projector = create_mlp_fn(dim, self.projection_size, self.projection_hidden_size, self.num_layer, self.bn_last)
        else:
            projector = nn.Identity()
        return projector.to(hidden)

    def get_representation(self, x):
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        # self.hidden.clear()
        _ = self.net(x)
        # hidden = self.hidden[x.device]
        # self.hidden.clear()
        hidden = self.hidden.pop(x.device)

        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x, return_projection = True):
        representation = self.get_representation(x)

        if not return_projection or not self.use_projector:
            return representation

        projector = self._get_projector(representation)
        projection = projector(representation)
        return projection, representation


