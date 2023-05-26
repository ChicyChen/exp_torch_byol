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

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

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

# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets

class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, layer = -2, use_simsiam_mlp = False, num_layer = 2, use_projector = True):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size
        self.num_layer = num_layer

        self.use_simsiam_mlp = use_simsiam_mlp
        self.use_projector = use_projector

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

    @singleton('projector') # make sure only one instance is allocated in memory
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        create_mlp_fn = MLP if not self.use_simsiam_mlp else SimSiamMLP
        projector = create_mlp_fn(dim, self.projection_size, self.projection_hidden_size, self.num_layer)
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
    

class LatentODEfunc(nn.Module):
    def __init__(self, latent_dim=512, nhidden=4096, odenorm=None):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, latent_dim)
        self.odenorm = odenorm
        self.btnorm = nn.BatchNorm1d(nhidden)

    def forward(self, t, x):
        out = self.fc1(x)
        if self.odenorm is None:
            out = self.btnorm(out)
        else:
            out = self.odenorm(out)
        out = self.elu(out)
        out = self.fc2(out)
        return out
    

class LatentODEblock(nn.Module):
    def __init__(self, odefunc: nn.Module = LatentODEfunc, solver: str = 'dopri5',
                 latent_dim=512, nhidden=4096,
                 rtol: float = 1e-4, atol: float = 1e-4, adjoint: bool = False,
                 odenorm = None):
        super().__init__()
        self.odefunc = odefunc(latent_dim=latent_dim, nhidden=nhidden, odenorm=odenorm)
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
    

# main class

class BYOL(nn.Module):
    def __init__(
        self,
        net,
        clip_size,
        image_size,
        hidden_layer = -2,
        projection_size = 512,
        projection_hidden_size = 4096,
        moving_average_decay = 0.99,
        use_momentum = True,
        asym_loss = False,
        closed_loop = False,
        useode = False,
        adjoint = False,
        rtol = 1e-4,
        atol = 1e-4,
        odenorm = None,
        num_layer = 2,
        use_projector = True
    ):
        super().__init__()
        self.net = net

        self.online_encoder = NetWrapper(net, projection_size, projection_hidden_size, layer=hidden_layer, use_simsiam_mlp=not use_momentum, num_layer=num_layer, use_projector = use_projector)

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.useode = useode


        if not useode:
            self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)
        else:
            self.online_predictor = LatentODEblock(latent_dim = projection_size, nhidden = projection_hidden_size, 
                                               adjoint = adjoint, rtol = rtol, atol = atol,
                                               odenorm = odenorm)

        self.asym_loss = asym_loss
        self.closed_loop = closed_loop

        # get device of network and make wrapper same device
        device = get_module_device(net)
        self.to(device)

        # send a mock image tensor to instantiate singleton parameters
        self.forward(torch.randn(2, 3, clip_size, image_size, image_size, device=device), torch.randn(2, 3, clip_size, image_size, image_size, device=device))

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(
        self,
        x,
        x2,
        return_embedding = False,
        return_projection = True
    ):
        assert not (self.training and x.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        if return_embedding:
            return self.online_encoder(x, return_projection = return_projection)

        image_one, image_two = x, x2

        online_proj_one, _ = self.online_encoder(image_one)
        online_proj_two, _ = self.online_encoder(image_two)

        if not self.useode: # mlp
            online_pred_one = self.online_predictor(online_proj_one)
            online_pred_two = self.online_predictor(online_proj_two)

            if self.closed_loop:
                closed_pred_one = self.online_predictor(online_pred_one)
                closed_pred_two = self.online_predictor(online_pred_two)

            # print(image_one.size()) # [20, 3, 256, 256], [B, C, H, W]
            # print(online_proj_one.size()) # [2, 512], [B, projection_size]
            # print(online_pred_one.size()) # [2, 512], [B, projection_size]

        else: # ode
            online_pred_one = self.online_predictor(online_proj_one)[-1]
            online_pred_two = self.online_predictor(online_proj_two, integforward=False)[-1]

            if self.closed_loop:
                closed_pred_one = self.online_predictor(online_pred_one, integforward=False)[-1]
                closed_pred_two = self.online_predictor(online_pred_two)[-1]

        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            target_proj_two, _ = target_encoder(image_two)
            target_proj_two.detach_()
            target_proj_one, _ = target_encoder(image_one)
            target_proj_one.detach_()


        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        if self.closed_loop:
            loss_one += loss_fn(closed_pred_one, target_proj_one.detach())
            loss_one = loss_one / 2

        if not self.asym_loss:
            loss_two = loss_fn(online_pred_two, target_proj_one.detach())
            if self.closed_loop:
                loss_two += loss_fn(closed_pred_two, target_proj_two.detach())
                loss_two = loss_two / 2
            loss = loss_one + loss_two
        else:
            loss = loss_one * 2
            
        return loss.mean()
