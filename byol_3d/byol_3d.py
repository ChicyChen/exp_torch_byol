import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F

from torchvision import transforms as T

from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint

from helpers import *
    

# main class, as base

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
        asym_loss = False, # BYOL uses sym_loss by default, which causes PC models to have feature collapse
        closed_loop = False,
        useode = False,
        adjoint = False,
        rtol = 1e-4,
        atol = 1e-4,
        num_layer = 2,
        use_simsiam_mlp = False,
    ):
        super().__init__()
        self.net = net

        self.online_encoder = NetWrapper(net, projection_size, projection_hidden_size, layer = hidden_layer, use_simsiam_mlp = use_simsiam_mlp, num_layer = num_layer)

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.useode = useode


        if not useode:
            self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)
        else:
            self.online_predictor = LatentODEblock(projection_size = projection_size, hidden_size = projection_hidden_size, num_layer=num_layer,
                                               adjoint = adjoint, rtol = rtol, atol = atol)

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

    def loss_fn(self, x, y):
        return normalized_mse_loss(x, y)

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


        loss_one = self.loss_fn(online_pred_one, target_proj_two)
        if self.closed_loop:
            loss_one += self.loss_fn(closed_pred_one, target_proj_one)
            loss_one = loss_one / 2

        if not self.asym_loss: # by default sym loss
            loss_two = self.loss_fn(online_pred_two, target_proj_one)
            if self.closed_loop:
                loss_two += self.loss_fn(closed_pred_two, target_proj_two)
                loss_two = loss_two / 2
            loss = loss_one + loss_two
        else:
            loss = loss_one * 2
            
        return loss.mean()
