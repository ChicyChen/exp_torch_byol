import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F

from torchvision import transforms as T

from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint

import numpy as np

from helpers import *
    
# main class

class BYOL_ODE(nn.Module):
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
        adjoint = False,
        rtol = 1e-4,
        atol = 1e-4,
        num_layer = 2,
        solver = 'dopri5',
        mse_l = 1,
        std_l = 1,
        cov_l = 0.1,
        use_projector = True,
        use_simsiam_mlp = False,
        bn_last = False,
        pred_bn_last = False
    ):
        super().__init__()
        self.net = net

        self.online_encoder = NetWrapper(net, projection_size, projection_hidden_size, layer=hidden_layer, use_simsiam_mlp=use_simsiam_mlp, num_layer=num_layer, bn_last = bn_last, use_projector = use_projector)

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)
        self.use_projector = use_projector 

        self.online_predictor = LatentODEblock(projection_size = projection_size, hidden_size = projection_hidden_size, num_layer=num_layer,
                                               adjoint = adjoint, rtol = rtol, atol = atol, 
                                               solver = solver, bn_last = pred_bn_last)

        self.asym_loss = asym_loss
        self.closed_loop = closed_loop

        self.mse_l = mse_l
        self.std_l = std_l
        self.cov_l = cov_l

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
        return vic_reg_loss(x, y, self.mse_l, self.std_l, self.cov_l)

    def forward(
        self,
        x, # start time's input
        x2, # predicted time's input
        return_embedding = False,
        return_projection = True,
        integration_time_f = None,
        integration_time_b = None,
        sequential = False
    ):
        assert not (self.training and x.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        if return_embedding:
            return self.online_encoder(x, return_projection = return_projection)

        if integration_time_f is not None:
            integration_time_f = torch.tensor(integration_time_f, dtype=torch.float32)
            integration_time_b = torch.tensor(integration_time_b, dtype=torch.float32)
        
        if not sequential:
            image_one, image_two = x, x2

            if self.use_projector:
                online_proj_one, _ = self.online_encoder(image_one)
            else:
                online_proj_one = self.online_encoder(image_one)
            # print(online_proj_one.size())
            online_pred_one = self.online_predictor(online_proj_one, integration_time=integration_time_f)[-1]
            # online_pred_one = self.online_predictor(online_proj_one, integration_time=integration_time_f)
            # print(online_pred_one.size())
            # online_pred_one = online_pred_one[-1]
            if self.closed_loop:
                closed_pred_one = self.online_predictor(online_pred_one, integforward=False, integration_time=integration_time_b)[-1]

            if not self.asym_loss:
                if self.use_projector:
                    online_proj_two, _ = self.online_encoder(image_two)
                else:
                    online_proj_two = self.online_encoder(image_two)
                online_pred_two = self.online_predictor(online_proj_two, integforward=False, integration_time=integration_time_b)[-1]
                if self.closed_loop:
                    closed_pred_two = self.online_predictor(online_pred_two, integration_time=integration_time_f)[-1]

            # print(image_one.size()) # [20, 3, 256, 256], [B, C, H, W]
            # print(online_proj_one.size()) # [2, 512], [B, projection_size]
            # print(online_pred_one.size()) # [2, 512], [B, projection_size]

            with torch.no_grad():
                target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
                if self.use_projector:
                    target_proj_two, _ = target_encoder(image_two)
                else:
                    target_proj_two = target_encoder(image_two)
                # target_proj_two.detach_() # in place, detach the tensor from the graph
                target_proj_two.detach()
                if self.use_projector:
                    target_proj_one, _ = target_encoder(image_one)
                else:
                    target_proj_one = target_encoder(image_one)
                # target_proj_one.detach_()
                target_proj_one.detach()

            # print(online_pred_one.size(), target_proj_two.size())
            # print(closed_pred_one.size(), target_proj_one.size())

            loss_one = self.loss_fn(online_pred_one, target_proj_two.detach())
            if self.closed_loop:
                loss_one += self.loss_fn(closed_pred_one, target_proj_one.detach())
                loss_one = loss_one / 2

            if not self.asym_loss: # sym loss, two way prediction
                loss_two = self.loss_fn(online_pred_two, target_proj_one.detach())
                if self.closed_loop:
                    loss_two += self.loss_fn(closed_pred_two, target_proj_two.detach())
                    loss_two = loss_two / 2
                loss = loss_one + loss_two
            else:
                loss = loss_one * 2

        else:
            image_one = x
            online_proj_one, _ = self.online_encoder(image_one)
            # print(integration_time_f)
            online_pred_next_list = self.online_predictor(online_proj_one, integration_time=integration_time_f)[1:]
            # print(online_proj_one.size(), online_pred_next_list.size())
            target_proj_next_list = []
            with torch.no_grad():
                target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
                for image_two in x2:
                    if self.use_projector:
                        target_proj_two, _ = target_encoder(image_two)
                    else:
                        target_proj_two = target_encoder(image_two)
                    target_proj_next_list.append(target_proj_two)
                # print(len(target_proj_next_list))
                # target_proj_first = target_encoder(image_one)

            if not self.asym_loss:
                target_proj_prev_list = target_proj_next_list[::-1][1:]
                with torch.no_grad():
                    target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
                    if self.use_projector:
                        target_proj_first, _ = target_encoder(image_one)
                    else:
                        target_proj_first = target_encoder(image_one)
                    target_proj_prev_list.append(target_proj_first)
                image_last = x2[-1]
                online_proj_last, _ = self.online_encoder(image_last)
                online_pred_prev_list = self.online_predictor(online_proj_last, integforward=False, integration_time=integration_time_b)[1:]
                # print(target_proj_prev_list[0].size())

            target_proj_next_list = torch.stack(target_proj_next_list, 0)
            # target_proj_next_list.detach_()
            target_proj_next_list.detach()

            # print(target_proj_prev_list[2])
            target_proj_prev_list = torch.stack(target_proj_prev_list, 0)
            # target_proj_prev_list.detach_()
            target_proj_prev_list.detac()

            loss_one = self.loss_fn(online_pred_next_list, target_proj_next_list.detach()) # forward prediction
            if self.closed_loop:
                online_pred_next_prev_list = self.online_predictor(online_pred_next_list[-1], integforward=False, integration_time=integration_time_b)[1:]
                loss_one += self.loss_fn(online_pred_next_prev_list, target_proj_prev_list.detach())

            if not self.asym_loss:
                loss_two = self.loss_fn(online_pred_prev_list, target_proj_prev_list.detach()) # backward prediction
                if self.closed_loop:
                    online_pred_prev_next_list = self.online_predictor(online_pred_prev_list[-1], integration_time=integration_time_f)[1:]
                    loss_two += self.loss_fn(online_pred_prev_next_list, target_proj_next_list.detach())

                loss = loss_one + loss_two
            else:
                loss = loss_one * 2
            
        return loss.mean()
