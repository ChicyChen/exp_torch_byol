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
from operator import add 

from helpers import *

    
# main class

class BYOL_MLP_2P(nn.Module):
    def __init__(
        self,
        net,
        clip_size,
        image_size,
        hidden_layer = -2,
        projection_size = 512,
        projection_hidden_size = 4096,
        num_layer = 2,
        moving_average_decay = 0.99,
        use_momentum = True,
        asym_loss = False,
        closed_loop = False,
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

        self.online_encoder = NetWrapper(net, projection_size, projection_hidden_size, layer = hidden_layer, use_simsiam_mlp = use_simsiam_mlp, num_layer = num_layer, bn_last = bn_last, use_projector = use_projector)

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size, pred_bn_last) # predict dfference instead
        self.online_predictor_reverse = MLP(projection_size, projection_size, projection_hidden_size, pred_bn_last) # predict dfference instead

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
        sequential = False
    ):
        # TODO: sym & closed loop can use two predictors

        assert not (self.training and x.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        if return_embedding:
            return self.online_encoder(x, return_projection = return_projection)
        
        if not sequential:
            image_one, image_two = x, x2

            online_proj_one, _ = self.online_encoder(image_one)
            online_pred_one = self.online_predictor(online_proj_one) + online_proj_one
            
            if self.closed_loop:
                closed_pred_one = online_pred_one + self.online_predictor_reverse(online_pred_one) # reverse pred

            if not self.asym_loss: # sym loss, two way prediction
                online_proj_two, _ = self.online_encoder(image_two)
                online_pred_two = online_proj_two + self.online_predictor_reverse(online_proj_two) # reverse pred
                if self.closed_loop:
                    closed_pred_two = self.online_predictor(online_pred_two) + online_proj_two

            # print(image_one.size()) # [20, 3, 256, 256], [B, C, H, W]
            # print(online_proj_one.size()) # [2, 512], [B, projection_size]
            # print(online_pred_one.size()) # [2, 512], [B, projection_size]

            with torch.no_grad():
                target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
                target_proj_two, _ = target_encoder(image_two)
                target_proj_two.detach_()
                target_proj_one, _ = target_encoder(image_one)
                target_proj_one.detach_()

            # print(online_pred_one.size(), target_proj_two.size())
            # print(closed_pred_one.size(), target_proj_one.size())

            loss_one = self.loss_fn(online_pred_one, target_proj_two)

            if self.closed_loop:
                loss_one = self.loss_fn(closed_pred_one, target_proj_one)
                loss_one = loss_one / 2

            if not self.asym_loss: # sym loss, two way prediction
                loss_two = self.loss_fn(online_pred_two, target_proj_one)
                if self.closed_loop:
                    loss_two += self.loss_fn(closed_pred_two, target_proj_two)
                    loss_two = loss_two / 2
                loss = loss_one + loss_two
            else:
                loss = loss_one * 2

        else:
            B, N, C, T, H, W = x.size()
            x = x.transpose(0,1) # N, B, C, T, H, W

            image_one = x[0]
            image_two = x[-1]
            # print(image_one.size()) # B, C, T, H, W

            ### calculate predictions
            online_proj_one, _ = self.online_encoder(image_one)
            online_pred_next_list = []
            for i in range(N-1): # first -> last, using actual first
                if i == 0:
                    online_pred_next = self.online_predictor(online_proj_one) + online_proj_one
                else:
                    online_pred_next = self.online_predictor(online_pred_next_list[i]) + online_pred_next_list[i]
                online_pred_next_list.append(online_pred_next)
            online_pred_next_list = torch.stack(online_pred_next_list, 0) # N-1, B, D
            # print(online_pred_next_list.size()) 
            if self.closed_loop:
                online_proj_two_pred = online_pred_next_list[-1] # predicted last latent
                online_pred_backprev_list = [] 
                for i in range(N-1): # last -> first, using predicted last
                    if i == 0:
                        online_pred_backprev = online_proj_two_pred + self.online_predictor_reverse(online_proj_two_pred) # minus, if using one predictor
                    else:
                        online_pred_backprev = online_pred_backprev_list[i] + self.online_predictor_reverse(online_pred_backprev_list[i]) # minus, if using one predictor
                    online_pred_backprev_list.append(online_pred_backprev)
                online_pred_backprev_list.reverse()
                online_pred_backprev_list = torch.stack(online_pred_backprev_list, 0) # N-1, B, D, need to reverse the list
                # print(online_pred_backprev_list.size())

            if not self.asym_loss: # sym loss, two way prediction, need to start from last as well
                online_proj_two, _ = self.online_encoder(image_two)
                online_pred_prev_list = []
                for i in range(N-1): # last -> first, using actual last
                    if i == 0:
                        online_pred_prev = online_proj_two + self.online_predictor_reverse(online_proj_two) # minus, if using one predictor
                    else:
                        online_pred_prev = online_pred_prev_list[i] + self.online_predictor_reverse(online_pred_prev_list[i]) # minus, if using one predictor
                    online_pred_prev_list.append(online_pred_prev)
                online_pred_prev_list.reverse()
                online_pred_prev_list = torch.stack(online_pred_prev_list, 0) # N-1, B, D, need to reverse the list
                # print(online_pred_prev_list.size())
                if self.closed_loop:
                    online_proj_one_pred = online_pred_prev_list[0] # have reversed the list, so index 0
                    online_pred_backnext_list = []
                    for i in range(N-1):  # first -> last, using predicted first
                        if i == 0:
                            online_pred_backnext = online_proj_one_pred + self.online_predictor(online_proj_one_pred)
                        else:
                            online_pred_backnext = online_pred_backnext_list[i] + self.online_predictor(online_pred_backnext_list[i])
                        online_pred_backnext_list.append(online_pred_backnext)
                    online_pred_backnext_list = torch.stack(online_pred_backnext_list, 0)
            ### calculate ground truth
            with torch.no_grad():
                target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
                target_proj_list, _ = target_encoder(x.reshape(N*B, C, T, H, W)) 
            target_proj_list.detach_()
            target_proj_list = target_proj_list.view(N, B, -1) # detach_() inplace, does not require gradient
            # print(target_proj_list.size()) # N, B, D
            ### calculate loss
            loss_one = self.loss_fn(online_pred_next_list.view((N-1)*B, -1), target_proj_list[1:].view((N-1)*B, -1)) 
            if self.closed_loop:
                loss_one += self.loss_fn(online_pred_backprev_list.view((N-1)*B, -1), target_proj_list[:-1].view((N-1)*B, -1))
                loss_one = loss_one / 2

            if not self.asym_loss: # sym loss, two way prediction
                loss_two = self.loss_fn(online_pred_prev_list.view((N-1)*B, -1), target_proj_list[:-1].view((N-1)*B, -1))
                if self.closed_loop:
                    loss_two += self.loss_fn(online_pred_backnext_list.view((N-1)*B, -1), target_proj_list[1:].view((N-1)*B, -1))
                    loss_two = loss_two / 2
                loss = loss_one + loss_two
            else:
                loss = loss_one * 2
            
        return loss.mean()
