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

import torch.distributed as dist

from helpers import *

    
# main class

class PCNET(nn.Module):
    def __init__(
        self,
        net,
        # hidden_layer = -2,
        feature_size = 2048,
        projection_size = 8192,
        projection_hidden_size = 8192,
        pred_hidden_size = 2048,
        pred_layer = 2,
        proj_layer = 2,
        sym_loss = False,
        closed_loop = False,
        mse_l = 1.0,
        loop_l = 1.0,
        std_l = 1.0,
        cov_l = 0.04,
        bn_last = False,
        pred_bn_last = False,
        # pred_mode = 0, # 0: z1->z2', z2->z3'; 1: z1->z2'->z3'
        predictor = 1, # 0: MLP; 1: Res_MLP; 2: Res_MLP_D; 3: ODE-not supported
        relu_last = False,
        num_predictor = 2,
        infonce = False,
        sub_loss = False,
        sub_frac = 0.2
    ):
        super().__init__()

        self.encoder = net

        if proj_layer > 0:
            create_mlp_fn = MLP
            self.projector = create_mlp_fn(feature_size, projection_size, projection_hidden_size, proj_layer, bn_last)
        else:
            self.projector = nn.Identity()

        self.pred_layer = pred_layer
        if predictor == 0:
            create_pred = MLP 
        elif predictor == 1:
            create_pred = Res_MLP
        elif predictor == 2:
            create_pred = Res_MLP_D # with direction
        else:
            create_pred = LatentODEblock

        if pred_layer > 0:
            self.predictor_forward = create_pred(projection_size, projection_size, pred_hidden_size, pred_layer, pred_bn_last, relu_last) 
            if num_predictor == 2:
                self.predictor_reverse = create_pred(projection_size, projection_size, pred_hidden_size, pred_layer, pred_bn_last, relu_last) 
            else:
                self.predictor_reverse = self.predictor_forward
        else:
            self.predictor_forward = nn.Identity()
            self.predictor_reverse = self.predictor_forward

        self.sym_loss = sym_loss
        self.closed_loop = closed_loop

        self.mse_l = mse_l
        self.loop_l = loop_l
        self.std_l = std_l
        self.cov_l = cov_l

        self.infonce = infonce
        self.sub_loss = sub_loss
        self.sub_frac = sub_frac

        # # self.pred_mode = pred_mode
        # self.hidden = {}
        # # self.hidden_layer = hidden_layer
        # self._register_hook()

        # device = get_module_device(net)
        # self.to(device)
        # self.forward(torch.randn(2, 2, 3, 3, 224, 224, device=device))

    # def _find_layer(self):
    #     if type(self.hidden_layer) == str:
    #         modules = dict([*self.encoder.named_modules()])
    #         return modules.get(self.hidden_layer, None)
    #     elif type(self.hidden_layer) == int:
    #         children = [*self.encoder.children()]
    #         return children[self.hidden_layer]
    #     return None

    # def _hook(self, _, input, output):
    #     device = input[0].device
    #     self.hidden[device] = flatten(output)

    # def _register_hook(self):
    #     layer = self._find_layer()
    #     assert layer is not None, f'hidden layer ({self.hidden_layer}) not found'
    #     handle = layer.register_forward_hook(self._hook)

    def loss_fn(self, x, y):
        # x: B, N-1, D
        # y: B, N-1, D
        (B, M, D) = x.shape
        x = x.reshape(B*M, D)
        y = y.reshape(B*M, D)
        # TODO: other losses
        if self.infonce:
            loss = infoNCE(x, y)
        else:
            loss = vic_reg_nonorm_loss(x, y, self.mse_l, self.std_l, self.cov_l, self.sub_loss)
        # TODO: if M > 1, how to adapt
        return loss

    def forward(
        self,
        x # B, N, C, T, H, W
    ):
        assert not (self.training and x.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'
        
        B, N, C, T, H, W = x.size()

        # ground truth latents
        hidden = flatten(self.encoder(x.view(B*N, C, T, H, W))) # encoder forward
        # _ = self.encoder(x.view(B*N, C, T, H, W)) # encoder forward
        # print(self.hidden[x.device])
        # hidden = self.hidden.pop(x.device) # get feature
        # print(hidden.shape)
        gt_z_all = self.projector(hidden) # projector forward
        gt_z_all = gt_z_all.reshape(B, N, -1) # B, N, D

        # if no predictor, VICReg or SimCLR
        if self.pred_layer == 0:
            loss_one = self.loss_fn(gt_z_all[:, :-1, :], gt_z_all[:, 1:, :]) 
            if self.sym_loss:
                loss_two = self.loss_fn(gt_z_all[:, 1:, :], gt_z_all[:, :-1, :])
                loss = loss_one + loss_two
            else:
                loss = loss_one * 2
            return loss

        # predicted latents
        # TODO: implement mode 1
        # mode 0
        pred_z_forward = self.predictor_forward(gt_z_all[:, :-1, :].reshape(B*(N-1),-1)).reshape(B, N-1, -1)
        pred_loss = mse_loss(pred_z_forward, gt_z_all[:, 1:, :]) 

        loop_loss = 0
        if self.closed_loop:
            closed_z_reverse = self.predictor_reverse(pred_z_forward.reshape(B*(N-1), -1), forward = False).reshape(B, N-1, -1)
            loop_loss += mse_loss(closed_z_reverse, gt_z_all[:, :-1, :])

        if self.sym_loss:
            pred_z_reverse = self.predictor_reverse(gt_z_all[:, 1:, :].reshape(B*(N-1),-1), forward = False).reshape(B, N-1, -1)
            pred_loss += mse_loss(pred_z_reverse, gt_z_all[:, :-1, :])
            pred_loss = pred_loss / 2
            if self.closed_loop:
                closed_z_forward = self.predictor_forward(pred_z_reverse.reshape(B*(N-1), -1)).reshape(B, N-1, -1)
                loop_loss += mse_loss(closed_z_forward, gt_z_all[:, 1:, :])
                loop_loss = loop_loss / 2

        gt_z_all = torch.cat(FullGatherLayer.apply(gt_z_all.reshape(B*N, -1)), dim=0)
        if self.sub_loss:
            D = gt_z_all.shape[1]
            gt_z_all = gt_z_all[:,torch.randperm(D)]
            D = int(D * self.sub_frac)
            gt_z_all = gt_z_all[:,:D]
        std_lo = std_loss(gt_z_all)
        cov_lo = cov_loss(gt_z_all)
        loss = pred_loss * self.mse_l + loop_loss * self.loop_l + std_lo*self.std_l + cov_lo*self.cov_l
            
        return loss


class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]
