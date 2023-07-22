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

class PCNET_VIC(nn.Module):
    def __init__(
        self,
        net,
        clip_size,
        image_size,
        hidden_layer = -2,
        projection_size = 512,
        projection_hidden_size = 4096,
        pred_layer = 2,
        proj_layer = 2,
        sym_loss = False,
        closed_loop = False,
        mse_l = 25.0,
        std_l = 25.0,
        cov_l = 1.0,
        bn_last = False,
        pred_bn_last = False,
        # pred_mode = 0, # 0: z1->z2', z2->z3'; 1: z1->z2'->z3'
        predictor = 1, # 0: MLP; 1: Res_MLP; 2: Res_MLP_D; 3: ODE-not supported
        relu_last = False,
        num_predictor = 2,
        infonce = False
    ):
        super().__init__()
        self.net = net

        self.encoder = NetWrapper(net, projection_size, projection_hidden_size, layer = hidden_layer, num_layer = proj_layer, bn_last = bn_last)

        if predictor == 0:
            create_pred = MLP 
        elif predictor == 1:
            create_pred = Res_MLP
        elif predictor == 2:
            create_pred = Res_MLP_D # with direction
        else:
            create_pred = LatentODEblock

        if pred_layer > 0:
            self.predictor_forward = create_pred(projection_size, projection_size, projection_hidden_size, pred_layer, pred_bn_last, relu_last) 
            if num_predictor == 2:
                self.predictor_reverse = create_pred(projection_size, projection_size, projection_hidden_size, pred_layer, pred_bn_last, relu_last) 
            else:
                self.predictor_reverse = self.predictor_forward
        else:
            self.predictor_forward = nn.Identity()
            self.predictor_reverse = self.predictor_forward

        self.sym_loss = sym_loss
        self.closed_loop = closed_loop

        self.mse_l = mse_l
        self.std_l = std_l
        self.cov_l = cov_l

        self.infonce = infonce

        # self.pred_mode = pred_mode

        # get device of network and make wrapper same device
        device = get_module_device(net)
        self.to(device)

        # while using singleton, if we don't do an initial pass, the memory would explode
        # send a mock image tensor to instantiate singleton parameters
        self.forward(torch.randn(2, 2, 3, clip_size, image_size, image_size, device=device))


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
            loss = vic_reg_nonorm_loss(x, y, self.mse_l, self.std_l, self.cov_l)
        # TODO: if M > 1, how to adapt
        return loss


    def forward(
        self,
        x # B, N, C, T, H, W
    ):
        assert not (self.training and x.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'
        
        B, N, C, T, H, W = x.size()

        # ground truth latents
        gt_z_all, _ = self.encoder(x.view(B*N, C, T, H, W)) # encoder return (projection, representation)
        gt_z_all = gt_z_all.reshape(B, N, -1) # B, N, D

        # predicted latents
        # TODO: implement mode 1
        # mode 0
        pred_z_forward = self.predictor_forward(gt_z_all[:, :-1, :].reshape(B*(N-1),-1)).reshape(B, N-1, -1)
        loss_one = self.loss_fn(pred_z_forward, gt_z_all[:, 1:, :]) 
        if self.closed_loop:
            closed_z_reverse = self.predictor_reverse(pred_z_forward.reshape(B*(N-1), -1), forward = False).reshape(B, N-1, -1)
            loss_one += self.loss_fn(closed_z_reverse, gt_z_all[:, :-1, :])
            loss_one = loss_one / 2

        if self.sym_loss:
            pred_z_reverse = self.predictor_reverse(gt_z_all[:, 1:, :].reshape(B*(N-1),-1), forward = False).reshape(B, N-1, -1)
            loss_two = self.loss_fn(pred_z_reverse, gt_z_all[:, :-1, :])
            if self.closed_loop:
                closed_z_forward = self.predictor_forward(pred_z_reverse.reshape(B*(N-1), -1)).reshape(B, N-1, -1)
                loss_two += self.loss_fn(closed_z_forward, gt_z_all[:, 1:, :])
                loss_two = loss_two / 2
            loss = loss_one + loss_two
        else:
            loss = loss_one * 2


            
        return loss.mean()