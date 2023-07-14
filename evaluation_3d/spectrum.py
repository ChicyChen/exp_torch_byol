# modified from https://github.com/facebookresearch/directclr/blob/main/spectrum.py

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import numpy as np
import matplotlib.pyplot as plt


class Spectrum():
    def __init__(self, model, device):
        super(Spectrum, self).__init__()
        self.device = device
        self.model = model.to(device)
        self.model.eval()

    def extract_data(self, loader):
        x_lst = []
        # features = []
        label_lst = []

        with torch.no_grad():
            for data_i in loader:
                input_tensor, label = data_i
                # h = self.model.get_representation(input_tensor.to(self.device))
                # features.append(h)
                x_lst.append(input_tensor)
                label_lst.append(label)

            x_total = torch.stack(x_lst)
            # h_total = torch.stack(features)
            label_total = torch.stack(label_lst)

        x_total = x_total[:,:,0,0,:,:] # N, B, H, W
        print(x_total.shape)
        (N, B, H, W) = x_total.shape
        x_total = x_total.view(N*B, H*W)
            # return x_total, h_total, label_total
        return x_total, label_total

    def extract_projection_feature(self, loader):
        projections = []
        features = []
        label_lst = []

        with torch.no_grad():
            for data_i in loader:
                
                input_tensor, label = data_i
                h, z = self.model(input_tensor.to(self.device))
                projections.append(h)
                features.append(z)
                label_lst.append(label)

            h_total = torch.stack(projections)
            z_total = torch.stack(features)
            label_total = torch.stack(label_lst)
            # print(h_total.shape)
            # print(z_total.shape)

        proj_dim = h_total.shape[-1]
        feature_dim = z_total.shape[-1]
        h_total = h_total.view(-1, proj_dim)
        z_total = z_total.view(-1, feature_dim)

        return h_total, z_total, label_total
    

    def singular_input(self, loader):
        x_total, _ = self.extract_data(loader)

        h = torch.nn.functional.normalize(x_total, dim=1)

        # calculate covariance of projection
        h = h.cpu().detach().numpy()
        h = np.transpose(h)
        ch = np.cov(h)
        _, dx, _ = np.linalg.svd(ch)

        return dx[:512]
    

    def singular(self, loader):
        h_total, z_total, _ = self.extract_projection_feature(loader)

        h = torch.nn.functional.normalize(h_total, dim=1)
        z = torch.nn.functional.normalize(z_total, dim=1)

        # calculate covariance of projection
        h = h.cpu().detach().numpy()
        h = np.transpose(h)
        ch = np.cov(h)
        _, dh, _ = np.linalg.svd(ch)

        # calculate covariance of representation
        z = z.cpu().detach().numpy()
        z = np.transpose(z)
        cz = np.cov(z)
        _, dz, _ = np.linalg.svd(cz)

        return dh, dz
    
    def visualize(self, loader, fpath, epoch_num, mode='train', log=False):
        dh, dz = self.singular(loader)
        len_dh = len(dh)
        len_dz = len(dz)

        if log:
            dh = np.log(dh)
            dz = np.log(dz)

        # plot training process
        plt.figure()
        plt.plot(range(len_dh), dh.tolist(), label = 'projection')
        # if not args.no_val:
        plt.plot(range(len_dz), dz.tolist(), label = 'feature')
        plt.title('Singular values')

        plt.legend()
        plt.savefig(os.path.join(fpath, '%s_singular_log%s_epoch%s.png' % (mode, log, epoch_num)))


    def visualize_input(self, loader, fpath, epoch_num, mode='train'):
        dx = self.singular_input(loader)
        len_dx = len(dx)

        # plot training process
        plt.figure()
        plt.plot(range(len_dx), dx.tolist(), label = 'input')
        plt.title('Singular values')

        plt.legend()
        plt.savefig(os.path.join(fpath, '%s_input_singular_epoch%s.png' % (mode, epoch_num)))



