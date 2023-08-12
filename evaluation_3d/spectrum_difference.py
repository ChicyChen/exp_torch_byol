import numpy as np
import torch
import torch.nn as nn
import os

import numpy as np
import matplotlib.pyplot as plt

class Spectrum_Difference():
    def __init__(self, model, device, input_num=500, return_dim=2048):
        super(Spectrum_Difference, self).__init__()
        self.device = device
        self.input_num = input_num
        self.model = model.to(device)
        self.model.eval()
        self.return_dim = return_dim

    def extract_projection_feature(self, loader):
        projections = []
        features = []
        label_lst = []

        with torch.no_grad():
            ni = 0
            for data_i in loader:
                
                input_tensor, label = data_i # B, N, C, T, H, W
                B, N, C, T, H, W = input_tensor.shape
                input_tensor = input_tensor.view(B*N, C, T, H, W)
                h, z = self.model(input_tensor.to(self.device))
                h = h.reshape(B, N, -1)
                z = z.reshape(B, N, -1)
                projections.append(h)
                features.append(z)
                label_lst.append(label)
                ni += 1
                if ni >= self.input_num:
                    break

            h_total = torch.stack(projections) # Bn, B, N, -1
            z_total = torch.stack(features) # Bn, B, N, -1
            label_total = torch.stack(label_lst).view(-1)
            # print(h_total.shape)
            # print(z_total.shape)

        proj_dim = h_total.shape[-1]
        feature_dim = z_total.shape[-1]
        h_total = h_total.view(-1, N, proj_dim)
        z_total = z_total.view(-1, N, feature_dim)

        return h_total, z_total, label_total

    # def extract_diff_rank(self, loader):
    #     h_total, z_total, _ = self.extract_projection_feature(loader)
    #     h_diff = h_total[:,1:,:] - h_total[:,:-1,:] # n, N-1, -1
    #     z_diff = z_total[:,1:,:] - z_total[:,:-1,:] # n, N-1, -1
    #     rank_h_diff = torch.linalg.matrix_rank(h_diff) # n
    #     rank_z_diff = torch.linalg.matrix_rank(z_diff) # n

    #     return np.mean(rank_h_diff.cpu().detach().numpy()), np.mean(rank_z_diff.cpu().detach().numpy())

    def singular(self, loader, diff=True):
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm


        h_total, z_total, labels = self.extract_projection_feature(loader)

        if not diff:
            h_diff = h_total
            z_diff = z_total
        else:
            h_diff = h_total[:,1:,:] - h_total[:,:-1,:] # n, N-1, -1
            z_diff = z_total[:,1:,:] - z_total[:,:-1,:] # n, N-1, -1

        a, b, d = h_diff.shape
        h_diff = h_diff.view(a*b, d)
        z_diff = z_diff.view(a*b, -1)

        h = h_diff - torch.mean(h_diff, dim=1, keepdim=True)
        z = z_diff - torch.mean(z_diff, dim=1, keepdim=True)

        # calculate covariance of projection
        h = h.cpu().detach().numpy()
        h = np.transpose(h) # D, N
        ch = np.cov(h)
        # print(h.shape, ch.shape) # D, N; D, D
        _, dh, _ = np.linalg.svd(ch)

        # calculate covariance of representation
        z = z.cpu().detach().numpy()
        z = np.transpose(z)
        cz = np.cov(z)
        _, dz, _ = np.linalg.svd(cz)

        return_dim = self.return_dim

        return dh[:return_dim], dz[:return_dim]
    
    def visualize(self, loader, fpath, epoch_num=100, mode='train', log=True):
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
        plt.savefig(os.path.join(fpath, '%s_diff_singular_log%s_epoch%s.png' % (mode, log, epoch_num)))

    # def spectrum(self, loader, return_dim, fname='figure.png'):
    #     h_total, z_total, _ = self.extract_projection_feature(loader)
    #     h_diff = h_total[:,1:,:] - h_total[:,:-1,:] # n, N-1, -1
    #     z_diff = z_total[:,1:,:] - z_total[:,:-1,:] # n, N-1, -1
    #     a, b, d = h_diff.shape
    #     h_diff = h_diff.view(a*b, d)
    #     z_diff = z_diff.view(a*b, -1)


