import numpy as np
import torch
import torch.nn as nn
import os


class Difference():
    def __init__(self, model, device, input_num=500):
        super(Difference, self).__init__()
        self.device = device
        self.input_num = input_num
        self.model = model.to(device)
        self.model.eval()

    def extract_projection_feature(self, loader):
        projections = []
        features = []
        label_lst = []

        with torch.no_grad():
            ni = 0
            for data_i in loader:
                
                input_tensor, label = data_i # B, N, C, T, H, W
                if label == 0 or label == 50 or label == 100:
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

    def extract_diff_rank(self, loader):
        h_total, z_total, _ = self.extract_projection_feature(loader)
        h_diff = h_total[:,1:,:] - h_total[:,:-1,:] # n, N-1, -1
        z_diff = z_total[:,1:,:] - z_total[:,:-1,:] # n, N-1, -1
        rank_h_diff = torch.linalg.matrix_rank(h_diff) # n
        rank_z_diff = torch.linalg.matrix_rank(z_diff) # n

        return np.mean(rank_h_diff.cpu().detach().numpy()), np.mean(rank_z_diff.cpu().detach().numpy())

    def visualize_class(self, loader, fpath, num_class=101, train=True, diff=True):
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        if train:
            addn = 'train'
        else:
            addn = 'test'

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

        # h_mask = h_diff[:, 0] > 0
        # h_mask = h_mask.unsqueeze(1)
        # h_diff = 2 * h_diff * h_mask - h_diff # convert direction
        # print(h_diff.shape)
        # h_diff = h_diff / torch.norm(h_diff, dim=0, keepdim=True)

        # z_mask = z_diff[:, 0] > 0
        # z_mask = z_mask.unsqueeze(1)
        # z_diff = 2 * z_diff * z_mask - z_diff # convert direction
        # print(z_diff.shape)
        # z_diff = z_diff / torch.norm(z_diff, dim=0, keepdim=True)

        colors = cm.rainbow(np.linspace(0, 1, num_class))

        plt.figure()
        U,S,V = torch.pca_lowrank(h_diff, q=10, center=True, niter=3) # V: *, N
        print(S)
        h_diff_pca = torch.matmul(h_diff, V[:, :2]).cpu().numpy() # len(x)*num_seq, 2
        h_diff_pca = h_diff_pca.reshape(a, b, 2)

        for i in range(a):
            c = colors[labels[i]]
            diff_seq_i = h_diff_pca[i] # b, 2
            plt.plot(diff_seq_i[:,0], diff_seq_i[:,1], color=c)
            for j in range(b):
                plt.scatter(diff_seq_i[j,0], diff_seq_i[j,1], color=c)

        ax = plt.gca()
        ax.set_aspect('equal')
        plt.savefig(os.path.join(fpath, addn + "_projection_mdiff%s.png" % diff))


        plt.figure()
        U,S,V = torch.pca_lowrank(z_diff, q=10, center=True, niter=3) # V: *, N
        print(S)
        z_diff_pca = torch.matmul(z_diff, V[:, :2]).cpu().numpy() # len(x)*num_seq, 2
        z_diff_pca = z_diff_pca.reshape(a, b, 2)

        for i in range(a):
            c = colors[labels[i]]
            diff_seq_i = z_diff_pca[i] # b, 2
            plt.plot(diff_seq_i[:,0], diff_seq_i[:,1], color=c)
            for j in range(b):
                plt.scatter(diff_seq_i[j,0], diff_seq_i[j,1], color=c)

        ax = plt.gca()
        ax.set_aspect('equal')
        plt.savefig(os.path.join(fpath, addn + "_feature_mdiff%s.png" % diff))

        return

    # def spectrum(self, loader, return_dim, fname='figure.png'):
    #     h_total, z_total, _ = self.extract_projection_feature(loader)
    #     h_diff = h_total[:,1:,:] - h_total[:,:-1,:] # n, N-1, -1
    #     z_diff = z_total[:,1:,:] - z_total[:,:-1,:] # n, N-1, -1
    #     a, b, d = h_diff.shape
    #     h_diff = h_diff.view(a*b, d)
    #     z_diff = z_diff.view(a*b, -1)


