import numpy as np
import torch
import torch.nn as nn
import os

from torch.optim.optimizer import Optimizer, required
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

import torch.nn.functional as F


class KNN_Difference():
    def __init__(self, model, device, k=1, input_num=500, only_diff=False):
        super(KNN_Difference, self).__init__()
        self.k = k
        self.only_diff = only_diff
        self.device = device
        self.input_num = input_num
        self.model = model.to(device)
        self.model.eval()
        

    def extract_knn_features(self, loader):
        # projections = []
        features = []
        label_lst = []

        with torch.no_grad():
            ni = 0
            for data_i in loader:
                
                input_tensor, label = data_i # B, N, C, T, H, W
                B, N, C, T, H, W = input_tensor.shape
                input_tensor = input_tensor.view(B*N, C, T, H, W)
                _, z = self.model(input_tensor.to(self.device))
                # h = h.reshape(B, N, -1)
                z = z.reshape(B, N, -1)
                # projections.append(h)
                features.append(z)
                label_lst.append(label)
                ni += 1
                if ni >= self.input_num:
                    break

            # h_total = torch.stack(projections) # Bn, B, N, -1
            z_total = torch.stack(features) # Bn, B, N, -1
            label_total = torch.stack(label_lst).view(-1)
            # print(h_total.shape)
            # print(z_total.shape)

        # proj_dim = h_total.shape[-1]
        feature_dim = z_total.shape[-1]
        # h_total = h_total.view(-1, N, proj_dim)
        z_total = z_total.view(-1, N, feature_dim)
        # print(z_total.shape)

        z_total = z_total[:,:2,:] # n, 2, -1
        z_diff = z_total[:,1:,:] - z_total[:,:-1,:] # n, 1, -1
        z_diff = z_diff.squeeze(1) # n, -1

        # z_mask = z_diff[:, 0] > 0
        # z_mask = z_mask.unsqueeze(1)
        # z_diff = 2 * z_diff * z_mask - z_diff # convert direction
        # # print(z_diff.shape)
        # z_diff = z_diff / torch.norm(z_diff, dim=0, keepdim=True)

        if self.only_diff:
            return z_diff, label_total
        z_concate = torch.cat((z_total[:, 0, :], z_diff), dim=1) # n, -1
        # print(z_concate.shape)
        return z_concate, label_total
    
    def knn(self, features, labels, k=1):
        """
        Evaluating knn accuracy in feature space.
        Calculates only top-1 accuracy (returns 0 for top-5)
        Args:
            features: [... , dataset_size, feat_dim]
            labels: [... , dataset_size]
            k: nearest neighbours
        Returns: train accuracy, or train and test acc
        """
        feature_dim = features.shape[-1]
        with torch.no_grad():
            features_np = features.cpu().view(-1, feature_dim).numpy()
            labels_np = labels.cpu().view(-1).numpy()
            # fit
            # print(features_np.shape, labels_np.shape)
            self.cls = KNeighborsClassifier(k, metric="cosine").fit(features_np, labels_np)
            acc = self.eval(features, labels)
            
        return acc
    
    def eval(self, features, labels):
        feature_dim = features.shape[-1]
        features = features.cpu().view(-1, feature_dim).numpy()
        labels = labels.cpu().view(-1).numpy()
        acc = 100 * np.mean(cross_val_score(self.cls, features, labels))
        return acc

    def _find_best_indices(self, h_query, h_ref):
        h_query = h_query / h_query.norm(dim=1).view(-1, 1)
        h_ref = h_ref / h_ref.norm(dim=1).view(-1, 1)
        scores = torch.matmul(h_query, h_ref.t())  # [query_bs, ref_bs]
        score, indices = scores.topk(1, dim=1)  # select top k best
        return score, indices

    def fit(self, train_loader, test_loader=None):
        with torch.no_grad():
            h_train, l_train = self.extract_knn_features(train_loader)
            train_acc = self.knn(h_train, l_train, k=self.k)

            if test_loader is not None:
                h_test, l_test = self.extract_knn_features(test_loader)
                test_acc = self.eval(h_test, l_test)
                return train_acc, test_acc
            



    

    