import numpy as np
import torch
import torch.nn as nn

from torchvision import transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

import copy
import torch.nn.functional as F

from torch.optim.optimizer import Optimizer, required
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import top_k_accuracy_score


"""# KNN evaluation to track classification accuray through ssl pretraining"""


class Retrieval():
    def __init__(self, model, k, device, num_seq=10, byol=True):
        super(Retrieval, self).__init__()
        self.k = k
        self.device = device
        self.num_seq = num_seq
        self.model = model
        self.model.eval()
        self.byol = byol

    # def extract_features(self, loader):
    #     """
    #     Infer/Extract features from a trained model
    #     Args:
    #         loader: train or test loader
    #     Returns: 3 tensors of all:  input_images, features , labels
    #     """
    #     # x_lst = []
    #     features = []
    #     label_lst = []

    #     with torch.no_grad():
    #         i = 0
    #         for data_i in loader:
    #             # B, N, C, T, H, W
    #             input_tensor, label = data_i
    #             input_tensor = input_tensor.to(self.device)
    #             B, N, C, T, H, W = input_tensor.shape
    #             # print(B, N, C, T, H, W)

    #             h = self.model(input_tensor.view(B*N, C, T, H, W))
    #             # print(hm.shape)
    #             h = h.reshape(B, N, -1) # B, N, D
    #             # print(hm.shape)
    #             # print(h.shape)

    #             # h = self.model(input_tensor.to(self.device))
    #             # h = h.reshape(h.shape[0], -1)
    #             # print(h.shape)

    #             features.append(h)
    #             # x_lst.append(input_tensor)
    #             label_lst.append(label)

    #             # if not test:
    #             #     h2 = self.model.get_representation(input_tensor2.to(self.device))
    #             #     features.append(h2)
    #             #     x_lst.append(input_tensor2)
    #             #     label_lst.append(label)

    #             i += 1
    #             if i % 100 == 0:
    #                 print(i)
    #             # if i > 100:
    #             #     break

    #         # x_total = torch.stack(x_lst)
    #         h_total = torch.stack(features)
    #         h_total = torch.mean(h_total, dim=1)
    #         label_total = torch.stack(label_lst)
            
    #         # h_total = torch.cat(FullGatherLayer.apply(h_total), dim=0)
    #         # label_total = torch.cat(FullGatherLayer.apply(label_total), dim=0)


    #         # print(h_total.shape)

    #     return h_total, label_total

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
            # self.cls2 = NearestNeighbors(k, metric="cosine").fit(features_np, labels_np)
            acc = self.cls.score(features_np, labels_np)
            
        return acc
    
    def eval(self, features, labels, y):
        feature_dim = features.shape[-1]
        with torch.no_grad():
            features = features.cpu().view(-1, feature_dim).numpy()
            labels = labels.cpu().view(-1).numpy()
            # acc = 100 * np.mean(cross_val_score(self.cls, features, labels))
            # acc = self.cls.score(features, labels)


            pred_nei = self.cls.kneighbors(features, n_neighbors=1, return_distance=False)
            total_num = pred_nei.shape[0]
            acc_num = 0
            for i in range(total_num):
                nei_indices = pred_nei[i]
                for j in nei_indices:
                    if y[j] == labels[i]:
                        acc_num += 1
                        break
            acc1 = acc_num/total_num

            pred_nei = self.cls.kneighbors(features, n_neighbors=5, return_distance=False)
            total_num = pred_nei.shape[0]
            acc_num = 0
            for i in range(total_num):
                nei_indices = pred_nei[i]
                for j in nei_indices:
                    if y[j] == labels[i]:
                        acc_num += 1
                        break
            acc5 = acc_num/total_num

            pred_nei = self.cls.kneighbors(features, n_neighbors=10, return_distance=False)
            total_num = pred_nei.shape[0]
            acc_num = 0
            for i in range(total_num):
                nei_indices = pred_nei[i]
                for j in nei_indices:
                    if y[j] == labels[i]:
                        acc_num += 1
                        break
            acc10 = acc_num/total_num


            """
            scores = self.cls.predict_proba(features) # B, NClass
            # print(labels.shape, scores.shape)
            acc1 = top_k_accuracy_score(labels, scores, k=1)
            acc5 = top_k_accuracy_score(labels, scores, k=5)
            acc10 = top_k_accuracy_score(labels, scores, k=10)
            acc50 = top_k_accuracy_score(labels, scores, k=50)
            """
        return acc1, acc5, acc10

    # def _find_best_indices(self, h_query, h_ref):
    #     h_query = h_query / h_query.norm(dim=1).view(-1, 1)
    #     h_ref = h_ref / h_ref.norm(dim=1).view(-1, 1)
    #     scores = torch.matmul(h_query, h_ref.t())  # [query_bs, ref_bs]
    #     score, indices = scores.topk(self.k, dim=1)  # select top k best
    #     return score, indices

    # def fit(self, train_loader, test_loader=None):
    #     # with torch.no_grad():
    #     h_train, l_train = self.extract_features(train_loader)
    #     train_acc = self.knn(h_train, l_train, k=1)

    #     # if test_loader is not None:
    #     h_test, l_test = self.extract_features(test_loader, test=True)
    #     test_acc = self.eval(h_test, l_test)
    #     return train_acc, test_acc