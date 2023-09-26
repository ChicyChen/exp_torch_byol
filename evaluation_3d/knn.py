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
from sklearn.neighbors import KNeighborsClassifier



"""# KNN evaluation to track classification accuray through ssl pretraining"""


class KNN():
    def __init__(self, model, k, device, input_num=2, byol=True):
        super(KNN, self).__init__()
        self.k = k
        self.device = device
        self.input_num = input_num
        self.model = model.to(device)
        self.model.eval()
        self.byol = byol

    def extract_features(self, loader, test=False):
        """
        Infer/Extract features from a trained model
        Args:
            loader: train or test loader
        Returns: 3 tensors of all:  input_images, features , labels
        """
        x_lst = []
        features = []
        label_lst = []

        with torch.no_grad():
            
            for data_i in loader:
                if self.input_num == 2:
                    input_tensor, input_tensor2, label = data_i
                else:
                    input_tensor, label = data_i
                if self.byol:
                    h = self.model.get_representation(input_tensor.to(self.device))
                else:
                    h = self.model(input_tensor.to(self.device))
                    h = h.reshape(h.shape[0], -1)
                    # print(h.shape)
                features.append(h)
                x_lst.append(input_tensor)
                label_lst.append(label)

                # if not test:
                #     h2 = self.model.get_representation(input_tensor2.to(self.device))
                #     features.append(h2)
                #     x_lst.append(input_tensor2)
                #     label_lst.append(label)

            x_total = torch.stack(x_lst)
            h_total = torch.stack(features)
            label_total = torch.stack(label_lst)

            return x_total, h_total, label_total

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
        # acc = 100 * np.mean(cross_val_score(self.cls, features, labels))
        acc = self.cls.score(features, labels)
        return acc

    def _find_best_indices(self, h_query, h_ref):
        h_query = h_query / h_query.norm(dim=1).view(-1, 1)
        h_ref = h_ref / h_ref.norm(dim=1).view(-1, 1)
        scores = torch.matmul(h_query, h_ref.t())  # [query_bs, ref_bs]
        score, indices = scores.topk(1, dim=1)  # select top k best
        return score, indices

    def fit(self, train_loader, test_loader=None):
        with torch.no_grad():
            x_train, h_train, l_train = self.extract_features(train_loader)
            train_acc = self.knn(h_train, l_train, k=self.k)

            if test_loader is not None:
                x_test, h_test, l_test = self.extract_features(test_loader, test=True)
                test_acc = self.eval(h_test, l_test)
                return train_acc, test_acc