import os
import sys
import argparse
# sys.path.append("/home/siyich/byol-pytorch/byol_3d")
sys.path.append("/home/siyich/byol-pytorch/evaluation_3d")
sys.path.append("/home/siyich/byol-pytorch/utils")

from retrieval import *

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms as T
import torch.nn.functional as F

from dataloader_v2 import get_data_ucf
from torch.utils.data import DataLoader

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import logging
import matplotlib.pyplot as plt

from augmentation import *


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--batch_size', default=16, type=int)
# parser.add_argument('--hmdb', action='store_true')
# parser.add_argument('--mnist', action='store_true')
parser.add_argument('--random', action='store_true')
parser.add_argument('--kinetics', action='store_true')

parser.add_argument('--k', default=1, type=int)

parser.add_argument('--ckpt_folder', default='checkpoints/3dbase_ucf101_lr0.0001_wd1e-05', type=str)
parser.add_argument('--epoch_num', default=100, type=int)

parser.add_argument('--num_seq', default=10, type=int)
parser.add_argument('--seq_len', default=8, type=int)
parser.add_argument('--downsample', default=3, type=int)
parser.add_argument('--inter_len', default=0, type=int)
# parser.add_argument('--num_aug', default=1, type=int)

parser.add_argument('--img_size', default=128, type=int)
parser.add_argument('--r21d', action='store_true')
parser.add_argument('--diff', action='store_true')


# def default_transform():
#     transform = transforms.Compose([
#         RandomHorizontalFlip(consistent=True),
#         RandomCrop(size=128, consistent=True),
#         Scale(size=(128,128)),
#         GaussianBlur(size=128, p=0.5, consistent=True),
#         ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.8),
#         RandomGray(consistent=False, p=0.2),
#         ToTensor(),
#         Normalize()
#     ])
#     return transform


def test_transform():
    transform = transforms.Compose([
        RandomCrop(size=args.img_size, consistent=True),
        Scale(size=(128, 128)),
        ToTensor(),
        Normalize()
    ])
    return transform


def extract_features(loader, model, test=True, diff=False):
    features = []
    label_lst = []

    i = 0
    with torch.no_grad():
        for data_i in loader:
            # B, N, C, T, H, W
            input_tensor, label = data_i
            input_tensor = input_tensor.to(torch.device('cuda'))
            B, N, C, T, H, W = input_tensor.shape
            print(B, N, C, T, H, W)
            input_tensor_diff = input_tensor[:,:,:,1:,:,:] - input_tensor[:,:,:,:-1,:,:]
            print(input_tensor_diff.shape)

            h = model(input_tensor.view(B*N, C, T, H, W))
            h_diff = model(input_tensor_diff.view(B*N, C, T-1, H, W))
            # # kind 1
            if test:
                h = h.reshape(B, N, -1) # B, N, D
                h_diff = h_diff.reshape(B, N, -1)
                if not diff:
                    print("0")
                    features.append(h)
                else:
                    print("1")
                    features.append(torch.cat((h, h_diff), -1))
                label_lst.append(label)
            # kind 2
            else:
                if not diff:
                    print("0")
                    features.append(h)
                else:
                    print("1")
                    features.append(torch.cat((h, h_diff), -1))
                label_lst.append(torch.ones(B,N)*label)

            i += 1
            if i % 10 == 0:
                print(i)
            # if i > 2:
            #     break

        h_total = torch.vstack(features)
        # print(h_total.shape)
        # # kind 1
        if test:
            h_total = torch.mean(h_total, dim=1)
        label_total = torch.vstack(label_lst)

        

    return h_total, label_total
        

def perform_knn(model, train_loader, test_loader, k=1, diff=False):
    ssl_evaluator = Retrieval(model=model, k=k, device=cuda, num_seq=args.num_seq)
    h_train, l_train = extract_features(train_loader, model, diff=diff)

    train_acc = ssl_evaluator.knn(h_train, l_train, k=1)
    h_test, l_test = extract_features(test_loader, model, diff=diff)
    acc1, acc5, acc10  = ssl_evaluator.eval(h_test, l_test, l_train)

    # train_acc, val_acc = ssl_evaluator.fit(train_loader, test_loader)
    print(f"k-nn accuracy k= {ssl_evaluator.k} for train split: {train_acc}")
    print(f"k-nn accuracy k= {ssl_evaluator.k} for test split: {acc1}, {acc5}, {acc10} \n")
    print('-----------------')
    logging.info(f"k-nn accuracy k= {ssl_evaluator.k} for train split: {train_acc}")
    logging.info(f"k-nn accuracy k= {ssl_evaluator.k} for test split: {acc1}, {acc5}, {acc10} \n")
    logging.info('-----------------')
    return train_acc, acc1, acc5, acc10

    

def main():
    torch.manual_seed(233)
    np.random.seed(233)

    global args
    args = parser.parse_args()

    ckpt_folder = args.ckpt_folder
    ckpt_path = os.path.join(ckpt_folder, 'resnet_epoch%s.pth.tar' % args.epoch_num)

    # if not args.hmdb:
    logging.basicConfig(filename=os.path.join(ckpt_folder, 'ucf_retrieval.log'), level=logging.INFO)
    # else:
        # logging.basicConfig(filename=os.path.join(ckpt_folder, 'hmdb_knn.log'), level=logging.INFO)
    logging.info('Started')
    if not args.random:
        logging.info(ckpt_path)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    global cuda
    cuda = torch.device('cuda')


    if args.r21d:
        model_name = 'r21d18'
        if not args.kinetics:
            resnet = models.video.r2plus1d_18()
        else:
            resnet = models.video.r2plus1d_18(pretrained=True)
    else:
        model_name = 'r3d18'
        if not args.kinetics:
            resnet = models.video.r3d_18()
        else:
            resnet = models.video.r3d_18(pretrained=True)

    # if not args.kinetics:
    #     resnet = models.video.r3d_18()
    #     # modify model
    #     # resnet.stem[0] = torch.nn.Conv3d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # else:
    #     resnet = models.video.r3d_18(pretrained=True)
    #     # modify model
    #     # resnet.layer4[1].conv2[0] = torch.nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
    
    if not args.random and not args.kinetics:
        resnet.load_state_dict(torch.load(ckpt_path)) # load model
    resnet.fc = torch.nn.Identity()

    resnet = nn.DataParallel(resnet)
    resnet = resnet.to(cuda)
    resnet.eval()

    if args.img_size == 224:
        dim = 240
    else:
        dim = 150

    # if not args.hmdb and not args.mnist:
    logging.info(f"k-nn accuracy performed on ucf \n")
    train_loader = get_data_ucf(batch_size=args.batch_size, 
                                mode='train', 
                                # transform=default_transform(), 
                                # transform2=default_transform(),
                                transform_consistent=test_transform(),
                                transform_inconsistent=None,
                                seq_len=args.seq_len, 
                                num_seq=args.num_seq, 
                                downsample=args.downsample,
                                inter_len=args.inter_len,
                                # num_aug=args.num_aug,
                                dim=dim,
                                frame_root='/data',
                                random=True
                                )
    test_loader = get_data_ucf(batch_size=args.batch_size, 
                                mode='test', 
                                # transform=default_transform(), 
                                # transform2=default_transform(),
                                transform_consistent=test_transform(),
                                transform_inconsistent=None,
                                seq_len=args.seq_len, 
                                num_seq=args.num_seq, 
                                inter_len=args.inter_len,
                                downsample=args.downsample,
                                # num_aug=args.num_aug,
                                dim=dim,
                                frame_root='/data',
                                random=True
                                )
    # elif args.hmdb:
    #     logging.info(f"k-nn accuracy performed on hmdb \n")
    #     train_loader = get_data_hmdb(batch_size=args.batch_size, 
    #                                 mode='train', 
    #                                 transform=test_transform(),
    #                                 transform2=test_transform(),
    #                                 # transform=default_transform(), 
    #                                 # transform2=default_transform(),
    #                                 seq_len=args.seq_len, 
    #                                 num_seq=args.num_seq, 
    #                                 downsample=args.downsample,
    #                                 num_aug=args.num_aug)
    #     test_loader = get_data_hmdb(batch_size=args.batch_size, 
    #                                 mode='test', 
    #                                 transform=test_transform(),
    #                                 transform2=test_transform(),
    #                                 # transform=default_transform(), 
    #                                 # transform2=default_transform(),
    #                                 seq_len=args.seq_len, 
    #                                 num_seq=args.num_seq, 
    #                                 downsample=args.downsample,
    #                                 num_aug=args.num_aug)
    # else:
    #     logging.info(f"k-nn accuracy performed on mnist \n")
    #     train_loader = get_data_mnist(batch_size=args.batch_size, 
    #                                 mode='train', 
    #                                 transform=test_transform(),
    #                                 transform2=test_transform(),
    #                                 # transform=default_transform(), 
    #                                 # transform2=default_transform(),
    #                                 seq_len=args.seq_len, 
    #                                 num_seq=args.num_seq, 
    #                                 downsample=args.downsample,
    #                                 num_aug=args.num_aug)
    #     test_loader = get_data_mnist(batch_size=args.batch_size, 
    #                                 mode='test', 
    #                                 transform=test_transform(),
    #                                 transform2=test_transform(),
    #                                 # transform=default_transform(), 
    #                                 # transform2=default_transform(),
    #                                 seq_len=args.seq_len, 
    #                                 num_seq=args.num_seq, 
    #                                 downsample=args.downsample,
    #                                 num_aug=args.num_aug)

    # random weight
    if args.random:
        logging.info(f"k-nn accuracy performed with random weight\n")
        perform_knn(resnet, train_loader, test_loader, args.k, args.diff)
    elif args.kinetics:
        logging.info(f"k-nn accuracy performed with kinetics weight\n")
        perform_knn(resnet, train_loader, test_loader, args.k, args.diff)
    else:
        # after training
        logging.info(f"k-nn accuracy performed after ssl\n")
        perform_knn(resnet, train_loader, test_loader, args.k, args.diff)




if __name__ == '__main__':
    main()