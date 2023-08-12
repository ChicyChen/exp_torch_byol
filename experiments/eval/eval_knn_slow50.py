import os
import sys
import argparse
sys.path.append("/home/siyich/byol-pytorch/byol_3d")
sys.path.append("/home/siyich/byol-pytorch/evaluation_3d")
sys.path.append("/home/siyich/byol-pytorch/utils")
from byol_3d import BYOL
from byolode_3d import BYOL_ODE
from knn import *

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms as T
import torch.nn.functional as F

from dataloader_3d import get_data_ucf, get_data_hmdb, get_data_mnist
from torch.utils.data import DataLoader

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import logging
import matplotlib.pyplot as plt

from augmentation import *


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--hmdb', action='store_true')
parser.add_argument('--mnist', action='store_true')
parser.add_argument('--random', action='store_true')
parser.add_argument('--kinetics', action='store_true')

parser.add_argument('--k', default=1, type=int)
parser.add_argument('--knn_input', default=1, type=int)

parser.add_argument('--ckpt_folder', default='checkpoints/3dbase_ucf101_lr0.0001_wd1e-05', type=str)
parser.add_argument('--epoch_num', default=50, type=int)

parser.add_argument('--num_seq', default=1, type=int)
parser.add_argument('--seq_len', default=8, type=int)
parser.add_argument('--downsample', default=4, type=int)
parser.add_argument('--num_aug', default=1, type=int)

parser.add_argument('--ode', action='store_true')


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
        RandomCrop(size=224, consistent=True),
        Scale(size=(224,224)),
        ToTensor(),
        Normalize()
    ])
    return transform



def perform_knn(model, train_loader, test_loader, k=1):
    ssl_evaluator = KNN(model=model, k=k, device=cuda, input_num=args.knn_input, byol=False)
    train_acc, val_acc = ssl_evaluator.fit(train_loader, test_loader)
    print(f"k-nn accuracy k= {ssl_evaluator.k} for train split: {train_acc}")
    print(f"k-nn accuracy k= {ssl_evaluator.k} for val split: {val_acc} \n")
    print('-----------------')
    logging.info(f"k-nn accuracy k= {ssl_evaluator.k} for train split: {train_acc}")
    logging.info(f"k-nn accuracy k= {ssl_evaluator.k} for val split: {val_acc} \n")
    logging.info('-----------------')
    return train_acc, val_acc

    

def main():
    torch.manual_seed(233)
    np.random.seed(233)

    global args
    args = parser.parse_args()

    ckpt_folder = args.ckpt_folder
    ckpt_path = os.path.join(ckpt_folder, 'resnet_epoch%s.pth.tar' % args.epoch_num)

    if not args.hmdb:
        logging.basicConfig(filename=os.path.join(ckpt_folder, 'ucf_knn.log'), level=logging.INFO)
    else:
        logging.basicConfig(filename=os.path.join(ckpt_folder, 'hmdb_knn.log'), level=logging.INFO)
    logging.info('Started')
    if not args.random:
        logging.info(ckpt_path)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    global cuda
    cuda = torch.device('cuda')

    resnet = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=False)
    resnet.blocks[5] = torch.nn.AdaptiveAvgPool3d(output_size=1)
    
    resnet = resnet.to(cuda)
    resnet.eval()

    if not args.hmdb and not args.mnist:
        logging.info(f"k-nn accuracy performed on ucf \n")
        train_loader = get_data_ucf(batch_size=args.batch_size, 
                                    mode='train', 
                                    # transform=default_transform(), 
                                    # transform2=default_transform(),
                                    transform=test_transform(),
                                    transform2=test_transform(),
                                    seq_len=args.seq_len, 
                                    num_seq=args.num_seq, 
                                    downsample=args.downsample,
                                    num_aug=args.num_aug,
                                    dim=240,
                                    )
        test_loader = get_data_ucf(batch_size=args.batch_size, 
                                    mode='val', 
                                    # transform=default_transform(), 
                                    # transform2=default_transform(),
                                    transform=test_transform(),
                                    transform2=test_transform(),
                                    seq_len=args.seq_len, 
                                    num_seq=args.num_seq, 
                                    downsample=args.downsample,
                                    num_aug=args.num_aug,
                                    dim=240,
                                    )
    elif args.hmdb:
        logging.info(f"k-nn accuracy performed on hmdb \n")
        train_loader = get_data_hmdb(batch_size=args.batch_size, 
                                    mode='train', 
                                    transform=default_transform(), 
                                    transform2=default_transform(),
                                    seq_len=args.seq_len, 
                                    num_seq=args.num_seq, 
                                    downsample=args.downsample,
                                    num_aug=args.num_aug)
        test_loader = get_data_hmdb(batch_size=args.batch_size, 
                                    mode='val', 
                                    transform=default_transform(), 
                                    transform2=default_transform(),
                                    seq_len=args.seq_len, 
                                    num_seq=args.num_seq, 
                                    downsample=args.downsample,
                                    num_aug=args.num_aug)
    else:
        logging.info(f"k-nn accuracy performed on mnist \n")
        train_loader = get_data_mnist(batch_size=args.batch_size, 
                                    mode='train', 
                                    transform=default_transform(), 
                                    transform2=default_transform(),
                                    seq_len=args.seq_len, 
                                    num_seq=args.num_seq, 
                                    downsample=args.downsample,
                                    num_aug=args.num_aug)
        test_loader = get_data_mnist(batch_size=args.batch_size, 
                                    mode='val', 
                                    transform=default_transform(), 
                                    transform2=default_transform(),
                                    seq_len=args.seq_len, 
                                    num_seq=args.num_seq, 
                                    downsample=args.downsample,
                                    num_aug=args.num_aug)

    # random weight
    if args.random:
        logging.info(f"k-nn accuracy performed with random weight\n")
        perform_knn(resnet, train_loader, test_loader, args.k)
    elif args.kinetics:
        logging.info(f"k-nn accuracy performed with kinetics weight\n")
        perform_knn(resnet, train_loader, test_loader, args.k)
    else:
        # after training
        logging.info(f"k-nn accuracy performed after ssl\n")
        resnet.load_state_dict(torch.load(ckpt_path)) # load model
        perform_knn(resnet, train_loader, test_loader, args.k)




if __name__ == '__main__':
    main()