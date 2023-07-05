import os
import sys
import argparse
sys.path.append("/home/siyich/byol-pytorch/byol_3d")
sys.path.append("/home/siyich/byol-pytorch/evaluation_3d")
sys.path.append("/home/siyich/byol-pytorch/utils")
from byol_3d import BYOL
from byolode_3d import BYOL_ODE
from nc_straighten import *

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms as T
import torch.nn.functional as F

from dataloader_3d import get_data_ucf, get_data_hmdb
from torch.utils.data import DataLoader

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import logging
import matplotlib.pyplot as plt

from augmentation import *

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--hmdb', action='store_true')
parser.add_argument('--random', action='store_true')

parser.add_argument('--ckpt_folder', default='checkpoints/3dbase_ucf101_lr0.0001_wd1e-05', type=str)
parser.add_argument('--epoch_num', default=100, type=int)

parser.add_argument('--num_seq', default=4, type=int)
parser.add_argument('--seq_len', default=4, type=int)
parser.add_argument('--downsample', default=4, type=int)
parser.add_argument('--num_aug', default=1, type=int)

parser.add_argument('--ode', action='store_true')

parser.add_argument('--num_batch', default=100, type=int)
parser.add_argument('--straighten', action='store_true')
parser.add_argument('--pca', action='store_true')
parser.add_argument('--proj_dim', default=100, type=int)
parser.add_argument('--vis_pca', action='store_true')
parser.add_argument('--vis_class', action='store_true')


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
        RandomCrop(size=128, consistent=True),
        Scale(size=(128,128)),
        ToTensor(),
        Normalize()
    ])
    return transform


def perform_nc_straighten(model, train_loader, test_loader):
    ssl_evaluator = NC(model=model, device=cuda)
    x_train, labels_train = ssl_evaluator.extract_data(train_loader, args.num_batch)
    x_test, labels_test = ssl_evaluator.extract_data(test_loader, args.num_batch)

    if args.vis_pca:
        ssl_evaluator.visualize_latent_pca(x_train, labels_train, args.seq_len, args.num_seq, "train.png")
        ssl_evaluator.visualize_latent_pca(x_test, labels_test, args.seq_len, args.num_seq, "test.png")
        return

    if args.vis_class:
        ssl_evaluator.visualize_class(x_train, labels_train, args.seq_len, args.num_seq, "class_train.png")
        ssl_evaluator.visualize_class(x_test, labels_test, args.seq_len, args.num_seq, "class_test.png")
        return

    if not args.straighten:
        m_train, s_train, v_train, m_train_c, s_train_c, v_train_c = ssl_evaluator.seq_similarity(x_train, args.seq_len, args.num_seq)
        m_test, s_test, v_test, m_test_c, s_test_c, v_test_c = ssl_evaluator.seq_similarity(x_test, args.seq_len, args.num_seq)

        print(f"Mean for train split: {m_train}")
        print(f"Mean for val split: {m_test} \n")
        print(f"Std for train split: {s_train}")
        print(f"Std for val split: {s_test} \n")
        print(f"Variance for train split: {v_train}")
        print(f"Variance for val split: {v_test} \n")

        print(f"Cross Mean for train split: {m_train_c}")
        print(f"Cross Mean for val split: {m_test_c} \n")
        print(f"Cross Std for train split: {s_train_c}")
        print(f"Cross Std for val split: {s_test_c} \n")
        print(f"Cross Variance for train split: {v_train_c}")
        print(f"Cross Variance for val split: {v_test_c} \n")
        print('-----------------')
        logging.info(f"Mean for train split: {m_train}")
        logging.info(f"Mean for val split: {m_test} \n")
        logging.info(f"Std for train split: {s_train}")
        logging.info(f"Std for val split: {s_test} \n")
        logging.info(f"Variance for train split: {v_train}")
        logging.info(f"Variance for val split: {v_test} \n")

        logging.info(f"Cross Mean for train split: {m_train_c}")
        logging.info(f"Cross Mean for val split: {m_test_c} \n")
        logging.info(f"Cross Std for train split: {s_train_c}")
        logging.info(f"Cross Std for val split: {s_test_c} \n")
        logging.info(f"Cross Variance for train split: {v_train_c}")
        logging.info(f"Cross Variance for val split: {v_test_c} \n")
        logging.info('-----------------')
        return m_train, s_train, v_train, m_test, s_test, v_test

    else:
        if args.pca:
            c_seq_train, c_out_train = ssl_evaluator.seq_pca_linear(x_train, args.seq_len, args.num_seq, args.proj_dim)
            c_seq_test, c_out_test = ssl_evaluator.seq_pca_linear(x_test, args.seq_len, args.num_seq, args.proj_dim)
        else:
            c_seq_train, c_out_train = ssl_evaluator.seq_linear(x_train, args.seq_len, args.num_seq)
            c_seq_test, c_out_test = ssl_evaluator.seq_linear(x_test, args.seq_len, args.num_seq)

        print(f"Curvature of videos for train split: {c_seq_train}")
        print(f"Curvature of videos for val split: {c_seq_test} \n")
        print(f"Curvature of latents for train split: {c_out_train}")
        print(f"Curvature of latents for val split: {c_out_test} \n")
        logging.info(f"Curvature of videos for train split: {c_seq_train}")
        logging.info(f"Curvature of videos for val split: {c_seq_test} \n")
        logging.info(f"Curvature of latents for train split: {c_out_train}")
        logging.info(f"Curvature of latents for val split: {c_out_test} \n")
        return c_seq_train, c_out_train, c_seq_test, c_out_test

def main():
    torch.manual_seed(233)
    np.random.seed(233)

    global args
    args = parser.parse_args()

    ckpt_folder = args.ckpt_folder
    ckpt_path = os.path.join(ckpt_folder, 'resnet_epoch%s.pth.tar' % args.epoch_num)

    if not args.straighten:
        if not args.hmdb:
            logging.basicConfig(filename=os.path.join(ckpt_folder, 'ucf_nc.log'), level=logging.INFO)
        else:
            logging.basicConfig(filename=os.path.join(ckpt_folder, 'hmdb_nc.log'), level=logging.INFO)
    else:
        if not args.hmdb:
            logging.basicConfig(filename=os.path.join(ckpt_folder, 'ucf_straighten.log'), level=logging.INFO)
        else:
            logging.basicConfig(filename=os.path.join(ckpt_folder, 'hmdb_straighten.log'), level=logging.INFO)
    logging.info('Started')
    if not args.random:
        logging.info(ckpt_path)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    global cuda
    cuda = torch.device('cuda')

    resnet = models.video.r3d_18()
    # modify model
    # resnet.stem[0] = torch.nn.Conv3d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # resnet.maxpool = torch.nn.Identity()

    # resnet = nn.DataParallel(resnet)
    resnet = resnet.to(cuda)
    resnet.eval()
    
    # load data
    if not args.hmdb:
        logging.info(f"k-nn accuracy performed on ucf \n")
        train_loader = get_data_ucf(batch_size=args.batch_size, 
                                    mode='train', 
                                    transform=test_transform(), 
                                    transform2=test_transform(),
                                    seq_len=args.seq_len, 
                                    num_seq=args.num_seq, 
                                    downsample=args.downsample,
                                    num_aug=args.num_aug)
        test_loader = get_data_ucf(batch_size=args.batch_size, 
                                    mode='val', 
                                    transform=test_transform(), 
                                    transform2=test_transform(),
                                    seq_len=args.seq_len, 
                                    num_seq=args.num_seq, 
                                    downsample=args.downsample,
                                    num_aug=args.num_aug)
    else:
        logging.info(f"k-nn accuracy performed on hmdb \n")
        train_loader = get_data_hmdb(batch_size=args.batch_size, 
                                    mode='train', 
                                    transform=test_transform(), 
                                    transform2=test_transform(),
                                    seq_len=args.seq_len, 
                                    num_seq=args.num_seq, 
                                    downsample=args.downsample,
                                    num_aug=args.num_aug)
        test_loader = get_data_hmdb(batch_size=args.batch_size, 
                                    mode='val', 
                                    transform=test_transform(), 
                                    transform2=test_transform(),
                                    seq_len=args.seq_len, 
                                    num_seq=args.num_seq, 
                                    downsample=args.downsample,
                                    num_aug=args.num_aug)
    
    # random weight
    if args.random:
        logging.info(f"k-nn accuracy performed with random weight\n")
        # print(resnet)
        resnet.fc = torch.nn.Identity()
        perform_nc_straighten(resnet, train_loader, test_loader)

    else:
        # after training
        logging.info(f"k-nn accuracy performed after ssl\n")
        resnet.load_state_dict(torch.load(ckpt_path)) # load model
        resnet.fc = torch.nn.Identity()
        perform_nc_straighten(resnet, train_loader, test_loader)




if __name__ == '__main__':
    main()