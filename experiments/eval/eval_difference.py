import os
import sys
import argparse

sys.path.append("/home/siyich/byol-pytorch/evaluation_3d")
sys.path.append("/home/siyich/byol-pytorch/utils")

# sys.path.append("/home/siyich/byol-pytorch/byol_3d")
# from byol_3d import BYOL
# from byolseq_3d import BYOL_SEQ
# from pc_vic_3d import PC_VIC

from difference import *

sys.path.append("/home/siyich/byol-pytorch/pcnet_3d")
from pcnet_vic import PCNET_VIC

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

# import logging
import matplotlib.pyplot as plt

from augmentation import *


parser = argparse.ArgumentParser()
parser.add_argument('--frame_root', default='/data', type=str,
                    help='root folder to store data like UCF101/..., better to put in servers SSD \
                    default path is mounted from data server for the home directory')
# --frame_root /data
parser.add_argument('--num_seq', default=8, type=int)
parser.add_argument('--seq_len', default=8, type=int)
parser.add_argument('--downsample', default=2, type=int)
parser.add_argument('--inter_len', default=0, type=int)    # does not need to be positive

parser.add_argument('--gpu', default='1', type=str)
parser.add_argument('--batch_size', default=1, type=int)

parser.add_argument('--random', action='store_true')
parser.add_argument('--kinetics', action='store_true')

parser.add_argument('--ckpt_folder', default='', type=str)
parser.add_argument('--epoch_num', default=100, type=int)
parser.add_argument('--ckpt_path', default='', type=str)

parser.add_argument('--byol', action='store_true')
parser.add_argument('--log', action='store_true')
parser.add_argument('--input_num', default=10000, type=int)



def test_transform():
    transform = transforms.Compose([
        RandomCrop(size=112, consistent=True),
        Scale(size=(112,112)),
        ToTensor(),
        Normalize()
    ])
    return transform


def exclude_bias_and_norm(p):
    return p.ndim == 1
    

def main():
    torch.manual_seed(233)
    np.random.seed(233)

    global args
    args = parser.parse_args()

    ckpt_folder = args.ckpt_folder
    ckpt_path = args.ckpt_path

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    global cuda
    cuda = torch.device('cuda')

    if not args.kinetics:
        resnet = models.video.r3d_18()
        # modify model
        # resnet.stem[0] = torch.nn.Conv3d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    else:
        resnet = models.video.r3d_18(pretrained=True)
        # modify model
        # resnet.layer4[1].conv2[0] = torch.nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
    
    # resnet.maxpool = torch.nn.Identity()

    # if args.byol:
    #     # model_select = BYOL
    #     model_select = BYOL_SEQ
    # else:
    #     model_select = PC_VIC

    model_select = PCNET_VIC

    model = model_select(
        resnet,
        clip_size = 8,
        image_size = 112,
        hidden_layer = 'avgpool',
        projection_size = 2048,
        projection_hidden_size = 2048,
        pred_hidden_size = 512,
        num_predictor = 1,
        pred_layer = 0,
        predictor = 1,
        proj_layer = 3,
        # pred_bn_last = True
    )

    # model = model_select(
    #     resnet,
    #     clip_size = 8,
    #     image_size = 112,
    #     hidden_layer = 'avgpool',
    #     projection_size = 256,
    #     projection_hidden_size = 4096,
    # )

    model = nn.DataParallel(model)
    model = model.to(cuda)
    model.eval()

    train_loader = get_data_ucf(batch_size=args.batch_size, 
                                mode='train', 
                                # transform=test_transform(), 
                                # transform2=None,
                                transform_consistent=test_transform(), 
                                transform_inconsistent=None,
                                seq_len=args.seq_len, 
                                num_seq=args.num_seq, 
                                downsample=args.downsample,
                                inter_len=args.inter_len,
                                frame_root=args.frame_root,
                                # num_aug=1,
                                )
    test_loader = get_data_ucf(batch_size=args.batch_size, 
                                mode='test',
                                # transform=test_transform(), 
                                # transform2=None,
                                transform_consistent=test_transform(), 
                                transform_inconsistent=None,
                                seq_len=args.seq_len, 
                                num_seq=args.num_seq, 
                                downsample=args.downsample,
                                inter_len=args.inter_len,
                                frame_root=args.frame_root,
                                # num_aug=1,
                                )

    # random weight
    if args.random:
        print("diff with random weight")
    elif args.kinetics:
        print("diff with kinetics weight")
    else:
        # after training
        print("diff with ssl")
        # model.load_state_dict(torch.load(ckpt_path)) # load model
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])

    if args.byol:
        ssl_evaluator = Difference(model=model.module.online_encoder, device=cuda, input_num=args.input_num)
    else:
        ssl_evaluator = Difference(model=model.module.encoder, device=cuda, input_num=args.input_num)
    
    # rh, rz = ssl_evaluator.extract_diff_rank(loader = train_loader)
    # print("rank for train diff obtained", rh, rz)
    # rh, rz = ssl_evaluator.extract_diff_rank(loader = test_loader)
    # print("rank for val diff obtained", rh, rz)

    ssl_evaluator.visualize_class(loader = train_loader, num_class=101, fpath=ckpt_folder)
    print("train diff visualized")
    ssl_evaluator.visualize_class(loader = test_loader, num_class=101, fpath=ckpt_folder, train=False)
    print("val diff visualized")

    ssl_evaluator.visualize_class(loader = train_loader, num_class=101, fpath=ckpt_folder, diff=False)
    print("train diff visualized")
    ssl_evaluator.visualize_class(loader = test_loader, num_class=101, fpath=ckpt_folder, train=False, diff=False)
    print("val diff visualized")




if __name__ == '__main__':
    main()