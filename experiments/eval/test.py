import os
import sys
import argparse
sys.path.append("/home/siyich/byol-pytorch/byol_3d")
sys.path.append("/home/siyich/byol-pytorch/evaluation_3d")
sys.path.append("/home/siyich/byol-pytorch/utils")
from byol_3d import BYOL
from finetune import Fine_Tune

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
from torch.optim.lr_scheduler import LinearLR


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0,1', type=str)
parser.add_argument('--batch_size', default=1, type=int)

parser.add_argument('--ckpt_folder', default='checkpoints_bad/3dseq_ucf101_lr0.0001_wd1e-05/ucf_tune_epoch100_lr0.001_wd0.001', type=str)
parser.add_argument('--ckpt_name', default='tune_epoch30.pth.tar', type=str)

parser.add_argument('--hmdb', action='store_true')
parser.add_argument('--input_dim', default=512, type=int)
parser.add_argument('--class_num', default=101, type=int)

parser.add_argument('--num_seq', default=1, type=int)
parser.add_argument('--seq_len', default=8, type=int)
parser.add_argument('--downsample', default=4, type=int)
parser.add_argument('--num_aug', default=1, type=int)


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


def calc_accuracy(output, target):
    '''output: (B, N); target: (B)'''
    target = target.squeeze()
    _, pred = torch.max(output, 1)
    return torch.mean((pred == target).float())


def tune_eval(predict_model, test_loader):
    predict_model.eval()
    acc_list =[]
    for data in test_loader:
        images, label = data
        images = images.to(cuda)
        label = label.to(cuda)
        output = predict_model(images)
        acc = calc_accuracy(output, label)
        acc_list.append(acc.cpu().detach().numpy())
    mean_acc = np.mean(acc_list)
    return mean_acc
    


def main():
    torch.manual_seed(233)
    np.random.seed(233)

    global args
    args = parser.parse_args()

    ckpt_path = os.path.join(args.ckpt_folder, args.ckpt_name)
    
    logging.basicConfig(filename=os.path.join(args.ckpt_folder, 'test.log'), level=logging.INFO)
    logging.info('Started')

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    global cuda
    cuda = torch.device('cuda')

    resnet = models.video.r3d_18()
    # modify model
    # resnet.stem[0] = torch.nn.Conv3d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # resnet.maxpool = torch.nn.Identity()

    model = BYOL(
        resnet,
        clip_size = 8,
        image_size = 128,
        hidden_layer = 'avgpool',
        projection_size = 256,
        projection_hidden_size = 4096,
    )

    model = nn.DataParallel(model)
    model = model.to(cuda)
    model.eval()

    if not args.hmdb:
        args.class_num = 101
        logging.info(f"test performed on ucf")
        test_loader = get_data_ucf(batch_size=args.batch_size, 
                                    mode='test', 
                                    transform=test_transform(), 
                                    transform2=test_transform(),
                                    seq_len=args.seq_len, 
                                    num_seq=args.num_seq, 
                                    downsample=args.downsample,
                                    num_aug=args.num_aug)
    else:
        args.class_num = 51
        logging.info(f"test performed on hmdb")
        test_loader = get_data_hmdb(batch_size=args.batch_size, 
                                    mode='test', 
                                    transform=test_transform(), 
                                    transform2=test_transform(),
                                    seq_len=args.seq_len, 
                                    num_seq=args.num_seq, 
                                    downsample=args.downsample,
                                    num_aug=args.num_aug)
    
    predict_model = Fine_Tune(model.module.online_encoder, args.input_dim, args.class_num)
    predict_model = nn.DataParallel(predict_model)
    predict_model = predict_model.to(cuda)

    predict_model.load_state_dict(torch.load(ckpt_path)) # load model
    logging.info(f"finetuning performed after ssl")
        
    test_acc = tune_eval(predict_model, test_loader)
        
    print('Test acc: %s' % (test_acc))
    logging.info('Test acc: %s' % (test_acc))


if __name__ == '__main__':
    main()