import os
import sys
import argparse
sys.path.append("/home/siyich/byol-pytorch/byol_3d")
sys.path.append("/home/siyich/byol-pytorch/evaluation_3d")
sys.path.append("/home/siyich/byol-pytorch/utils")
from byol_3d import BYOL
from byolode_3d import BYOL_ODE
from linear_evaluation import Linear_Eval

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
parser.add_argument('--epochs', default=30, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')

parser.add_argument('--pretrain_path', default='', type=str)
parser.add_argument('--pretrain', action='store_true')

parser.add_argument('--gpu', default='0,1', type=str)
parser.add_argument('--batch_size', default=8, type=int)

parser.add_argument('--hmdb', action='store_true')
parser.add_argument('--random', action='store_true')
parser.add_argument('--input_dim', default=512, type=int)
parser.add_argument('--class_num', default=101, type=int)

parser.add_argument('--ckpt_folder', default='/home/siyich/byol-pytorch/checkpoints_bad/3dseq_ucf101_lr0.0001_wd1e-05', type=str)
parser.add_argument('--epoch_num', default=100, type=int)

parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--wd', default=0, type=float, help='weight decay')

parser.add_argument('--dropout', default=0.2, type=float)

parser.add_argument('--num_seq', default=1, type=int)
parser.add_argument('--seq_len', default=4, type=int)
parser.add_argument('--downsample', default=4, type=int)
parser.add_argument('--num_aug', default=1, type=int)

parser.add_argument('--ode', action='store_true')


def default_transform():
    transform = transforms.Compose([
        RandomHorizontalFlip(consistent=True),
        RandomCrop(size=128, consistent=True),
        Scale(size=(128,128)),
        GaussianBlur(size=128, p=0.5, consistent=True),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.8),
        RandomGray(consistent=False, p=0.2),
        ToTensor(),
        Normalize()
    ])
    return transform


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


def linear_train(predict_model, train_loader, criterion, optimizer):
    predict_model.train()

    total_loss = 0.
    total_acc = 0.
    num_batches = len(train_loader)

    for data in train_loader:
        images, label = data
        images = images.to(cuda)
        label = label.to(cuda).squeeze(1)
        output = predict_model(images)

        # print(output.size(), label.size())

        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = calc_accuracy(output, label)
        total_loss += loss.sum().item()
        total_acc += acc.cpu().detach().numpy()

    mean_loss = total_loss/num_batches
    mean_acc = total_acc/num_batches
    return mean_loss, mean_acc


def linear_eval(predict_model, test_loader):
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

    ckpt_folder = args.ckpt_folder
    ckpt_path = os.path.join(ckpt_folder, 'resnet_epoch%s.pth.tar' % args.epoch_num)

    if not args.hmdb:
        args.class_num = 101
        if args.random:
            linear_folder = os.path.join(ckpt_folder, 'ucf_linear_epoch0') 
        else:
            linear_folder = os.path.join(ckpt_folder, 'ucf_linear_epoch%s' % args.epoch_num)
    else:
        args.class_num = 51
        if args.random:
            linear_folder = os.path.join(ckpt_folder, 'hmdb_linear_epoch0') 
        else:
            linear_folder = os.path.join(ckpt_folder, 'hmdb_linear_epoch%s' % args.epoch_num)
    linear_folder = linear_folder+'_lr%s_wd%s_dr%s' % (args.lr, args.wd, args.dropout)
    if not os.path.exists(linear_folder):
        os.makedirs(linear_folder)
    
    logging.basicConfig(filename=os.path.join(linear_folder, 'ucf_linear.log'), level=logging.INFO)
    logging.info('Started')

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    global cuda
    cuda = torch.device('cuda')

    resnet = models.video.r3d_18()
    # modify model
    resnet.stem[0] = torch.nn.Conv3d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # resnet.maxpool = torch.nn.Identity()

    if not args.ode:
        model = BYOL(
            resnet,
            clip_size = 8,
            image_size = 128,
            hidden_layer = 'avgpool',
            projection_size = 256,
            projection_hidden_size = 4096,
        )
    else:
        model = BYOL_ODE(
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

    predict_model = Linear_Eval(model.module.online_encoder, args.input_dim, args.class_num, args.dropout)
    predict_model = nn.DataParallel(predict_model)
    predict_model = predict_model.to(cuda)

    if args.pretrain:
        predict_model.load_state_dict(torch.load(args.pretrain_path)) # load model
        logging.info(args.pretrain_path)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(predict_model.parameters(), lr=args.lr, weight_decay=args.wd)
    # scheduler = LinearLR(optimizer, start_factor=0.3, total_iters=10)

    if not args.hmdb:
        logging.info(f"linear evaluation performed on ucf")
        train_loader = get_data_ucf(batch_size=args.batch_size, 
                                    mode='train', 
                                    transform=default_transform(), 
                                    transform2=default_transform(),
                                    seq_len=args.seq_len, 
                                    num_seq=args.num_seq, 
                                    downsample=args.downsample,
                                    num_aug=args.num_aug)
        test_loader = get_data_ucf(batch_size=args.batch_size, 
                                    mode='val', 
                                    transform=default_transform(), 
                                    transform2=default_transform(),
                                    seq_len=args.seq_len, 
                                    num_seq=args.num_seq, 
                                    downsample=args.downsample,
                                    num_aug=args.num_aug)
    else:
        logging.info(f"linear evaluation performed on hmdb")
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
    


    if (not args.random) and (not args.pretrain):
        resnet.load_state_dict(torch.load(ckpt_path)) # load model
        logging.info(ckpt_path)
        logging.info(f"linear evaluation performed after ssl")
    else:
        pass
    
    epoch_list = range(args.start_epoch, args.epochs)
    best_acc = 0
    for i in epoch_list:
        train_loss, train_acc = linear_train(predict_model, train_loader, criterion, optimizer)
        test_acc = linear_eval(predict_model, test_loader)
        # scheduler.step()
        if test_acc > best_acc:
            best_acc = test_acc
        
        print('Epoch: %s, Train loss: %s' % (i, train_loss))
        print('Epoch: %s, Train acc: %s' % (i, train_acc))
        print('Epoch: %s, Test acc: %s' % (i, test_acc))
        logging.info('Epoch: %s, Train loss: %s' % (i, train_loss))
        logging.info('Epoch: %s, Train acc: %s' % (i, train_acc))
        logging.info('Epoch: %s, Test acc: %s' % (i, test_acc))
        
        if (i+1)%5 == 0:
            # save your improved network
            checkpoint_path = os.path.join(
                linear_folder, 'linear_epoch%s.pth.tar' % str(i+1))
            torch.save(predict_model.state_dict(), checkpoint_path)

    print('Linear Eval Acc: %s \n' % best_acc)
    logging.info('Linear Eval Acc: %s \n' % best_acc)


if __name__ == '__main__':
    main()