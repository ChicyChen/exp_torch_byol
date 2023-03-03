import os
import sys
import argparse
sys.path.append("/home/siyich/byol-pytorch/byol_3d")
from byol_3d import BYOL

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms as T
import torch.nn.functional as F

from dataloader_3d import get_data_ucf
from torch.utils.data import DataLoader

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import logging
import matplotlib.pyplot as plt

from augmentation import *


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--pretrain_folder', default='', type=str)
parser.add_argument('--pretrain', action='store_true')
parser.add_argument('--gpu', default='0,1', type=str)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--knn', action='store_true')

parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')


def default_transform():
    transform = transforms.Compose([
        RandomHorizontalFlip(consistent=True),
        RandomCrop(size=128, consistent=True),
        Scale(size=(128,128)),
        GaussianBlur(size=128, p=0.5, consistent=True),
        # ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=1.0), # DPC
        # RandomGray(consistent=False, p=0.5), # DPC
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.8),
        RandomGray(consistent=False, p=0.2),
        ToTensor(),
        Normalize()
    ])
    return transform


def train_one_epoch(model, train_loader, optimizer, train=True):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.
    num_batches = len(train_loader)

    for data in train_loader:
        images, images2, label = data
        images = images.to(cuda)
        images2 = images2.to(cuda)
        label = label.to(cuda)
        # print(images.size(), images2.size())
        optimizer.zero_grad()
        loss = model(images, images2)
        if train:
            loss.sum().backward()
            optimizer.step()
            # EMA update
            model.module.update_moving_average()
        else:
            pass
        total_loss += loss.sum().item()
    
    return total_loss/num_batches

def main():
    torch.manual_seed(233)
    np.random.seed(233)

    global args
    args = parser.parse_args()

    ckpt_folder='/home/siyich/byol-pytorch/checkpoints/3dbase_ucf101_lr%s_wd%s' % (args.lr, args.wd)

    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)

    logging.basicConfig(filename=os.path.join(ckpt_folder, 'byol_train.log'), level=logging.INFO)
    logging.info('Started')

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    global cuda
    cuda = torch.device('cuda')

    resnet = models.video.r3d_18()
    # modify model
    resnet.stem[0] = torch.nn.Conv3d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # resnet.maxpool = torch.nn.Identity()

    model = BYOL(
        resnet,
        clip_size = 8,
        image_size = 128,
        hidden_layer = 'avgpool',
        projection_size = 256,
        projection_hidden_size = 4096,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    model = nn.DataParallel(model)
    model = model.to(cuda)

    if args.pretrain:
        pretrain_path = os.path.join(args.pretrain_folder, 'byol_epoch%s.pth.tar' % args.start_epoch)
        model.load_state_dict(torch.load(pretrain_path)) # load model

    train_loader = get_data_ucf(batch_size=args.batch_size, 
                                mode='train', 
                                transform=default_transform(), 
                                transform2=default_transform(),
                                seq_len=8, 
                                num_seq=1, 
                                downsample=8)
    test_loader = get_data_ucf(batch_size=args.batch_size, 
                                mode='val',
                                transform=default_transform(), 
                                transform2=default_transform(),
                                seq_len=8, 
                                num_seq=1, 
                                downsample=8)
    
    train_loss_list = []
    test_loss_list = []
    epoch_list = range(args.start_epoch, args.epochs)
    lowest_loss = np.inf
    best_epoch = 0

    for i in epoch_list:
        train_loss = train_one_epoch(model, train_loader, optimizer)
        test_loss = train_one_epoch(model, test_loader, optimizer, False)
        if test_loss < lowest_loss:
            lowest_loss = test_loss
            best_epoch = i + 1
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        print('Epoch: %s, Train loss: %s' % (i, train_loss))
        print('Epoch: %s, Test loss: %s' % (i, test_loss))
        logging.info('Epoch: %s, Train loss: %s' % (i, train_loss))
        logging.info('Epoch: %s, Test loss: %s' % (i, test_loss))

        if (i+1)%10 == 0:
            # save your improved network
            checkpoint_path = os.path.join(
                ckpt_folder, 'resnet_epoch%s.pth.tar' % str(i+1))
            torch.save(resnet.state_dict(), checkpoint_path)
            checkpoint_path = os.path.join(
                ckpt_folder, 'byol_epoch%s.pth.tar' % str(i+1))
            torch.save(model.state_dict(), checkpoint_path)
            


    logging.info('Training from ep %d to ep %d finished' %
          (args.start_epoch, args.epochs))
    logging.info('Best epoch: %s' % best_epoch)

    # plot training process
    plt.plot(epoch_list, train_loss_list, label = 'train')
    # if not args.no_val:
    plt.plot(epoch_list, test_loss_list, label = 'val')
    plt.title('Train and test loss')
    # plt.xticks(knn_list, knn_list)
    plt.legend()
    plt.savefig(os.path.join(
        ckpt_folder, 'epoch%s_bs%s_loss.png' % (args.epochs, args.batch_size)))

    # save your improved network
    checkpoint_path = os.path.join(
        ckpt_folder, 'renet_epoch%s.pth.tar' % str(args.epochs))
    torch.save(resnet.state_dict(), checkpoint_path)
    checkpoint_path = os.path.join(
        ckpt_folder, 'byol_epoch%s.pth.tar' % str(args.epochs))
    torch.save(model.state_dict(), checkpoint_path)



if __name__ == '__main__':
    main()