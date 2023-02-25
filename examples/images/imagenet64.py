import os
import sys
import argparse
sys.path.append("/home/siyich/byol-pytorch/byol_pytorch")
from byol_pytorch import BYOL
from knn import *

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms as T
import torch.nn.functional as F

from dataloader import ImageNet32, ImageNet64
from torch.utils.data import DataLoader

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import logging
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpu', default='0,1,2,3', type=str)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--knn', action='store_true')


def get_ImageNet64_dataloader(batch_size, train=True):
    transform = T.Compose([
            T.ToTensor(),          
        ])
    dataset = ImageNet64(root="/home/siyich/byol-pytorch/data/", train=train, transform=transform)
    return DataLoader(dataset=dataset, batch_size=batch_size, num_workers=4, drop_last=True)

def train_one_epoch(model, train_loader, optimizer, train=True):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.
    num_batches = len(train_loader)

    for data in train_loader:
        images, label = data
        images = images.to(cuda)
        label = label.to(cuda)
        # print(images.size(), label.size())
        optimizer.zero_grad()
        loss = model(images)
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

    ckpt_folder='/home/siyich/byol-pytorch/checkpoints/toy_imagenet64'

    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)

    logging.basicConfig(filename=os.path.join(ckpt_folder, 'byol_train.log'), level=logging.INFO)
    logging.info('Started')

    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    global cuda
    cuda = torch.device('cuda')

    resnet = models.resnet18(pretrained=False)
    # modify model
    resnet.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    resnet.maxpool = torch.nn.Identity()

    model = BYOL(
        resnet,
        image_size = 64,
        hidden_layer = 'avgpool',
        projection_size = 256,
        projection_hidden_size = 4096,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    model = nn.DataParallel(model)
    model = model.to(cuda)

    train_loader = get_ImageNet64_dataloader(batch_size=args.batch_size, train=True)
    test_loader = get_ImageNet64_dataloader(batch_size=args.batch_size, train=False)
    
    train_loss_list = []
    test_loss_list = []
    train_knns = []
    val_knns = []
    epoch_list = range(args.start_epoch, args.epochs)
    knn_list = list(range(args.start_epoch, args.epochs,10))
    if args.epochs - 1 not in knn_list:
        knn_list.append(args.epochs - 1)
    lowest_loss = np.inf
    best_epoch = 0

    for i in epoch_list:
        train_loss = train_one_epoch(model, train_loader, optimizer)
        test_loss = train_one_epoch(model, train_loader, optimizer, False)
        if test_loss < lowest_loss:
            lowest_loss = test_loss
            best_epoch = i + 1
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        print('Epoch: %s, Train loss: %s' % (i, train_loss))
        print('Epoch: %s, Test loss: %s' % (i, test_loss))
        logging.info('Epoch: %s, Train loss: %s' % (i, train_loss))
        logging.info('Epoch: %s, Test loss: %s' % (i, test_loss))

        if args.knn and i in knn_list:
            # KNN evaluation
            ssl_evaluator = KNN(model=resnet, k=1, device=cuda)
            train_acc, val_acc = ssl_evaluator.fit(train_loader, test_loader)
            train_knns.append(train_acc)
            val_knns.append(val_acc)
            print(f"k-nn accuracy k= {ssl_evaluator.k} for train split: {train_acc}")
            print(f"k-nn accuracy k= {ssl_evaluator.k} for val split: {val_acc} \n")
            print('-----------------')
            logging.info(f"k-nn accuracy k= {ssl_evaluator.k} for train split: {train_acc}")
            logging.info(f"k-nn accuracy k= {ssl_evaluator.k} for val split: {val_acc} \n")
            logging.info('-----------------')

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
        ckpt_folder, 'epoch%s_bs%s_loss.png' % (args.epochs+1, args.batch_size)))

    if args.knn:
        plt.plot()
        plt.plot(knn_list, train_knns, label = 'train')
        # if not args.no_val:
        plt.plot(knn_list, val_knns, label = 'val')
        plt.title('Train and test knn')
        # plt.xticks(knn_list, knn_list)
        plt.legend()
        plt.savefig(os.path.join(
            ckpt_folder, 'epoch%s_bs%s_knn.png' % (args.epochs+1, args.batch_size)))

    # save your improved network
    checkpoint_path = os.path.join(
        ckpt_folder, 'epoch%s.pth.tar' % str(args.epochs+1))
    torch.save(resnet.state_dict(), checkpoint_path)



if __name__ == '__main__':
    main()