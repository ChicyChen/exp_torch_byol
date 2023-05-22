import os
import sys
import argparse
sys.path.append("/home/siyich/byol-pytorch/byol_3d")
sys.path.append("/home/siyich/byol-pytorch/utils")
from byol_3d import BYOL

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms as T
import torch.nn.functional as F

from dataloader_3d import get_data_ucf, UCF101
from torch.utils.data import DataLoader

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

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
parser.add_argument('--gpu_num', default=2, type=int)
parser.add_argument('--batch_size', default=16, type=int)

parser.add_argument('--asym_loss', action='store_true')
parser.add_argument('--closed_loop', action='store_true')
parser.add_argument('--useode', action='store_true')
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--rtol', default=1e-4, type=float, help='rtol of ode solver')
parser.add_argument('--atol', default=1e-4, type=float, help='atol of ode solver')
# parser.add_argument('--odenorm', action='store_true')

parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')

parser.add_argument('--num_seq', default=1, type=int)
parser.add_argument('--seq_len', default=4, type=int)
parser.add_argument('--downsample', default=4, type=int)
parser.add_argument('--num_aug', default=2, type=int)

parser.add_argument('--backbone', default='r3d18', type=str, help='r3d18, r2118')

parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')


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
    cuda = torch.device('cuda')
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

    args = parser.parse_args()

    
    ckpt_folder='/home/siyich/byol-pytorch/checkpoints_ddp/3dbase_ode%s_closed%s_ucf101_lr%s_wd%s' % (args.useode, args.closed_loop, args.lr, args.wd)
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)
    args.ckpt_folder = ckpt_folder

    logging.basicConfig(filename=os.path.join(ckpt_folder, 'byol_train.log'), level=logging.INFO)
    logging.info('Started')

    #########################################################
    args.world_size = args.gpu_num              
    os.environ['MASTER_ADDR'] = 'localhost'              
    os.environ['MASTER_PORT'] = '8800'                      
    mp.spawn(main_worker, nprocs=args.gpu_num, args=(args,))         
    #########################################################



def main_worker(gpu, args):
    ckpt_folder = args.ckpt_folder
    logging.basicConfig(filename=os.path.join(ckpt_folder, 'byol_train.log'), level=logging.INFO)
    ############################################################
    rank = gpu

    print('find rank', rank)
    print('world size', args.world_size) 
    print(os.environ['MASTER_ADDR'] == 'localhost')
    print(os.environ['MASTER_PORT'] == '8800')

    dist.init_process_group(                                   
    	backend='gloo',                                         
   		init_method='env://',                                   
    	world_size=args.world_size,                              
    	rank=rank                                               
    )     

    print('process initialzed')                                                     
    ############################################################

    torch.cuda.set_device(gpu)

    # Check whether is using GPU
    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print('using CPU, this will be slow')
    else:
        print('using GPU')

    if args.backbone == 'r3d18':
        resnet = models.video.r3d_18()
        # modify model
        resnet.stem[0] = torch.nn.Conv3d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # resnet.maxpool = torch.nn.Identity()
    elif args.backbone == 'r2118':
        resnet = models.video.r2plus1d_18()
        # ??resnet.stem[0] = torch.nn.Conv3d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # ??resnet.stem[3] = torch.nn.Conv3d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    print('resnet created')

    model = BYOL(
        resnet,
        clip_size = 8,
        image_size = 128,
        hidden_layer = 'avgpool',
        projection_size = 256,
        projection_hidden_size = 4096,
        asym_loss = args.asym_loss,
        closed_loop = args.closed_loop,
        useode = args.useode,
        adjoint = args.adjoint,
        rtol = args.rtol,
        atol = args.atol,
        odenorm = None
    )

    print('model created')
    
    model.cuda(gpu)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    if args.pretrain:
        pretrain_path = os.path.join(args.pretrain_folder, 'byol_epoch%s.pth.tar' % args.start_epoch)
        model.load_state_dict(torch.load(pretrain_path)) # load model

    ###############################################################
    # Wrap the model
    model = DDP(model, device_ids=[gpu], find_unused_parameters=True)
    ###############################################################

    print('model wrapped')

    train_loader = get_data_ucf(batch_size=args.batch_size, 
                                mode='train', 
                                transform=default_transform(), 
                                transform2=default_transform(),
                                seq_len=args.seq_len, 
                                num_seq=args.num_seq, 
                                downsample=args.downsample,
                                num_aug=args.num_aug,
                                ddp=True)
    test_loader = get_data_ucf(batch_size=args.batch_size, 
                                mode='val',
                                transform=default_transform(), 
                                transform2=default_transform(),
                                seq_len=args.seq_len, 
                                num_seq=args.num_seq, 
                                downsample=args.downsample,
                                num_aug=args.num_aug,
                                ddp=True)


    print('data loaded')
    
    train_loss_list = []
    test_loss_list = []
    epoch_list = range(args.start_epoch, args.epochs)
    lowest_loss = np.inf
    best_epoch = 0

    print('start training')

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

        if (i < 10 or (i+1)%10 == 0) and rank == 0:
            # save your improved network
            checkpoint_path = os.path.join(
                ckpt_folder, 'resnet_epoch%s.pth.tar' % str(i+1))
            torch.save(resnet.state_dict(), checkpoint_path)
            checkpoint_path = os.path.join(
                ckpt_folder, 'byol_epoch%s.pth.tar' % str(i+1))
            torch.save(model.state_dict(), checkpoint_path)
            

    print('finish training')

    logging.info('Training from ep %d to ep %d finished' %
          (args.start_epoch, args.epochs))
    logging.info('Best epoch: %s' % best_epoch)

    # plot training process
    plt.plot(epoch_list, train_loss_list, label = 'train')
    # if not args.no_val:
    plt.plot(epoch_list, test_loss_list, label = 'val')
    plt.title('Train and test loss')
    plt.legend()
    plt.savefig(os.path.join(
        ckpt_folder, 'epoch%s_bs%s_loss.png' % (args.epochs, args.batch_size)))

    if rank == 0:
        # save your improved network
        checkpoint_path = os.path.join(
            ckpt_folder, 'resnet_epoch%s.pth.tar' % str(args.epochs))
        torch.save(resnet.state_dict(), checkpoint_path)
        checkpoint_path = os.path.join(
            ckpt_folder, 'byol_epoch%s.pth.tar' % str(args.epochs))
        torch.save(model.state_dict(), checkpoint_path)



if __name__ == '__main__':
    main()