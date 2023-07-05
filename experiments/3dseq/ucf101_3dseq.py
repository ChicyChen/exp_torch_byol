import os
import sys
from importlib import reload
# reload(sys)
# sys.setdefaultencoding('utf-8')
import argparse
sys.path.append("/home/siyich/byol-pytorch/byol_3d")
sys.path.append("/home/siyich/byol-pytorch/utils")
from byolseq_3d import BYOL_SEQ

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

parser.add_argument('--frame_root', default='/home/siyich/Datasets/Videos', type=str,
                    help='root folder to store data like UCF101/..., better to put in servers SSD \
                    default path is mounted from data server for the home directory')
                    
parser.add_argument('--gpu', default='0,1,2,3', type=str)

parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--pretrain_folder', default='', type=str)
parser.add_argument('--pretrain', action='store_true')

parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')

parser.add_argument('--num_seq', default=2, type=int)
parser.add_argument('--seq_len', default=8, type=int)
parser.add_argument('--downsample', default=4, type=int)
parser.add_argument('--num_aug', default=1, type=int)
parser.add_argument('--inter_len', default=0, type=int)    # does not need to be positive

parser.add_argument('--sym_loss', action='store_true')
parser.add_argument('--closed_loop', action='store_true')
parser.add_argument('--sequential', action='store_true')

parser.add_argument('--pred_hidden', default=4096, type=int)
parser.add_argument('--projection', default=256, type=int)
parser.add_argument('--pred_layer', default=2, type=int)

parser.add_argument('--mse_l', default=1.0, type=float)
parser.add_argument('--std_l', default=0.0, type=float)
parser.add_argument('--cov_l', default=0.0, type=float)

parser.add_argument('--ema', default=0.99, type=float, help='EMA')
parser.add_argument('--no_mom', action='store_true')
parser.add_argument('--no_projector', action='store_true')
parser.add_argument('--use_simsiam_mlp', action='store_true')

parser.add_argument('--bn_last', action='store_true')
parser.add_argument('--pred_bn_last', action='store_true')



def train_one_epoch(model, train_loader, optimizer, train = True, num_aug = 1):
    # global have_print

    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.
    num_batches = len(train_loader)

    for data in train_loader:
        if num_aug == 1:
            video, label = data # B, C, T, H, W
        else:
            video, video2, label = data # B, C, T, H, W
        label = label.to(cuda)

        if not args.sequential:
            for i in range(args.num_seq-1):
                index1 = i*args.seq_len
                index2 = (i+1)*args.seq_len
                index3 = (i+2)*args.seq_len
                images = video[:, :, index1:index2, :, :]
                if num_aug == 1: # notice, even consistent crop etc, still inconsistent colorjitter, grey etc
                    images2 = video[:, :, index2:index3, :, :]
                else:
                    images2 = video2[:, :, index2:index3, :, :]
                images = images.to(cuda)
                images2 = images2.to(cuda)

                # if not have_print:
                #     print(images.size(), images2.size())
                #     have_print = True

                optimizer.zero_grad()
                loss = model(images, images2)

                if train: # train separately for each step
                    loss.sum().backward()
                    optimizer.step()
                    # EMA update
                    model.module.update_moving_average()
                else:
                    pass
                total_loss += loss.sum().item() / (args.num_seq-1)
        
        else:
            # TODO: not supporting different augmentations
            if num_aug > 1:
                raise ValueError('Not supporting > 1 augmentations in sequential mode')
                
            video = video.to(cuda)
            images_list = []
            for i in range(args.num_seq):
                # index = i*args.seq_len
                index1 = i*args.seq_len
                index2 = (i+1)*args.seq_len
                images = video[:, :, index1:index2, :, :]
                images = images.to(cuda)
                images_list.append(images)
            images = torch.stack(images_list, 1) # B, N, C, T, H, W: parallel split along the first axis

            # if not have_print:
            #     print(images.size())
            #     have_print = True

            optimizer.zero_grad()
            loss = model(images, None, sequential=True)

            if train: # train together after predicting all
                loss.sum().backward()
                optimizer.step()
                # EMA update
                model.module.update_moving_average()
            else:
                pass
            total_loss += loss.sum().item() / (args.num_seq-1)
        
        # print("done one batch.")
    
    return total_loss/num_batches


def main():
    torch.manual_seed(233)
    np.random.seed(233)

    global have_print
    have_print = False

    global args
    args = parser.parse_args()

    ckpt_folder='/home/siyich/byol-pytorch/checkpoints_seq_na%s_bnl%s_pbnl%s_il%s_ns%s/ema%s_mse%s_std%s_cov%s_hid%s_prj%s_sym%s_closed%s_sequential%s_bs%s_lr%s_wd%s' \
        % (args.num_aug, args.bn_last, args.pred_bn_last, args.inter_len, args.num_seq, args.ema, args.mse_l, args.std_l, args.cov_l, args.pred_hidden, args.projection, args.sym_loss, args.closed_loop, args.sequential, args.batch_size, args.lr, args.wd)

    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)

    logging.basicConfig(filename=os.path.join(ckpt_folder, 'byol_train.log'), level=logging.INFO)
    logging.info('Started')

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    global cuda
    cuda = torch.device('cuda')

    resnet = models.video.r3d_18()
    # modify model
    # resnet.stem[0] = torch.nn.Conv3d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # resnet.maxpool = torch.nn.Identity()

    model = BYOL_SEQ(
        resnet,
        clip_size = args.seq_len,
        image_size = 128,
        hidden_layer = 'avgpool',
        projection_size = args.projection,
        projection_hidden_size = args.pred_hidden,
        num_layer=args.pred_layer,
        moving_average_decay = args.ema,
        asym_loss = not args.sym_loss,
        closed_loop = args.closed_loop,
        mse_l = args.mse_l,
        std_l = args.std_l,
        cov_l = args.cov_l,
        use_momentum = not args.no_mom,
        use_projector = not args.no_projector,
        use_simsiam_mlp = args.use_simsiam_mlp,
        bn_last = args.bn_last,
        pred_bn_last = args.pred_bn_last
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
                                seq_len=args.seq_len, 
                                num_seq=args.num_seq, 
                                downsample=args.downsample,
                                num_aug=args.num_aug,
                                inter_len=args.inter_len,
                                frame_root=args.frame_root,
                                )
    test_loader = get_data_ucf(batch_size=args.batch_size, 
                                mode='val',
                                transform=default_transform(), 
                                transform2=default_transform(),
                                seq_len=args.seq_len, 
                                num_seq=args.num_seq, 
                                downsample=args.downsample,
                                num_aug=args.num_aug,
                                inter_len=args.inter_len,
                                frame_root=args.frame_root,
                                )
    
    train_loss_list = []
    test_loss_list = []
    epoch_list = range(args.start_epoch, args.epochs)
    lowest_loss = np.inf
    best_epoch = 0

    for i in epoch_list:
        train_loss = train_one_epoch(model, train_loader, optimizer, num_aug = args.num_aug)
        test_loss = train_one_epoch(model, test_loader, optimizer, False, num_aug = args.num_aug)
        if test_loss < lowest_loss:
            lowest_loss = test_loss
            best_epoch = i + 1
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        print('Epoch: %s, Train loss: %s' % (i, train_loss))
        print('Epoch: %s, Test loss: %s' % (i, test_loss))
        logging.info('Epoch: %s, Train loss: %s' % (i, train_loss))
        logging.info('Epoch: %s, Test loss: %s' % (i, test_loss))

        if (i+1)%10 == 0 or i < 20:
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