import os
import sys
from importlib import reload
# reload(sys)
# sys.setdefaultencoding('utf-8')
import argparse
sys.path.append("/home/siyich/byol-pytorch/byol_3d")
sys.path.append("/home/siyich/byol-pytorch/utils")
from byolode_3d import BYOL_ODE

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
parser.add_argument('--pretrain_other', action='store_true')
parser.add_argument('--gpu', default='0,1,2,3', type=str)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--ema', default=0.99, type=float, help='EMA')

parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')

parser.add_argument('--num_seq', default=2, type=int)
parser.add_argument('--seq_len', default=4, type=int)
parser.add_argument('--downsample', default=4, type=int)
parser.add_argument('--num_aug', default=1, type=int)

parser.add_argument('--asym_loss', action='store_true')
parser.add_argument('--ordered', action='store_true') # not fully supported
parser.add_argument('--closed_loop', action='store_true')
parser.add_argument('--sequential', action='store_true')

parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--rtol', default=1e-4, type=float, help='rtol of ode solver')
parser.add_argument('--atol', default=1e-4, type=float, help='atol of ode solver')
parser.add_argument('--tstep', default=1.0, type=float)
# parser.add_argument('--odenorm', action='store_true')

parser.add_argument('--pred_hidden', default=4096, type=int)
parser.add_argument('--pred_layer', default=2, type=int)
parser.add_argument('--solver', default='dopri5', type=str)

parser.add_argument('--mse_l', default=1, type=float)
parser.add_argument('--std_l', default=0, type=float)
parser.add_argument('--cov_l', default=0, type=float)

parser.add_argument('--random', action='store_true', help='whether use random data')
parser.add_argument('--no_mom', action='store_true')
parser.add_argument('--no_projector', action='store_true')
parser.add_argument('--use_simsiam_mlp', action='store_true')
parser.add_argument('--ode_lr_frac', default = 1.0, type=float)

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
    # global have_print

    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.
    num_batches = len(train_loader)

    for data in train_loader:
        video, label = data # B, C, T, H, W
        label = label.to(cuda)

        if not args.sequential:
            images = video[:, :, :args.seq_len, :, :]
            images = images.to(cuda)
            for i in range(args.num_seq-1):
                # index = i*args.seq_len
                index2 = (i+1)*args.seq_len
                index3 = (i+2)*args.seq_len
                # images = video[:, :, index:index2, :, :]
                images2 = video[:, :, index2:index3, :, :]
                # images = images.to(cuda)
                images2 = images2.to(cuda)

                # print(idxs.size())
                # integration_time_f = torch.tensor([0, i+1], dtype=torch.float32)
                # integration_time_b = torch.tensor([i+1, 0], dtype=torch.float32)
                integration_time_f = [0, (i+1)*args.tstep]
                integration_time_b = [(i+1)*args.tstep, 0]

                # if not have_print:
                #     print(images.size(), images2.size())
                #     have_print = True

                optimizer.zero_grad()
                loss = model(images, images2, integration_time_f=integration_time_f, integration_time_b=integration_time_b)
                if train: # train separately for each step
                    loss.sum().backward()
                    optimizer.step()
                    # EMA update
                    if not args.no_mom:
                        model.module.update_moving_average()
                else:
                    pass
                total_loss += loss.sum().item() / (args.num_seq-1)
        
        else:
            images = video[:, :, :args.seq_len, :, :]
            images = images.to(cuda)
            images2_list = []
            for i in range(args.num_seq-1):
                # index = i*args.seq_len
                index2 = (i+1)*args.seq_len
                index3 = (i+2)*args.seq_len
                # images = video[:, :, index:index2, :, :]
                images2 = video[:, :, index2:index3, :, :]
                # images = images.to(cuda)
                images2 = images2.to(cuda)
                images2_list.append(images2)
            # integration_time_f = torch.tensor(np.arange(0, args.num_seq), dtype=torch.float32)
            # integration_time_b = torch.tensor(np.arange(args.num_seq-1, -1, -1), dtype=torch.float32)
            # integration_time_f = torch.tensor([0, 1], dtype=torch.float32)
            # integration_time_b = torch.tensor([1, 0], dtype=torch.float32)
            integration_time_f = np.arange(0, args.num_seq)*args.tstep
            integration_time_b = np.arange(args.num_seq-1, -1, -1)*args.tstep

            # print(integration_time_f, integration_time_b)

            optimizer.zero_grad()
            loss = model(images, images2_list, 
                         integration_time_f=integration_time_f, 
                         integration_time_b=integration_time_b,
                         sequential=True)
            if train: # train separately for each step
                loss.sum().backward()
                optimizer.step()
                # EMA update
                if not args.no_mom: 
                    model.module.update_moving_average()
            else:
                pass
            total_loss += loss.sum().item() / (args.num_seq-1)
        
        # print("done one batch.")
    
    return total_loss/num_batches

def main():
    torch.manual_seed(233)
    np.random.seed(233)

    # global have_print
    # have_print = False

    global args
    args = parser.parse_args()

    # ckpt_folder='/home/siyich/byol-pytorch/checkpoints_ode_ema%s/t%s_mse%s_std%s_cov%s_solver%s_3dseq_ode_po%s_predln%s_hid%s_adj%s_%s_asym%s_closed%s_sequential%s_ordered%s_ucf101_lr%s_wd%s_rt%s_at%s' % (args.ema, args.tstep, args.mse_l, args.std_l, args.cov_l, args.solver, args.pretrain_other, args.pred_layer, args.pred_hidden, args.adjoint, args.num_seq, args.asym_loss, 
    # args.closed_loop, args.sequential, args.ordered, args.lr, args.wd, args.rtol, args.atol)
    if args.no_mom:
        args.ema = 1.0

    ckpt_folder='/home/siyich/byol-pytorch/checkpoints_ode_lr%s_ema%s_pj%s/ucf101_t%s_mse%s_std%s_cov%s_predln%s_hid%s_ns%s_lr%s_wd%s' \
    % (args.ode_lr_frac, args.ema, not args.no_projector, args.tstep, args.mse_l, args.std_l, args.cov_l, args.pred_layer, args.pred_hidden, args.num_seq, args.lr, args.wd)

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

    model = BYOL_ODE(
        resnet,
        clip_size = args.seq_len,
        image_size = 128,
        hidden_layer = 'avgpool',
        projection_size = 256,
        projection_hidden_size = args.pred_hidden,
        moving_average_decay = args.ema,
        asym_loss = args.asym_loss,
        closed_loop = args.closed_loop,
        adjoint = args.adjoint,
        rtol = args.rtol,
        atol = args.atol,
        odenorm = None,
        num_layer=args.pred_layer,
        mse_l = args.mse_l,
        std_l = args.std_l,
        cov_l = args.cov_l,
        use_momentum = not args.no_mom,
        use_projector = not args.no_projector,
        use_simsiam_mlp = args.use_simsiam_mlp
    )

    
    model = nn.DataParallel(model)
    model = model.to(cuda)

    if args.pretrain:
        pretrain_path = os.path.join(args.pretrain_folder, 'byolode_epoch%s.pth.tar' % args.start_epoch)
        model.load_state_dict(torch.load(pretrain_path)) # load model
    if args.pretrain_other: # only resnet is pretrained
        pretrain_path = os.path.join(args.pretrain_folder, 'resnet_epoch%s.pth.tar' % args.start_epoch)
        resnet.load_state_dict(torch.load(pretrain_path)) # load model

    # if args.freeze_backbone:
    #     if args.freeze_backbone_path != '':
    #         backbone_path = args.freeze_backbone_path
    #         resnet.load_state_dict(torch.load(backbone_path)) # load resnet
    #     for param in resnet.parameters():
    #         param.requires_grad = False

    params = []
    for name, param in model.named_parameters():
        if 'predict' in name:
            params.append({'params': param, 'lr': args.lr * args.ode_lr_frac})
        else:
            params.append({'params': param})
    print(len(params))


    print('\n===========Check Grad============')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.requires_grad)
    print('=================================\n')

    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)

    train_loader = get_data_ucf(batch_size=args.batch_size, 
                                mode='train', 
                                transform=default_transform(), 
                                transform2=default_transform(),
                                seq_len=args.seq_len, 
                                num_seq=args.num_seq, 
                                downsample=args.downsample,
                                num_aug=args.num_aug,
                                random=args.random)
    test_loader = get_data_ucf(batch_size=args.batch_size, 
                                mode='val',
                                transform=default_transform(), 
                                transform2=default_transform(),
                                seq_len=args.seq_len, 
                                num_seq=args.num_seq, 
                                downsample=args.downsample,
                                num_aug=args.num_aug,
                                random=args.random)
    
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

        if (i+1)%10 == 0 or i<20:
            # save your improved network
            checkpoint_path = os.path.join(
                ckpt_folder, 'resnet_epoch%s.pth.tar' % str(i+1))
            torch.save(resnet.state_dict(), checkpoint_path)
            checkpoint_path = os.path.join(
                ckpt_folder, 'byolode_epoch%s.pth.tar' % str(i+1))
            torch.save(model.state_dict(), checkpoint_path)
            


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

    # save your improved network
    checkpoint_path = os.path.join(
        ckpt_folder, 'renet_epoch%s.pth.tar' % str(args.epochs))
    torch.save(resnet.state_dict(), checkpoint_path)
    checkpoint_path = os.path.join(
        ckpt_folder, 'byolode_epoch%s.pth.tar' % str(args.epochs))
    torch.save(model.state_dict(), checkpoint_path)



if __name__ == '__main__':
    main()