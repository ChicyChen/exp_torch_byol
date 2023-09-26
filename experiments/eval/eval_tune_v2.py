import os
import sys
import argparse
sys.path.append("/home/siyich/byol-pytorch/byol_3d")
sys.path.append("/home/siyich/byol-pytorch/evaluation_3d")
sys.path.append("/home/siyich/byol-pytorch/utils")
from byol_3d import BYOL
from byolode_3d import BYOL_ODE
from finetune_v2 import Fine_Tune

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
from torch.optim.lr_scheduler import LinearLR, MultiStepLR
from distributed_utils import init_distributed_mode

# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/eval/eval_tune_v2.py

parser = argparse.ArgumentParser()
parser.add_argument('--frame_root', default='/data', type=str,
                    help='root folder to store data like UCF101/..., better to put in servers SSD \
                    default path is mounted from data server for the home directory')

parser.add_argument('--epochs', default=120, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')

parser.add_argument('--pretrain_path', default='', type=str)
parser.add_argument('--pretrain', action='store_true')

parser.add_argument('--gpu', default='0,1,2,3,4,5,6,7', type=str)
parser.add_argument('--batch_size', default=128, type=int)

parser.add_argument('--ckpt_folder', default='/home/siyich/byol-pytorch/checkpoints_bad/3dseq_ucf101_lr0.0001_wd1e-05', type=str)
parser.add_argument('--epoch_num', default=100, type=int)

parser.add_argument('--hmdb', action='store_true')
parser.add_argument('--random', action='store_true')
parser.add_argument('--input_dim', default=512, type=int)
parser.add_argument('--class_num', default=101, type=int)

parser.add_argument('--lr', default=0.16, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')

parser.add_argument('--dropout', default=0.5, type=float)
parser.add_argument('--num_layer', default=1, type=int)

parser.add_argument('--num_seq', default=1, type=int)
parser.add_argument('--seq_len', default=16, type=int)
parser.add_argument('--downsample', default=4, type=int)
parser.add_argument('--num_aug', default=1, type=int)

parser.add_argument('--dim', default=150, type=int)
parser.add_argument('--linear', action='store_true')
parser.add_argument('--r21d', action='store_true')

# Running
parser.add_argument("--num-workers", type=int, default=128)
parser.add_argument('--device', default='cuda',
                    help='device to use for training / testing')
# Distributed
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
# parser.add_argument('--local-rank', default=-1, type=int)
parser.add_argument('--dist-url', default='tcp://localhost:12345',
                    help='url used to set up distributed training')

# parser.add_argument('--ode', action='store_true')



def test_transform():
    transform = transforms.Compose([
        # RandomCrop(size=128, consistent=True),
        RandomSizedCrop(size=128, consistent=True), # Siyi: strong crop
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


def tune_train(predict_model, train_loader, criterion, optimizer, gpu):
    predict_model.train()

    total_loss = 0.
    total_acc = 0.
    num_batches = len(train_loader)

    for data in train_loader:
        images, label = data
        images = images.to(gpu) # B, C, T, H, W
        label = label.to(gpu).squeeze(1)
        output = predict_model(images)

        # images_diff = images[:,:,1:,:,:] - images[:,:,:-1,:,:]
        # output_diff = predict_model(images_diff)

        # print(output.size(), label.size())

        loss = criterion(output, label)
        # loss = criterion(output, label) + criterion(output_diff, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = calc_accuracy(output, label)
        total_loss += loss.sum().item()
        total_acc += acc.cpu().detach().numpy()

    mean_loss = total_loss/num_batches
    mean_acc = total_acc/num_batches
    return mean_loss, mean_acc


def tune_eval(predict_model, test_loader, gpu):
    predict_model.eval()
    acc_list =[]
    for data in test_loader:
        images, label = data
        images = images.to(gpu)
        label = label.to(gpu)
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

    torch.backends.cudnn.benchmark = True
    init_distributed_mode(args)
    print(args)
    gpu = torch.device(args.device)

    ckpt_folder = args.ckpt_folder
    ckpt_path = os.path.join(ckpt_folder, 'resnet_epoch%s.pth.tar' % args.epoch_num)

    if not args.hmdb:
        args.class_num = 101
        if args.random:
            tune_folder = os.path.join(ckpt_folder, 'ucf_tune_epoch0') 
        else:
            tune_folder = os.path.join(ckpt_folder, 'ucf_tune_epoch%s' % args.epoch_num)
    else:
        args.class_num = 51
        if args.random:
            tune_folder = os.path.join(ckpt_folder, 'hmdb_tune_epoch0') 
        else:
            tune_folder = os.path.join(ckpt_folder, 'hmdb_tune_epoch%s' % args.epoch_num)
    tune_folder = tune_folder+'_lr%s_wd%s_mo%s_dr%s_bs%s_ds%s_sl%s_dim%s_linear%s_all' % (args.lr, args.wd, args.momentum, args.dropout, args.batch_size, args.downsample, args.seq_len, args.dim, args.linear)
    
    if args.rank == 0:
        if not os.path.exists(tune_folder):
            os.makedirs(tune_folder)
        logging.basicConfig(filename=os.path.join(tune_folder, 'ucf_tune.log'), level=logging.INFO)
        logging.info('Started')

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # global cuda
    # cuda = torch.device('cuda')

    if args.r21d:
        # model_name = 'r21d18'
        resnet = models.video.r2plus1d_18()
    else:
        # model_name = 'r3d18'
        resnet = models.video.r3d_18()

    if (not args.random) and (not args.pretrain):
        resnet.load_state_dict(torch.load(ckpt_path)) # load model
        # logging.info(ckpt_path)
        # logging.info(f"finetuning performed after ssl")
    # else:
        # logging.info(f"finetuning performed with random weight")
    
    resnet.fc = torch.nn.Identity()

    predict_model = Fine_Tune(resnet, args.input_dim, args.class_num, args.dropout, args.num_layer).cuda(gpu)
    predict_model = nn.SyncBatchNorm.convert_sync_batchnorm(predict_model)
    predict_model = torch.nn.parallel.DistributedDataParallel(predict_model, device_ids=[gpu])
    # predict_model = torch.nn.parallel.DistributedDataParallel(predict_model, device_ids=[gpu], find_unused_parameters=True)

    
    # predict_model = nn.DataParallel(predict_model)
    # predict_model = predict_model.to(cuda)

    if args.pretrain:
        predict_model.load_state_dict(torch.load(args.pretrain_path)) # load model
        # logging.info(args.pretrain_path)

    params = []
    for name, param in predict_model.named_parameters():
        if 'linear_pred' in name:
            params.append({'params': param})
        else:
            # params.append({'params': param})
            if args.linear:
                params.append({'params': param, 'lr': 0.0})
            else:
                # params.append({'params': param, 'lr': args.lr/10})
                params.append({'params': param})

    # print(len(params))
    print('\n===========Check Grad============')
    for name, param in predict_model.named_parameters():
        print(name, param.requires_grad)
    print('=================================\n')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
    # scheduler = LinearLR(optimizer, start_factor=0.3, total_iters=10)
    scheduler = MultiStepLR(optimizer, milestones=[60,80,100], gamma=0.1)

    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size

    if not args.hmdb:
        logging.info(f"finetuning performed on ucf")
        train_loader = get_data_ucf(batch_size=per_device_batch_size, 
                                    mode='train', 
                                    transform=default_transform(), 
                                    transform2=default_transform(),
                                    seq_len=args.seq_len, 
                                    num_seq=args.num_seq, 
                                    downsample=args.downsample,
                                    num_aug=args.num_aug,
                                    frame_root=args.frame_root,
                                    dim=args.dim,
                                    )
        test_loader = get_data_ucf(batch_size=per_device_batch_size, 
                                    mode='val', 
                                    transform=default_transform(), 
                                    transform2=default_transform(),
                                    seq_len=args.seq_len, 
                                    num_seq=args.num_seq, 
                                    downsample=args.downsample,
                                    num_aug=args.num_aug,
                                    frame_root=args.frame_root,
                                    dim=args.dim,
                                    )
    else:
        logging.info(f"finetuning performed on hmdb")
        train_loader = get_data_hmdb(batch_size=per_device_batch_size, 
                                    mode='train', 
                                    transform=default_transform(), 
                                    transform2=default_transform(),
                                    seq_len=args.seq_len, 
                                    num_seq=args.num_seq, 
                                    downsample=args.downsample,
                                    num_aug=args.num_aug,
                                    frame_root=args.frame_root,
                                    # dim=240,
                                    )
        test_loader = get_data_hmdb(batch_size=per_device_batch_size, 
                                    mode='val', 
                                    transform=default_transform(), 
                                    transform2=default_transform(),
                                    seq_len=args.seq_len, 
                                    num_seq=args.num_seq, 
                                    downsample=args.downsample,
                                    num_aug=args.num_aug,
                                    frame_root=args.frame_root,
                                    # dim=240,
                                    )
        

    # scaler = torch.cuda.amp.GradScaler()

    best_acc = 0
    epoch_list = range(args.start_epoch, args.epochs)
    for i in epoch_list:
        train_loss, train_acc = tune_train(predict_model, train_loader, criterion, optimizer, gpu)
        test_acc = tune_eval(predict_model, test_loader, gpu)
        scheduler.step()

        if args.rank == 0:

            if test_acc > best_acc:
                best_acc = test_acc
            print('Epoch: %s, Train loss: %s' % (i, train_loss))
            print('Epoch: %s, Train acc: %s' % (i, train_acc))
            print('Epoch: %s, Test acc: %s' % (i, test_acc))
            logging.info('Epoch: %s, Train loss: %s' % (i, train_loss))
            logging.info('Epoch: %s, Train acc: %s' % (i, train_acc))
            logging.info('Epoch: %s, Test acc: %s' % (i, test_acc))

            if (i+1)%5 == 0 or i >= 80:
                # save your improved network
                checkpoint_path = os.path.join(
                    tune_folder, 'tune_epoch%s.pth.tar' % str(i+1))
                torch.save(predict_model.state_dict(), checkpoint_path)
    
    if args.rank == 0:
        print('Finetune Acc: %s \n' % best_acc)
        logging.info('Finetune Acc: %s \n' % best_acc)


if __name__ == '__main__':
    main()