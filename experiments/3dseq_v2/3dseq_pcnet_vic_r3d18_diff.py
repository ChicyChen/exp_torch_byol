import os
import sys
from importlib import reload
# reload(sys)
# sys.setdefaultencoding('utf-8')
import argparse
sys.path.append("/home/siyich/byol-pytorch/pcnet_3d")
sys.path.append("/home/siyich/byol-pytorch/utils")
# sys.path.append("/home/siyich/byol-pytorch/resnet")
from pcnet_vic import PCNET_VIC
# from resnet_modify import r3d_18_slow

import random
import math
import numpy as np
import torch
from torch import nn, optim
from torchvision import models
from torchvision import transforms as T
import torch.nn.functional as F

from dataloader_v2 import get_data_ucf, get_data_k400, get_data_mk200, get_data_mk400, get_data_minik
from torch.utils.data import DataLoader

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import logging
import time
import matplotlib.pyplot as plt

from augmentation import *
from distributed_utils import init_distributed_mode

# python -m torch.distributed.launch --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --sym_loss
# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --sym_loss

# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18_diff.py --epochs 100 --batch_size 256 --sym_loss --base_lr 2.4 --projection 8192 --proj_hidden 8192 --pred_layer 0 --proj_layer 3 --cov_l 0.04 --std_l 1.0 --spa_l 0.0 --minik

parser = argparse.ArgumentParser()

parser.add_argument('--frame_root', default='/data', type=str,
                    help='root folder to store data like UCF101/..., better to put in servers SSD \
                    default path is mounted from data server for the home directory')
# --frame_root /data
                    
parser.add_argument('--gpu', default='0,1,2,3,4,5,6,7', type=str)

parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--pretrain_folder', default='', type=str)
parser.add_argument('--pretrain', action='store_true')

parser.add_argument('--batch_size', default=256, type=int)
# parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-6, type=float, help='weight decay')

parser.add_argument('--random', action='store_true')
parser.add_argument('--num_seq', default=2, type=int)
parser.add_argument('--seq_len', default=8, type=int)
parser.add_argument('--downsample', default=3, type=int)
parser.add_argument('--inter_len', default=0, type=int)    # does not need to be positive

parser.add_argument('--sym_loss', action='store_true')
parser.add_argument('--closed_loop', action='store_true')

# parser.add_argument('--feature_size', default=512, type=int)
parser.add_argument('--projection', default=2048, type=int)
parser.add_argument('--proj_hidden', default=2048, type=int)
parser.add_argument('--pred_hidden', default=16, type=int)
parser.add_argument('--pred_layer', default=2, type=int)
parser.add_argument('--proj_layer', default=3, type=int)

parser.add_argument('--mse_l', default=1.0, type=float)
parser.add_argument('--loop_l', default=0.0, type=float)
parser.add_argument('--std_l', default=1.0, type=float)
parser.add_argument('--cov_l', default=0.04, type=float)
parser.add_argument('--spa_l', default=0.0, type=float)

parser.add_argument('--bn_last', action='store_true')
parser.add_argument('--pred_bn_last', action='store_true')

parser.add_argument('--num_predictor', default=1, type=int)
parser.add_argument('--predictor', default=1, type=int)

parser.add_argument('--infonce', action='store_true')
parser.add_argument('--sub_loss', action='store_true')
parser.add_argument('--sub_frac', default=0.25, type=float)

parser.add_argument('--base_lr', default=4.8, type=float)

# Running
parser.add_argument("--num-workers", type=int, default=32)
parser.add_argument('--device', default='cuda',
                    help='device to use for training / testing')

# Distributed
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
# parser.add_argument('--local-rank', default=-1, type=int)
parser.add_argument('--dist-url', default='env://',
                    help='url used to set up distributed training')

parser.add_argument('--r21d', action='store_true')

parser.add_argument('--mk200', action='store_true')
parser.add_argument('--mk400', action='store_true')
parser.add_argument('--minik', action='store_true')
parser.add_argument('--k400', action='store_true')
parser.add_argument('--fraction', default=1.0, type=float)

parser.add_argument('--reg_all', action='store_true')


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    # warmup_steps = 10 * len(loader)
    warmup_steps = 0
    base_lr = args.base_lr * args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

def exclude_bias_and_norm(p):
    return p.ndim == 1

class LARS(optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        weight_decay=0,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=None,
        lars_adaptation_filter=None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if g["lars_adaptation_filter"] is None or not g[
                    "lars_adaptation_filter"
                ](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])


def train_one_epoch(args, model, train_loader, optimizer, epoch, gpu=None, scaler=None, train=True, diff=False, mix=False, mix2=False):
    # global have_print

    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.
    num_batches = len(train_loader)

    # for data in train_loader:
    for step, data in enumerate(train_loader, start=epoch * len(train_loader)):
        video, label = data # B, N, C, T, H, W
        label = label.to(gpu)
        video = video.to(gpu)

        if random.random() < 0.5:
            video = video[:,:,:,1:,:,:] - video[:,:,:,:-1,:,:]

        if diff:
            video = video[:,:,:,1:,:,:] - video[:,:,:,:-1,:,:]
        if mix:
            video_diff = video[:,:,:,1:,:,:] - video[:,:,:,:-1,:,:]
            video = video[:,:,:,:-1,:,:]
            video[:,1,:,:,:,:] = video_diff[:,1,:,:,:,:]
        if mix2:
            video_diff = video[:,:,:,1:,:,:] - video[:,:,:,:-1,:,:]
            video = video[:,:,:,:-1,:,:]
            video[:,0,:,:,:,:] = video_diff[:,0,:,:,:,:]

        # video_diff = video[:,:,:,1:,:,:] - video[:,:,:,:-1,:,:]
        # print(video.shape)

        lr = adjust_learning_rate(args, optimizer, train_loader, step)

        # if not have_print:
        #     print(images.size(), images2.size())
        #     have_print = True

        optimizer.zero_grad()
        # loss = model(video)
        with torch.cuda.amp.autocast():
            # loss = (model(video) + model(video_diff)) / 2
            loss = model(video)
            # print(loss)
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        # if train: # train separately for each step
        #     # loss.sum().backward()
        #     loss.mean().backward()
        #     optimizer.step()
        # else:
        #     pass
        # total_loss += loss.sum().item() 
        total_loss += loss.mean().item() 

    
    return total_loss/num_batches

def main():
    torch.manual_seed(233)
    np.random.seed(233)

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    init_distributed_mode(args)
    print(args)
    gpu = torch.device(args.device)
    
    model_select = PCNET_VIC

    if args.infonce:
        ind_name = 'nce'
    else:
        ind_name = 'pcn'

    if args.r21d:
        model_name = 'r21d18'
        resnet = models.video.r2plus1d_18()
    else:
        model_name = 'r3d18'
        resnet = models.video.r3d_18()

    if args.k400:
        dataname = 'k400'
    elif args.mk200:
        dataname = 'mk200'
    elif args.mk400:
        dataname = 'mk400'
    elif args.minik:
        dataname = 'minik'
    else:
        dataname = 'ucf'

    ckpt_folder='/home/siyich/byol-pytorch/checkpoints_diff6_%s_f%s_%s_%s_112/prj%s_hidproj%s_hidpre%s_prl%s_pre%s_np%s_pl%s_il%s_ns%s/mse%s_loop%s_std%s_cov%s_spa%s_rall%s_sym%s_closed%s_sub%s_sf%s/bs%s_lr%s_wd%s_ds%s_sl%s_nw_rand%s' \
        % (dataname, args.fraction, ind_name, model_name, args.projection, args.proj_hidden, args.pred_hidden, args.proj_layer, args.predictor, args.num_predictor, args.pred_layer, args.inter_len, args.num_seq, args.mse_l, args.loop_l, args.std_l, args.cov_l, args.spa_l, args.reg_all, args.sym_loss, args.closed_loop, args.sub_loss, args.sub_frac, args.batch_size, args.base_lr, args.wd, args.downsample, args.seq_len, args.random)

    if args.rank == 0:
        if not os.path.exists(ckpt_folder):
            os.makedirs(ckpt_folder)
        logging.basicConfig(filename=os.path.join(ckpt_folder, 'pcnet_vic_train.log'), level=logging.INFO)
        logging.info('Started')
   

    model = model_select(
        resnet,
        clip_size = args.seq_len,
        # image_size = 224,
        image_size = 112,
        hidden_layer = 'avgpool',
        projection_size = args.projection,
        projection_hidden_size = args.proj_hidden,
        pred_hidden_size = args.pred_hidden,
        pred_layer = args.pred_layer,
        proj_layer = args.proj_layer,
        sym_loss = args.sym_loss,
        closed_loop = args.closed_loop,
        mse_l = args.mse_l,
        loop_l = args.loop_l,
        std_l = args.std_l,
        cov_l = args.cov_l,
        spa_l = args.spa_l,
        bn_last = args.bn_last,
        pred_bn_last = args.pred_bn_last,
        predictor = args.predictor,
        num_predictor = args.num_predictor,
        infonce = args.infonce,
        sub_loss = args.sub_loss,
        sub_frac = args.sub_frac,
        reg_all = args.reg_all
    ).cuda(gpu)
    # sync bn does not works for ode
    if args.predictor == 3:
        model.encoder = nn.SyncBatchNorm.convert_sync_batchnorm(model.encoder)
    else:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    optimizer = LARS(
        model.parameters(),
        lr=0,
        weight_decay=args.wd,
        weight_decay_filter=exclude_bias_and_norm,
        lars_adaptation_filter=exclude_bias_and_norm,
    )
    

    if args.pretrain:
        pretrain_path = os.path.join(args.pretrain_folder, 'pcnet_epoch%s.pth.tar' % args.start_epoch)
        ckpt = torch.load(pretrain_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])

    assert args.batch_size % args.world_size == 0
    
    per_device_batch_size = args.batch_size // args.world_size
    # print(per_device_batch_size)

    if args.k400:
        loader_method = get_data_k400
    elif args.mk200:
        loader_method = get_data_mk200
    elif args.mk400:
        loader_method = get_data_mk400
    elif args.minik:
        loader_method = get_data_minik
    else:
        loader_method = get_data_ucf

    train_loader = loader_method(batch_size=per_device_batch_size, 
                                mode='train', 
                                # transform_consistent = transform_consistent(),
                                # transform_inconsistent = transform_inconsistent(),
                                transform_consistent=None, 
                                transform_inconsistent=default_transform(),
                                # transform_inconsistent=default_transform2(),
                                seq_len=args.seq_len, 
                                num_seq=args.num_seq, 
                                downsample=args.downsample,
                                random=args.random,
                                inter_len=args.inter_len,
                                frame_root=args.frame_root,
                                ddp=True,
                                dim=150,
                                # dim=240,
                                fraction=args.fraction,
                                )
    # test_loader = get_data_ucf(batch_size=per_device_batch_size, 
    #                             mode='val',
    #                             # transform_consistent = transform_consistent(),
    #                             # transform_inconsistent = transform_inconsistent(),
    #                             transform_consistent=None, 
    #                             # transform_inconsistent=default_transform(),
    #                             transform_inconsistent=default_transform2(),
    #                             seq_len=args.seq_len, 
    #                             num_seq=args.num_seq, 
    #                             downsample=args.downsample,
    #                             random=args.random,
    #                             inter_len=args.inter_len,
    #                             frame_root=args.frame_root,
    #                             ddp=True,
    #                             # dim=150,
    #                             dim = 240,
    #                             fraction = args.fraction
    #                             )
    
    train_loss_list = []
    test_loss_list = []
    train_loss_list3 = []
    train_loss_list4 = []
    epoch_list = range(args.start_epoch, args.epochs)
    lowest_loss = np.inf
    best_epoch = 0

    # start_time = last_logging = time.time()
    scaler = torch.cuda.amp.GradScaler()
    for i in epoch_list:
        # train_loss = train_one_epoch(args, model, train_loader, optimizer, i, gpu, scaler)

        train_loss2 = train_one_epoch(args, model, train_loader, optimizer, i, gpu, scaler, diff=True) 
        # train_loss3 = train_one_epoch(args, model, train_loader, optimizer, i, gpu, scaler) 

        train_loss3 = train_one_epoch(args, model, train_loader, optimizer, i, gpu, scaler, mix=True)
        train_loss4 = train_one_epoch(args, model, train_loader, optimizer, i, gpu, scaler, mix2=True)

        train_loss = train_one_epoch(args, model, train_loader, optimizer, i, gpu, scaler)

        # train_loss2 = train_one_epoch(args, model, train_loader, optimizer, i, gpu, scaler, diff=True) 
        # test_loss = train_one_epoch(args, model, test_loader, optimizer, i, gpu, scaler, False)
        # test_loss = train_loss
        test_loss = train_loss2
        
        # current_time = time.time()
        if args.rank == 0:
            if test_loss < lowest_loss:
                lowest_loss = test_loss
                best_epoch = i + 1
            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)
            train_loss_list3.append(train_loss3)
            train_loss_list4.append(train_loss4)

            print('Epoch: %s, Train loss: %s' % (i, train_loss))
            print('Epoch: %s, Train2 loss: %s' % (i, test_loss))
            print('Epoch: %s, Train3 loss: %s' % (i, train_loss3))
            print('Epoch: %s, Train4 loss: %s' % (i, train_loss4))
            logging.info('Epoch: %s, Train loss: %s' % (i, train_loss))
            logging.info('Epoch: %s, Train2 loss: %s' % (i, test_loss))
            logging.info('Epoch: %s, Train3 loss: %s' % (i, train_loss3))
            logging.info('Epoch: %s, Train4 loss: %s' % (i, train_loss4))


            if (i+1)%10 == 0 or i<20:
                # save your improved network
                checkpoint_path = os.path.join(
                    ckpt_folder, 'resnet_epoch%s.pth.tar' % str(i+1))
                torch.save(resnet.state_dict(), checkpoint_path)
                # save whole model and optimizer
                state = dict(
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                )
                checkpoint_path = os.path.join(
                    ckpt_folder, 'pcnet_epoch%s.pth.tar' % str(i+1))
                torch.save(state, checkpoint_path)

    if args.rank == 0:
        logging.info('Training from ep %d to ep %d finished' %
            (args.start_epoch, args.epochs))
        logging.info('Best epoch: %s' % best_epoch)

        # plot training process
        plt.plot(epoch_list, train_loss_list, label = 'train')
        # if not args.no_val:
        # plt.plot(epoch_list, test_loss_list, label = 'val')
        plt.plot(epoch_list, test_loss_list, label = 'train2')
        plt.plot(epoch_list, train_loss_list3, label = 'train3')
        plt.plot(epoch_list, train_loss_list4, label = 'train4')


        # plt.title('Train and test loss')

        plt.legend()
        plt.savefig(os.path.join(
            ckpt_folder, 'epoch%s_bs%s_loss.png' % (args.epochs, args.batch_size)))

        # save your improved network
        checkpoint_path = os.path.join(
            ckpt_folder, 'resnet_epoch%s.pth.tar' % str(args.epochs))
        torch.save(resnet.state_dict(), checkpoint_path)
        state = dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
            )
        checkpoint_path = os.path.join(
            ckpt_folder, 'pcnet_epoch%s.pth.tar' % str(args.epochs))
        torch.save(state, checkpoint_path)



if __name__ == '__main__':
    main()