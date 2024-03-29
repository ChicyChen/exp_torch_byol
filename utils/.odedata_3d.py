import os.path
import pickle
from typing import Any, Callable, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from augmentation import *

import torch
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms as T

from torchvision.transforms import Compose, Lambda

from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def get_ode_ucf(transform=None,
                    transform2=None,
                    mode='train', 
                    seq_len=4, 
                    num_seq=2, 
                    downsample=4, 
                    which_split=1, 
                    return_label=True, 
                    batch_size=16, 
                    dim=150,
                    csv_root='/home/siyich/byol-pytorch/data_video',
                    frame_root='/home/siyich/Datasets/Videos',
                    num_aug=2,
                    ordered=True):
    print('Loading data for "%s" ...' % mode)
    dataset = UCF101_ODE(mode=mode,
                        transform=transform,
                        transform2=transform2,
                        seq_len=seq_len,
                        num_seq=num_seq,
                        downsample=downsample,
                        which_split=which_split,
                        return_label=return_label,
                        dim=dim,
                        csv_root=csv_root,
                        frame_root=frame_root,
                        num_aug=num_aug,
                        ordered=ordered
                        )
    sampler = data.RandomSampler(dataset)
    if mode == 'train':
        data_loader = data.DataLoader(dataset,
                                      batch_size=batch_size,
                                      sampler=sampler,
                                      shuffle=False,
                                      num_workers=4,
                                      pin_memory=True,
                                      drop_last=True)
    else:
        data_loader = data.DataLoader(dataset,
                                      batch_size=batch_size,
                                      sampler=sampler,
                                      shuffle=False,
                                      num_workers=4,
                                      pin_memory=True,
                                      drop_last=True)
    print('"%s" dataset size: %d' % (mode, len(dataset)))
    return data_loader


class UCF101_ODE(data.Dataset):
    def __init__(self,
                mode='train',
                transform=None, 
                transform2=None, 
                seq_len=4,
                num_seq=2,
                downsample=4,
                which_split=1,
                return_label=False,
                dim=150,
                csv_root='/home/siyich/byol-pytorch/data_video',
                frame_root='/home/siyich/Datasets/Videos',
                num_aug=2,
                ordered=True
                ):
        self.mode = mode
        self.transform = transform
        self.transform2 = transform2
        self.seq_len=seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.which_split = which_split
        self.return_label = return_label
        self.dim = dim
        self.csv_root = csv_root
        self.frame_root = frame_root
        self.num_aug = num_aug
        self.ordered = ordered

        if dim == 150:
            folder_name = 'ucf101_150'
        else:
            folder_name = 'ucf101_240'

        # splits
        if mode == 'train':
            if self.which_split == 0:
                split = os.path.join(self.csv_root, folder_name, 'train.csv')
            else:
                split = os.path.join(self.csv_root, folder_name, 'train_split%02d.csv' % self.which_split)
            video_info = pd.read_csv(split, header=None)
        elif (mode == 'val') or (mode == 'test'): # use val for test
            if self.which_split == 0:
                split = os.path.join(self.csv_root, folder_name, 'test.csv')
            else:
                split = os.path.join(self.csv_root, folder_name, 'test_split%02d.csv' % self.which_split)
            video_info = pd.read_csv(split, header=None)
        else: raise ValueError('wrong mode')

        # filter out too short videos:
        drop_idx = []
        for idx, row in video_info.iterrows():
            _, vlen, _ = row
            if vlen-self.seq_len*self.num_seq*self.downsample <= 0:
                drop_idx.append(idx)
        self.video_info = video_info.drop(drop_idx, axis=0)
        print("Droped number of videos:", len(drop_idx))

        if mode == 'val': self.video_info = self.video_info.sample(frac=0.3)
        # shuffle not required due to external sampler

    def idx_sampler(self, vlen, vpath):
        '''sample index from a video'''
        if vlen-self.seq_len*self.num_seq*self.downsample <= 0: raise ValueError('video too short')

        if self.ordered:
            start_idx = np.random.choice(range(vlen-self.seq_len*self.num_seq*self.downsample), 1)
            start_idx = np.arange(self.num_seq)*self.downsample*self.seq_len + start_idx
        else:
            start_idx = np.random.choice(range(vlen-self.seq_len*self.downsample), self.num_seq)
        seq_idx = start_idx[:, np.newaxis] + np.arange(self.seq_len)*self.downsample
        seq_idx = seq_idx.flatten()

        return [start_idx, seq_idx, vpath]
        


    def __getitem__(self, index):
        vpath, vlen, aid = self.video_info.iloc[index]

        items = self.idx_sampler(vlen, vpath) # will resample a different clip for each epoch
        if items is None: print(vpath) 
        
        idx_start, idx_block, vpath = items
        # if index == 0:
        #     print(idx_block[0])

        seq = [pil_loader(os.path.join(self.frame_root, vpath, 'image_%05d.jpg' % (i+1))) for i in idx_block]

        if self.return_label:
            label = torch.LongTensor([aid])

        if self.transform is not None: 
            t_seq = self.transform(seq) # apply same transform
        else:
            t_seq = seq
        (C, H, W) = t_seq[0].size()
        t_seq = torch.stack(t_seq, 0)
        t_seq = t_seq.view(self.seq_len*self.num_seq, C, H, W)
        t_seq=t_seq.permute(1,0,2,3)
        
        if self.num_aug > 1:
            if self.transform2 is not None:
                t_seq2 = self.transform2(seq)
            else:
                t_seq2 = seq
            t_seq2 = torch.stack(t_seq2, 0)
            t_seq2 = t_seq2.view(self.seq_len*self.num_seq, C, H, W)
            t_seq2=t_seq2.permute(1,0,2,3)

            if self.return_label:
                return t_seq, t_seq2, label, idx_start
            return t_seq, t_seq2, idx_start
        
        if self.return_label:
            return t_seq, label, idx_start
        return t_seq, idx_start

    def __len__(self):
        return len(self.video_info)

def test():
    transform = transforms.Compose([
        RandomHorizontalFlip(consistent=True),
        RandomCrop(size=128, consistent=True),
        Scale(size=(128,128)),
        GaussianBlur(size=128, p=0.5, consistent=True),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.8),
        RandomGray(consistent=False, p=0.2),
        ToTensor(),
        # Normalize()
    ])
    dataset = UCF101_ODE(mode='val')
    print(len(dataset))
    val_data = get_ode_ucf(transform, transform, 'val', ordered=False)
    
    # i=0
    for ep in range(10):
        for data in val_data:
            images, _, label, idx = data
            # print(images, label)
            # print(images.size())
            # transform_back = T.ToPILImage()
            # images0 = transform_back(images.permute(0,2,1,3,4)[0,0])
            # images0.save("vis%s.jpg" % i)
            # i += 1
            # if i >= 20:
            #     break


if __name__ == '__main__':
    # transform = transforms.Compose([
    #     RandomHorizontalFlip(consistent=True),
    #     RandomCrop(size=128, consistent=True),
    #     Scale(size=(128,128)),
    #     RandomGray(consistent=False, p=0.5),
    #     ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=1.0),
    #     ToTensor(),
    #     Normalize()
    # ])
    test()