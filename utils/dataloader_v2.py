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


def get_data_ucf(transform_consistent=None,
                transform_inconsistent=None,
                mode='train', 
                seq_len=4, 
                num_seq=3, 
                downsample=8, 
                which_split=1, 
                return_label=True, 
                batch_size=16, 
                dim=150,
                csv_root='/home/siyich/byol-pytorch/data_video',
                frame_root='/home/siyich/Datasets/Videos',
                ddp=False,
                random=False,
                inter_len=0 # num of frames (after downsampling) between two clips
                ):
    print('Loading data for "%s" ...' % mode)
    dataset = UCF101(mode=mode,
                        transform_consistent=transform_consistent,
                        transform_inconsistent=transform_inconsistent,
                        seq_len=seq_len,
                        num_seq=num_seq,
                        downsample=downsample,
                        which_split=which_split,
                        return_label=return_label,
                        dim=dim,
                        csv_root=csv_root,
                        frame_root=frame_root,
                        random=random,
                        inter_len=inter_len
                        )
    if not ddp:
        sampler = data.RandomSampler(dataset)
    else:
        sampler = data.distributed.DistributedSampler(dataset)

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


class UCF101(data.Dataset):
    def __init__(self,
                mode='train',
                transform_consistent=None,
                transform_inconsistent=None,
                seq_len:int=4,
                num_seq:int=3,
                downsample:int=4,
                which_split=1,
                return_label=False,
                dim=150,
                csv_root='/home/siyich/byol-pytorch/data_video',
                frame_root='/home/siyich/Datasets/Videos',
                random=False,
                inter_len:int=0 # num of frames (after downsampling) between two clips
                ):
        self.mode = mode
        self.transform_consistent = transform_consistent
        self.transform_inconsistent = transform_inconsistent
        self.seq_len=seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.which_split = which_split
        self.return_label = return_label
        self.dim = dim
        self.csv_root = csv_root
        self.frame_root = frame_root
        self.random = random
        self.inter_len = inter_len
        self.total_len = ((self.seq_len + self.inter_len)*self.num_seq - self.inter_len)*self.downsample
        
        begin_idxs = np.arange(self.num_seq)*self.downsample*(self.seq_len+self.inter_len) 
        inter_idxs = (np.arange(self.seq_len)*self.downsample).reshape(-1,1)
        self.base_seq_idx = (inter_idxs + begin_idxs).T.flatten()

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
            if vlen-self.total_len <= 0:
                drop_idx.append(idx)
        self.video_info = video_info.drop(drop_idx, axis=0)
        print("Droped number of videos:", len(drop_idx))

        if mode == 'val': self.video_info = self.video_info.sample(frac=0.3)
        # shuffle not required due to external sampler

    def idx_sampler(self, vlen, vpath):
        '''sample index from a video'''
        
        if vlen-self.total_len <= 0: raise ValueError('video too short')
        if not self.random:
            n = 1
            start_idx = np.random.choice(range(vlen-self.total_len), n)
            seq_idx = self.base_seq_idx + start_idx

            # seq_idx = np.arange(self.seq_len*self.num_seq)*self.downsample + start_idx
        else:
            n = self.seq_len*self.num_seq
            seq_idx = np.random.choice(range(vlen-self.total_len), n)


        return [seq_idx, vpath]


    def __getitem__(self, index):
        vpath, vlen, aid = self.video_info.iloc[index]
        items = self.idx_sampler(vlen, vpath)
        if items is None: print(vpath) 
        
        idx_block, vpath = items
        
        seq = [pil_loader(os.path.join(self.frame_root, vpath, 'image_%05d.jpg' % (i+1))) for i in idx_block]

        if self.transform_consistent is not None: 
                seq = self.transform_consistent(seq) # apply same transform

        if self.transform_inconsistent is not None: 
            for i in range(self.num_seq):
                sub_seq = self.transform_inconsistent(seq[i*self.seq_len:(i+1)*self.seq_len])
                seq[i*self.seq_len:(i+1)*self.seq_len] = sub_seq
                

        (C, H, W) = seq[0].size()
        seq = torch.stack(seq, 0)
        seq = seq.view(self.num_seq, self.seq_len, C, H, W) # N, T, C, H, W
        seq = seq.permute(0,2,1,3,4) # N, C, T, H, W
        # seq = seq.view(self.num_seq*self.seq_len, C, H, W) # N*T, C, H, W
        # seq = seq.permute(1,0,2,3) # C, N*T, H, W
        
        if self.return_label:
            label = torch.LongTensor([aid])
            return seq, label
        return seq

    def __len__(self):
        return len(self.video_info)




def test():
    transform_consistent = transforms.Compose([
        RandomHorizontalFlip(consistent=True),
        RandomCrop(size=112, consistent=True),
        Scale(size=(112,112)),
        # GaussianBlur(size=112, p=0.5, consistent=True),
        # ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.8),
        # RandomGray(consistent=False, p=0.2),
        # ToTensor(),
        # Normalize()
    ])

    transform_inconsistent = transforms.Compose([
        # RandomHorizontalFlip(consistent=True),
        # RandomCrop(size=112, consistent=True),
        # Scale(size=(112,112)),
        GaussianBlur(size=112, p=0.5, consistent=True),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.8),
        RandomGray(consistent=False, p=0.2),
        ToTensor(),
        # Normalize()
    ])

    dataset = UCF101(mode='val', frame_root='/data')
    print(len(dataset))
    val_data = get_data_ucf(transform_consistent, transform_inconsistent, 'val', frame_root='/data')

    i=0
    for data in val_data:
        images, _ = data
        # print(images, label)
        # print(images.size())
        transform_back = T.ToPILImage()
        for j in range(4):
            for k in range(3):
                images0 = transform_back(images.permute(0,1,3,2,4,5)[0,k,j])
                images0.save("vis%s_%s_%s.jpg" % (i, k, j))
        i += 1
        if i >= 5:
            break


if __name__ == '__main__':
    # transform = transforms.Compose([
    #     RandomHorizontalFlip(consistent=True),
    #     RandomCrop(size=112, consistent=True),
    #     Scale(size=(112,112)),
    #     RandomGray(consistent=False, p=0.5),
    #     ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=1.0),
    #     ToTensor(),
    #     Normalize()
    # ])
    test()

