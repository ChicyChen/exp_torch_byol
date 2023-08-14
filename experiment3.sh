#!/bin/bash

torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --sym_loss --base_lr 4.8 --downsample 3 --pred_layer 2 --pred_hidden 512
torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --sym_loss --base_lr 4.8 --downsample 3 --pred_layer 2 --pred_hidden 256
torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --sym_loss --base_lr 4.8 --downsample 3 --pred_layer 2 --pred_hidden 64
