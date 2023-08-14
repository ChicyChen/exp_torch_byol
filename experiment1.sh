#!/bin/bash

# gpu='0,1,2,3,4,5,6,7'
# frame_root='/data'

# # byol
# python experiments/3dseq_v2/3dseq_pcnet_vic.py --gpu $gpu --frame_root $frame_root --batch_size 256 --predictor 1 --num_predictor 1 --sym_loss --projection 64 --proj_hidden 1024 --pred_hidden 1024
# vicreg
# python experiments/3dseq_v2/3dseq_pcnet_vic.py --gpu $gpu --frame_root $frame_root --batch_size 256 --predictor 1 --num_predictor 1 --sym_loss --projection 2048 --proj_hidden 2048 --proj_layer 3 --pred_hidden 2048 --pred_layer 0
# # others
# python experiments/3dseq_v2/3dseq_pcnet_vic.py --gpu $gpu --frame_root $frame_root --batch_size 256 --predictor 1 --num_predictor 1 --sym_loss --projection 2048 --pred_hidden 2048
# python experiments/3dseq_v2/3dseq_pcnet_vic.py --gpu $gpu --frame_root $frame_root --batch_size 256 --predictor 1 --num_predictor 1 --sym_loss --projection 2048 --pred_hidden 4096

# python experiments/3dseq_v2/3dseq_pcnet_vic.py --gpu $gpu --frame_root $frame_root --batch_size 256 --predictor 1 --num_predictor 1 --sym_loss --projection 512 --pred_hidden 128

# python experiments/3dseq_v2/3dseq_pcnet_vic.py --gpu $gpu --frame_root $frame_root --batch_size 256 --predictor 1 --num_predictor 1 --sym_loss --projection 1024 --pred_hidden 4096
# python experiments/3dseq_v2/3dseq_pcnet_vic.py --gpu $gpu --frame_root $frame_root --batch_size 256 --predictor 1 --num_predictor 1 --sym_loss --projection 1024 --pred_hidden 1024
# python experiments/3dseq_v2/3dseq_pcnet_vic.py --gpu $gpu --frame_root $frame_root --batch_size 256 --predictor 1 --num_predictor 1 --sym_loss --projection 1024 --pred_hidden 256

# python experiments/3dseq_v2/3dseq_pcnet_vic.py --gpu $gpu --frame_root $frame_root --batch_size 256 --predictor 1 --num_predictor 1 --sym_loss --projection 4096 --pred_hidden 4096
# python experiments/3dseq_v2/3dseq_pcnet_vic.py --gpu $gpu --frame_root $frame_root --batch_size 256 --predictor 1 --num_predictor 1 --sym_loss --projection 4096 --pred_hidden 1024

# python experiments/3dseq_v2/3dseq_pcnet_vic.py --gpu $gpu --frame_root $frame_root --batch_size 256 --predictor 1 --num_predictor 1 --sym_loss --projection 512 --proj_layer 0 --pred_hidden 512
# python experiments/3dseq_v2/3dseq_pcnet_vic.py --gpu $gpu --frame_root $frame_root --batch_size 256 --predictor 1 --num_predictor 1 --sym_loss --projection 512 --proj_layer 0 --pred_hidden 64

# python experiments/3dseq_v2/3dseq_pcnet_vic.py --gpu $gpu --frame_root $frame_root --batch_size 256 --predictor 1 --num_predictor 2 --sym_loss --projection 2048 --pred_hidden 512
# python experiments/3dseq_v2/3dseq_pcnet_vic.py --gpu $gpu --frame_root $frame_root --batch_size 256 --predictor 3 --num_predictor 1 --sym_loss --projection 2048 --pred_hidden 512
# python experiments/3dseq_v2/3dseq_pcnet_vic.py --gpu $gpu --frame_root $frame_root --batch_size 256 --predictor 3 --num_predictor 2 --sym_loss --projection 2048 --pred_hidden 512

# python experiments/3dseq_v2/3dseq_pcnet_vic.py --batch_size 256 --sym_loss --projection 2048 --pred_hidden 512 --pred_layer 2
# python experiments/3dseq_v2/3dseq_pcnet_vic.py --frame_root $frame_root --gpu $gpu --batch_size 256 --sym_loss --projection 2048 --pred_hidden 512 --pred_layer 0
# python experiments/3dseq_v2/3dseq_pcnet_vic.py --frame_root $frame_root --gpu $gpu --batch_size 256 --sym_loss --projection 2048 --pred_hidden 2048 --pred_layer 0 --proj_layer 3 
# python experiments/3dseq_v2/3dseq_pcnet_vic.py --frame_root $frame_root --gpu $gpu --batch_size 256 --sym_loss --projection 2048 --pred_hidden 2048 --proj_hidden 2048 --pred_layer 0 --proj_layer 3 

# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --sym_loss --base_lr 9.6
# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --sym_loss --base_lr 4.8
# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --sym_loss --base_lr 4.8 --infonce --proj_layer 2 --projection 128 --proj_hidden 4096 

# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --sym_loss --base_lr 4.8 --r21d
# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --sym_loss --base_lr 4.8 --infonce --proj_layer 3 --projection 128 --proj_hidden 2048 --r21d
# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_byol_r3d18.py --sym_loss --base_lr 4.8 --r21d

# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --sym_loss --base_lr 4.8 --downsample 4
# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --sym_loss --base_lr 4.8 --infonce --proj_layer 3 --projection 128 --proj_hidden 2048 --downsample 4
# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_byol_r3d18.py --sym_loss --base_lr 4.8 --downsample 4

# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --sym_loss --base_lr 4.8 --downsample 3
# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --sym_loss --base_lr 4.8 --infonce --proj_layer 3 --projection 128 --proj_hidden 2048 --downsample 3
# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_byol_r3d18.py --sym_loss --base_lr 4.8 --downsample 3

torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_byol_r3d18.py --sym_loss --base_lr 4.8 --proj_hidden 2048 --pred_hidden 2048 --downsample 3
torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_byol_r3d18.py --sym_loss --base_lr 4.8 --proj_hidden 2048 --pred_hidden 2048 --downsample 4
torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_byol_r3d18.py --sym_loss --base_lr 4.8 --proj_hidden 2048 --pred_hidden 2048 --downsample 8



