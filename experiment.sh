#!/bin/bash

gpu='0,1,2,3,4,5,6,7'
frame_root='/data'

pretrain_folder='checkpoints_vic/hid4096_prj256_prl2_pre3_np1_pl2_il0_ns2/mse1.0_std1.0_cov0.04_symTrue_closedFalse/bs256_lr0.0001_wd1e-05'
python experiments/3dseq_v2/3dseq_pcnet_vic.py --gpu $gpu --frame_root $frame_root --batch_size 256 --num_seq 2 --predictor 3 --num_predictor 1 --pretrain --pretrain_folder $pretrain_folder --start-epoch 50

python experiments/3dseq_v2/3dseq_pcnet_vic.py --gpu $gpu --frame_root $frame_root --batch_size 256 --num_seq 2 --predictor 1 --num_predictor 1
# python experiments/3dseq_v2/3dseq_pcnet_vic.py --gpu $gpu --frame_root $frame_root --batch_size 128 --num_seq 3 --predictor 1 --num_predictor 1 --sym_loss --closed_loop
# python experiments/3dseq_v2/3dseq_pcnet_vic.py --gpu $gpu --frame_root $frame_root --batch_size 128 --num_seq 3 --predictor 1

python experiments/3dseq_v2/3dseq_pcnet_vic.py --gpu $gpu --frame_root $frame_root --batch_size 256 --num_seq 2 --predictor 1 --num_predictor 2 --closed_loop
# python experiments/3dseq_v2/3dseq_pcnet_vic.py --gpu $gpu --frame_root $frame_root --batch_size 128 --num_seq 3 --predictor 1 --num_predictor 2 --closed_loop
# python experiments/3dseq_v2/3dseq_pcnet_vic.py --gpu $gpu --frame_root $frame_root --batch_size 128 --num_seq 3 --predictor 1 --num_predictor 2 --sym_loss --closed_loop

python experiments/3dseq_v2/3dseq_pcnet_vic.py --gpu $gpu --frame_root $frame_root --batch_size 256 --num_seq 2 --predictor 2 --num_predictor 1 --closed_loop
python experiments/3dseq_v2/3dseq_pcnet_vic.py --gpu $gpu --frame_root $frame_root --batch_size 256 --num_seq 2 --predictor 2 --num_predictor 1 --sym_loss --closed_loop
# python experiments/3dseq_v2/3dseq_pcnet_vic.py --gpu $gpu --frame_root $frame_root --batch_size 128 --num_seq 3 --predictor 2 --num_predictor 1 --closed_loop
# python experiments/3dseq_v2/3dseq_pcnet_vic.py --gpu $gpu --frame_root $frame_root --batch_size 128 --num_seq 3 --predictor 2 --num_predictor 1 --sym_loss --closed_loop
# python experiments/3dseq_v2/3dseq_pcnet_vic.py --gpu $gpu --frame_root $frame_root --batch_size 128 --num_seq 3 --predictor 2 --num_predictor 1

# python experiments/3dseq_v2/3dseq_pcnet_vic.py --gpu $gpu --frame_root $frame_root --batch_size 128 --num_seq 3 --predictor 0 --num_predictor 1 --sym_loss