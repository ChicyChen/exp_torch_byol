#!/bin/bash

gpu='0,1,2,3,4,5,6,7'
frame_root='/data'

# python experiments/3dseq_v2/3dseq_pcnet_vic.py --gpu $gpu --frame_root $frame_root --batch_size 256 --num_seq 2 --predictor 1 --num_predictor 1
python experiments/3dseq_v2/3dseq_pcnet_vic.py --gpu $gpu --frame_root $frame_root --batch_size 128 --num_seq 3 --predictor 1 --num_predictor 1 --sym_loss --closed_loop
python experiments/3dseq_v2/3dseq_pcnet_vic.py --gpu $gpu --frame_root $frame_root --batch_size 128 --num_seq 3 --predictor 1

# python experiments/3dseq_v2/3dseq_pcnet_vic.py --gpu $gpu --frame_root $frame_root --batch_size 256 --num_seq 2 --predictor 1 --num_predictor 2 --closed_loop
python experiments/3dseq_v2/3dseq_pcnet_vic.py --gpu $gpu --frame_root $frame_root --batch_size 128 --num_seq 3 --predictor 1 --num_predictor 2 --closed_loop
python experiments/3dseq_v2/3dseq_pcnet_vic.py --gpu $gpu --frame_root $frame_root --batch_size 128 --num_seq 3 --predictor 1 --num_predictor 2 --sym_loss --closed_loop

# # python experiments/3dseq_v2/3dseq_pcnet_vic.py --gpu $gpu --frame_root $frame_root --batch_size 256 --num_seq 2 --predictor 2 --num_predictor 1 --closed_loop
# # python experiments/3dseq_v2/3dseq_pcnet_vic.py --gpu $gpu --frame_root $frame_root --batch_size 256 --num_seq 2 --predictor 2 --num_predictor 1 --sym_loss --closed_loop
# python experiments/3dseq_v2/3dseq_pcnet_vic.py --gpu $gpu --frame_root $frame_root --batch_size 128 --num_seq 3 --predictor 2 --num_predictor 1 --closed_loop
# python experiments/3dseq_v2/3dseq_pcnet_vic.py --gpu $gpu --frame_root $frame_root --batch_size 128 --num_seq 3 --predictor 2 --num_predictor 1 --sym_loss --closed_loop
# python experiments/3dseq_v2/3dseq_pcnet_vic.py --gpu $gpu --frame_root $frame_root --batch_size 128 --num_seq 3 --predictor 2 --num_predictor 1

# python experiments/3dseq_v2/3dseq_pcnet_vic.py --gpu $gpu --frame_root $frame_root --batch_size 128 --num_seq 3 --predictor 0 --num_predictor 1 --sym_loss