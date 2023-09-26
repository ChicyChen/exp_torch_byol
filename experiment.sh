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

# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_byol_r3d18.py --sym_loss --base_lr 4.8 --proj_hidden 2048 --pred_hidden 2048 --downsample 3
# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_byol_r3d18.py --sym_loss --base_lr 4.8 --proj_hidden 2048 --pred_hidden 2048 --downsample 4
# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_byol_r3d18.py --sym_loss --base_lr 4.8 --proj_hidden 2048 --pred_hidden 2048 --downsample 8

# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_byol_r3d18.py --sym_loss --base_lr 4.8 --downsample 3 --cos_ema
# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_byol_r3d18.py --sym_loss --base_lr 4.8 --downsample 4 --cos_ema
# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_byol_r3d18.py --sym_loss --base_lr 4.8 --downsample 8 --cos_ema

# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --sym_loss --base_lr 4.8 --downsample 3 --proj_layer 2 --proj_hidden 4096 --projection 256 --pred_layer 2 --pred_hidden 4096

# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --sym_loss --base_lr 4.8 --downsample 3 --proj_layer 2 --proj_hidden 4096 --projection 2048 --pred_layer 2 --pred_hidden 16
# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --sym_loss --base_lr 4.8 --downsample 3 --proj_layer 2 --proj_hidden 4096 --projection 2048 --pred_layer 2 --pred_hidden 256

# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --sym_loss --base_lr 4.8 --downsample 3 --proj_layer 2 --proj_hidden 2048 --projection 2048 --pred_layer 2 --pred_hidden 16
# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --sym_loss --base_lr 4.8 --downsample 3 --proj_layer 2 --proj_hidden 2048 --projection 2048 --pred_layer 2 --pred_hidden 256

# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --sym_loss --base_lr 4.8 --downsample 3 --proj_layer 0 --proj_hidden 512 --projection 512 --pred_layer 2 --pred_hidden 16
# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --sym_loss --base_lr 4.8 --downsample 3 --proj_layer 0 --proj_hidden 512 --projection 512 --pred_layer 2 --pred_hidden 256

# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --sym_loss --base_lr 4.8 --downsample 3 --proj_layer 2 --proj_hidden 4096 --projection 2048
# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --sym_loss --base_lr 4.8 --downsample 3 --proj_layer 2 --proj_hidden 2048 --projection 2048 --pred_layer 2 --pred_hidden 16 --predictor 0 
# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --sym_loss --base_lr 4.8 --downsample 3 --proj_layer 2 --proj_hidden 2048 --projection 2048 --pred_layer 2 --pred_hidden 16 --predictor 2

# MLP
# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --sym_loss --base_lr 4.8 --downsample 3 --proj_layer 3 --proj_hidden 2048 --projection 2048 --pred_layer 2 --pred_hidden 16 --predictor 0 
# case 1
# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --sym_loss --base_lr 4.8 --downsample 3 --proj_layer 3 --proj_hidden 2048 --projection 2048 --pred_layer 2 --pred_hidden 16 --predictor 1 --closed_loop --mse_l 0.5 --loop_l 0.5
# case 2
# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --sym_loss --base_lr 4.8 --downsample 3 --proj_layer 3 --proj_hidden 2048 --projection 2048 --pred_layer 2 --pred_hidden 16 --predictor 1 --num_predictor 2
# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --sym_loss --base_lr 4.8 --downsample 3 --proj_layer 3 --proj_hidden 2048 --projection 2048 --pred_layer 2 --pred_hidden 16 --predictor 1 --num_predictor 2 --closed_loop --mse_l 0.5 --loop_l 0.5
# case 3
# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --sym_loss --base_lr 4.8 --downsample 3 --proj_layer 3 --proj_hidden 2048 --projection 2048 --pred_layer 2 --pred_hidden 16 --predictor 2 
# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --sym_loss --base_lr 4.8 --downsample 3 --proj_layer 3 --proj_hidden 2048 --projection 2048 --pred_layer 2 --pred_hidden 16 --predictor 2 --closed_loop --mse_l 0.5 --loop_l 0.5

# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --sym_loss --base_lr 4.8 --downsample 3 --proj_layer 3 --proj_hidden 2048 --projection 2048 --pred_layer 2 --pred_hidden 16 --predictor 1 --num_seq 3
# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --sym_loss --base_lr 4.8 --downsample 3 --proj_layer 3 --proj_hidden 2048 --projection 2048 --pred_layer 2 --pred_hidden 16 --predictor 0 --num_seq 3
# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --sym_loss --base_lr 4.8 --downsample 3 --proj_layer 3 --proj_hidden 2048 --projection 2048 --pred_layer 2 --pred_hidden 2048 --predictor 0
# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --sym_loss --base_lr 4.8 --downsample 3 --proj_layer 3 --proj_hidden 2048 --projection 2048 --pred_layer 2 --pred_hidden 1024 --predictor 0
# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --sym_loss --base_lr 4.8 --downsample 3 --proj_layer 3 --proj_hidden 2048 --projection 2048 --pred_layer 2 --pred_hidden 512 --predictor 0
# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --sym_loss --base_lr 4.8 --downsample 3 --proj_layer 3 --proj_hidden 2048 --projection 2048 --pred_layer 2 --pred_hidden 256 --predictor 0
# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --sym_loss --base_lr 4.8 --downsample 3 --proj_layer 3 --proj_hidden 2048 --projection 2048 --pred_layer 2 --pred_hidden 128 --predictor 0
# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --sym_loss --base_lr 4.8 --downsample 3 --proj_layer 3 --proj_hidden 2048 --projection 2048 --pred_layer 2 --pred_hidden 64 --predictor 0
# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --sym_loss --base_lr 4.8 --downsample 3 --proj_layer 3 --proj_hidden 2048 --projection 2048 --pred_layer 2 --pred_hidden 32 --predictor 0

# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --sym_loss --base_lr 4.8 --downsample 3 --proj_layer 3 --proj_hidden 2048 --projection 2048 --pred_layer 2 --pred_hidden 128 --predictor 1




# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --sym_loss --base_lr 4.8 --downsample 3 --proj_layer 3 --proj_hidden 2048 --projection 2048 --pred_layer 2 --pred_hidden 16 --predictor 3



# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_byol_r3d18.py --sym_loss --base_lr 4.8 --downsample 3 --cos_ema --projection 64 --pred_hidden 1024 --proj_hidden 1024



# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --sym_loss --base_lr 4.8 --infonce --proj_layer 3 --projection 32 --proj_hidden 512 --downsample 3

# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18_diff.py --epochs 100 --batch_size 256 --sym_loss --base_lr 4.8 --projection 8192 --proj_hidden 8192 --pred_layer 0 --proj_layer 3 --cov_l 0.04 --std_l 1.0 --spa_l 0.0 --minik
# # torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18_diff.py --epochs 100 --batch_size 256 --sym_loss --base_lr 3.6 --projection 8192 --proj_hidden 8192 --pred_layer 0 --proj_layer 3 --cov_l 0.04 --std_l 1.0 --spa_l 0.0 --minik

# python experiments/eval/eval_knn.py --ckpt_folder checkpoints_diff_minik_f1.0_pcn_r3d18_112/prj8192_hidproj8192_hidpre16_prl3_pre1_np1_pl0_il0_ns2/mse1.0_loop0.0_std1.0_cov0.04_spa0.0_rallFalse_symTrue_closedFalse_subFalse_sf0.25/bs256_lr4.8_wd1e-06_ds3_nw

# python experiments/eval/eval_retrieval.py --ckpt_folder checkpoints_diff_minik_f1.0_pcn_r3d18_112/prj8192_hidproj8192_hidpre16_prl3_pre1_np1_pl0_il0_ns2/mse1.0_loop0.0_std1.0_cov0.04_spa0.0_rallFalse_symTrue_closedFalse_subFalse_sf0.25/bs256_lr4.8_wd1e-06_ds3_nw

# python experiments/eval/eval_retrieval.py --diff --ckpt_folder checkpoints_diff_minik_f1.0_pcn_r3d18_112/prj8192_hidproj8192_hidpre16_prl3_pre1_np1_pl0_il0_ns2/mse1.0_loop0.0_std1.0_cov0.04_spa0.0_rallFalse_symTrue_closedFalse_subFalse_sf0.25/bs256_lr4.8_wd1e-06_ds3_nw

# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/eval/eval_tune_v2.py --ckpt_folder checkpoints_diff_minik_f1.0_pcn_r3d18_112/prj8192_hidproj8192_hidpre16_prl3_pre1_np1_pl0_il0_ns2/mse1.0_loop0.0_std1.0_cov0.04_spa0.0_rallFalse_symTrue_closedFalse_subFalse_sf0.25/bs256_lr4.8_wd1e-06_ds3_nw

# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/eval/eval_tune_v3.py --ckpt_folder checkpoints_diff_minik_f1.0_pcn_r3d18_112/prj8192_hidproj8192_hidpre16_prl3_pre1_np1_pl0_il0_ns2/mse1.0_loop0.0_std1.0_cov0.04_spa0.0_rallFalse_symTrue_closedFalse_subFalse_sf0.25/bs256_lr4.8_wd1e-06_ds3_nw

# python experiments/eval/eval_knn.py --ckpt_folder checkpoints_diff6_ucf_f1.0_pcn_r3d18_112/prj2048_hidproj2048_hidpre16_prl3_pre1_np1_pl0_il0_ns2/mse1.0_loop0.0_std1.0_cov0.04_spa0.0_rallFalse_symTrue_closedFalse_subFalse_sf0.25/bs64_lr4.8_wd1e-06_ds3_nw

# python experiments/eval/eval_retrieval.py --ckpt_folder checkpoints_diff6_ucf_f1.0_pcn_r3d18_112/prj2048_hidproj2048_hidpre16_prl3_pre1_np1_pl0_il0_ns2/mse1.0_loop0.0_std1.0_cov0.04_spa0.0_rallFalse_symTrue_closedFalse_subFalse_sf0.25/bs64_lr4.8_wd1e-06_ds3_nw --diff

# python experiments/eval/eval_retrieval.py --ckpt_folder checkpoints_diff6_ucf_f1.0_pcn_r3d18_112/prj2048_hidproj2048_hidpre16_prl3_pre1_np1_pl0_il0_ns2/mse1.0_loop0.0_std1.0_cov0.04_spa0.0_rallFalse_symTrue_closedFalse_subFalse_sf0.25/bs64_lr4.8_wd1e-06_ds3_nw

# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18_diff.py --epochs 100 --batch_size 64 --sym_loss --base_lr 4.8 --projection 2048 --proj_hidden 2048 --pred_layer 0 --proj_layer 3 --cov_l 0.04 --std_l 1.0 --spa_l 0.0 --minik

python experiments/eval/eval_retrieval.py --diff --ckpt_folder checkpoints_diff6_byol_ucf_f1.0_r3d18_112/hid4096_hidpre4096_prj256_prl2_pre1_np1_pl2_il0_ns2/mse1.0_std0.0_cov0.0_symTrue_closedFalse/bs256_lr4.8_wd1e-06_ds3_ema0.99_cosemaTrue

python experiments/eval/eval_retrieval.py --ckpt_folder checkpoints_diff6_byol_ucf_f1.0_r3d18_112/hid4096_hidpre4096_prj256_prl2_pre1_np1_pl2_il0_ns2/mse1.0_std0.0_cov0.0_symTrue_closedFalse/bs256_lr4.8_wd1e-06_ds3_ema0.99_cosemaTrue

python experiments/eval/eval_retrieval.py --diff --ckpt_folder checkpoints_diff6_ucf_f1.0_nce_r3d18_112/prj512_hidproj4096_hidpre16_prl3_pre1_np1_pl0_il0_ns2/mse1.0_loop0.0_std1.0_cov0.04_spa0.0_rallFalse_symTrue_closedFalse_subFalse_sf0.25/bs64_lr4.8_wd1e-06_ds3_nw_1

python experiments/eval/eval_retrieval.py --ckpt_folder checkpoints_diff6_ucf_f1.0_nce_r3d18_112/prj512_hidproj4096_hidpre16_prl3_pre1_np1_pl0_il0_ns2/mse1.0_loop0.0_std1.0_cov0.04_spa0.0_rallFalse_symTrue_closedFalse_subFalse_sf0.25/bs64_lr4.8_wd1e-06_ds3_nw_1