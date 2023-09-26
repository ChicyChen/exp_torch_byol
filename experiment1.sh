#!/bin/bash

# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --sym_loss --base_lr 4.8 --downsample 3 --proj_layer 3 --proj_hidden 2048 --projection 2048 --pred_layer 2 --pred_hidden 16 --predictor 1 --spa_l 1.0 --epochs 5

# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_stage2.py --pretrain --pretrain_folder checkpoints_gather_ucf_pcn_r3d18_112/prj2048_hidproj2048_hidpre16_prl3_pre1_np1_pl2_il0_ns2/mse1.0_loop0.0_std1.0_cov0.04_rallFalse_symTrue_closedFalse/bs256_lr4.8_wd1e-06_ds3 --sym_loss --base_lr 0.48
# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_stage2.py --pretrain --pretrain_folder checkpoints_gather_ucf_pcn_r3d18_112/prj2048_hidproj2048_hidpre16_prl3_pre1_np1_pl2_il0_ns2/mse1.0_loop0.0_std1.0_cov0.04_rallFalse_symTrue_closedFalse/bs256_lr4.8_wd1e-06_ds3 --sym_loss --base_lr 0.48 --spa_l 0.5
# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_stage2.py --pretrain --pretrain_folder checkpoints_gather_ucf_pcn_r3d18_112/prj2048_hidproj2048_hidpre16_prl3_pre1_np1_pl2_il0_ns2/mse1.0_loop0.0_std1.0_cov0.04_rallFalse_symTrue_closedFalse/bs256_lr4.8_wd1e-06_ds3 --sym_loss --base_lr 0.48 --spa_l 1.0

# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_stage2.py --pretrain --pretrain_folder checkpoints_gather_ucf_pcn_r3d18_112/prj2048_hidproj2048_hidpre16_prl3_pre1_np1_pl2_il0_ns2/mse1.0_loop0.0_std1.0_cov0.04_rallFalse_symTrue_closedFalse/bs256_lr4.8_wd1e-06_ds3 --sym_loss --base_lr 0.48 --spa_l 1.0 --num_seq 3
# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_stage2.py --pretrain --pretrain_folder checkpoints_gather_ucf_pcn_r3d18_112/prj2048_hidproj2048_hidpre16_prl3_pre1_np1_pl2_il0_ns2/mse1.0_loop0.0_std1.0_cov0.04_rallFalse_symTrue_closedFalse/bs256_lr4.8_wd1e-06_ds3 --sym_loss --base_lr 0.48 --spa_l 1.0 --inter_len 4
# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_stage2.py --pretrain --pretrain_folder checkpoints_gather_ucf_pcn_r3d18_112/prj2048_hidproj2048_hidpre16_prl3_pre1_np1_pl2_il0_ns2/mse1.0_loop0.0_std1.0_cov0.04_rallFalse_symTrue_closedFalse/bs256_lr4.8_wd1e-06_ds3 --sym_loss --base_lr 0.48 --spa_l 1.0 --inter_len 8

# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --sym_loss --base_lr 4.8 --downsample 3 --proj_layer 3 --proj_hidden 2048 --projection 2048 --pred_layer 2 --pred_hidden 16 --predictor 1 --epochs 10 --spa_l 1.0
# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --sym_loss --base_lr 4.8 --downsample 3 --proj_layer 3 --proj_hidden 2048 --projection 2048 --pred_layer 2 --pred_hidden 16 --predictor 1 --epochs 10 --spa_l 0.5
# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --sym_loss --base_lr 4.8 --downsample 3 --proj_layer 3 --proj_hidden 2048 --projection 2048 --pred_layer 2 --pred_hidden 16 --predictor 1 --epochs 10 --spa_l 0.1
# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --sym_loss --base_lr 4.8 --downsample 3 --proj_layer 3 --proj_hidden 2048 --projection 2048 --pred_layer 2 --pred_hidden 16 --predictor 1 --epochs 10 --spa_l 0.01

# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_stage2.py --pretrain_resnet --pretrain_folder checkpoints_gather_ucf_pcn_r3d18_112/prj2048_hidproj2048_hidpre16_prl3_pre1_np1_pl2_il0_ns2/mse1.0_loop0.0_std1.0_cov0.04_rallFalse_symTrue_closedFalse/bs256_lr4.8_wd1e-06_ds3 --sym_loss --base_lr 0.48 --num_seq 2 --projection 512 --proj_hidden 512 --proj_layer 0 --spa_l 0.0
# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_stage2.py --pretrain_resnet --pretrain_folder checkpoints_gather_ucf_pcn_r3d18_112/prj2048_hidproj2048_hidpre16_prl3_pre1_np1_pl2_il0_ns2/mse1.0_loop0.0_std1.0_cov0.04_rallFalse_symTrue_closedFalse/bs256_lr4.8_wd1e-06_ds3 --sym_loss --base_lr 0.48 --num_seq 2 --projection 512 --proj_hidden 512 --proj_layer 0 --spa_l 0.1

# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_stage2.py --pretrain_resnet --pretrain_folder checkpoints_gather_ucf_pcn_r3d18_112/prj2048_hidproj2048_hidpre16_prl3_pre1_np1_pl2_il0_ns2/mse1.0_loop0.0_std1.0_cov0.04_rallFalse_symTrue_closedFalse/bs256_lr4.8_wd1e-06_ds3 --sym_loss --base_lr 4.8 --num_seq 2 --projection 512 --proj_hidden 512 --proj_layer 0 --spa_l 0.0



# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --sym_loss --base_lr 4.8 --downsample 3 --proj_layer 3 --proj_hidden 2048 --projection 2048 --pred_layer 2 --pred_hidden 16 --predictor 1 --spa_l 1.0 --epochs 5

# torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_stage2.py --pretrain --pretrain_folder checkpoints_gather_ucf_pcn_r3d18_112/prj2048_hidproj2048_hidpre16_prl3_pre1_np1_pl2_il0_ns2/mse1.0_loop0.0_std1.0_cov0.04_rallFalse_symTrue_closedFalse/bs256_lr4.8_wd1e-06_ds3 --sym_loss --base_lr 0.48 --spa_l 0.0

torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --epochs 100 --batch_size 256 --sym_loss --base_lr 4.8 --projection 2048 --proj_hidden 2048 --pred_layer 2 --proj_layer 3 --cov_l 0.03 --std_l 1.0 --spa_l 1.0 
torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --epochs 100 --batch_size 256 --sym_loss --base_lr 4.8 --projection 2048 --proj_hidden 2048 --pred_layer 2 --proj_layer 3 --cov_l 0.03 --std_l 1.0 --spa_l 2.0 
torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --epochs 100 --batch_size 256 --sym_loss --base_lr 4.8 --projection 2048 --proj_hidden 2048 --pred_layer 2 --proj_layer 3 --cov_l 0.03 --std_l 1.0 --spa_l 5.0

torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --epochs 100 --batch_size 256 --sym_loss --base_lr 4.8 --projection 2048 --proj_hidden 2048 --pred_layer 2 --proj_layer 3 --cov_l 0.04 --std_l 1.0 --spa_l 1.0
torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --epochs 100 --batch_size 256 --sym_loss --base_lr 4.8 --projection 2048 --proj_hidden 2048 --pred_layer 2 --proj_layer 3 --cov_l 0.04 --std_l 1.0 --spa_l 2.0
torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --epochs 100 --batch_size 256 --sym_loss --base_lr 4.8 --projection 2048 --proj_hidden 2048 --pred_layer 2 --proj_layer 3 --cov_l 0.04 --std_l 1.0 --spa_l 5.0

torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --epochs 100 --batch_size 256 --sym_loss --base_lr 4.8 --projection 2048 --proj_hidden 2048 --pred_layer 2 --proj_layer 3 --cov_l 0.02 --std_l 1.0 --spa_l 1.0
torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --epochs 100 --batch_size 256 --sym_loss --base_lr 4.8 --projection 2048 --proj_hidden 2048 --pred_layer 2 --proj_layer 3 --cov_l 0.02 --std_l 1.0 --spa_l 2.0
torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments/3dseq_v2/3dseq_pcnet_vic_r3d18.py --epochs 100 --batch_size 256 --sym_loss --base_lr 4.8 --projection 2048 --proj_hidden 2048 --pred_layer 2 --proj_layer 3 --cov_l 0.02 --std_l 1.0 --spa_l 5.0






