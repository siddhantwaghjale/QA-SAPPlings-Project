#!/bin/bash
#SBATCH --job-name=rtp_2
#SBATCH --partition=general
#SBATCH --output=qa_down.out
#SBATCH --error=qa_down.err
#SBATCH --mem=256g
#SBATCH --gres=gpu:A6000:4
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=priyansk@andrew.cmu.edu

source /home/priyansk/.bashrc
conda activate qa
# export HF_HOME="/data/datasets/models/hf_cache/"

# python /home/priyansk/qa/MultiTabQA/train_custom.py \
# --output_dir /data/tir/projects/tir7/user_data/priyansk/qa_s1_codet5base \
# --dataset_name tapex_pretraining \
# --pretrained_model_name "Salesforce/codet5-base" \
# --learning_rate 1e-4 \
# --train_batch_size 8 \
# --eval_batch_size 4 \
# --gradient_accumulation_steps 32 \
# --eval_gradient_accumulation 32 \
# --num_train_epochs 2 \
# --num_workers 8 \
# --decoder_max_length 1024 \
# --seed 47 \
# --local_rank -1 \

# python /home/priyansk/qa/MultiTabQA/train_streaming.py \
# --output_dir /data/tir/projects/tir7/user_data/priyansk/qa_tapex_e2_codet5base_html \
# --dataset_name tapex_pretraining \
# --pretrained_model_name "Salesforce/codet5-base" \
# --learning_rate 1e-4 \
# --train_batch_size 8 \
# --eval_batch_size 4 \
# --gradient_accumulation_steps 32 \
# --eval_gradient_accumulation 32 \
# --num_train_epochs 1 \
# --num_workers 8 \
# --decoder_max_length 1024 \
# --seed 47 \
# --local_rank -1 \
# --resume_from_checkpoint /data/tir/projects/tir7/user_data/priyansk/qa_tapex_e2_codet5base_html/checkpoint-4600/ \


# python /home/priyansk/qa/MultiTabQA/train_pretraining.py \
# --output_dir /data/tir/projects/tir7/user_data/priyansk/qa_s2_codet5base_html_v2 \
# --dataset_name multitable_pretraining \
# --pretrained_model_name /data/tir/projects/tir7/user_data/priyansk/qa_tapex_e2_codet5base_html/checkpoint-5000/ \
# --learning_rate 5e-5 \
# --train_batch_size 8 \
# --eval_batch_size 4 \
# --gradient_accumulation_steps 32 \
# --eval_gradient_accumulation 32 \
# --num_train_epochs 30 \
# --num_workers 4 \
# --decoder_max_length 1024 \
# --seed 47 \
# --local_rank -1 \


# python /home/priyansk/qa/MultiTabQA/train_downstream.py \
# --output_dir /data/tir/projects/tir7/user_data/priyansk/qa_gq_codet5base_html \
# --dataset_name geoquery \
# --pretrained_model_name /data/tir/projects/tir7/user_data/priyansk/qa_s2_codet5base_html/checkpoint-3300 \
# --learning_rate 1e-4 \
# --train_batch_size 8 \
# --eval_batch_size 4 \
# --gradient_accumulation_steps 32 \
# --eval_gradient_accumulation 32 \
# --num_train_epochs 60 \
# --num_workers 4 \
# --decoder_max_length 1024 \
# --seed 47 \
# --local_rank -1 \
# --table_type html

python /home/priyansk/qa/MultiTabQA/train_downstream.py \
--output_dir /data/tir/projects/tir7/user_data/priyansk/qa_gq_v2_codet5base_html \
--dataset_name geoquery \
--pretrained_model_name /data/tir/projects/tir7/user_data/priyansk/qa_s2_codet5base_html_v2/checkpoint-3300 \
--learning_rate 1e-4 \
--train_batch_size 8 \
--eval_batch_size 4 \
--gradient_accumulation_steps 32 \
--eval_gradient_accumulation 32 \
--num_train_epochs 60 \
--num_workers 4 \
--decoder_max_length 1024 \
--seed 47 \
--local_rank -1 \
--table_type html