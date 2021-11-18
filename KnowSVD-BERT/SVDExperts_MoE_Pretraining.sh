#!/bin/bash

export PATH=~/.local/bin:$PATH

base_dir=`pwd`

#MASTER_PORT=1231
# Where should we save checkpoints and tensorboard events?
#JOB_NAME=lamb_64k_seq128



JOB_NAME=know_only_svd_moe_pretraining_lr2e_4_bsz256_epoch256
OUTPUT_DIR=/home/users/nus/e0792473/scratch/output_svd_bert/bert_model_outputs
PRETRAINED_MODEL_DIR=/home/users/nus/e0792473/scratch/BERT_base_model
DATA_PREFIX=/home/users/nus/e0792473/scratch/BERT_pretraining_dataset/hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/books_wiki_en_corpus/
LOG_DIR="/home/users/nus/e0792473/scratch/output_svd_bert/log"

# Size of expert parallel world (should be less than total world size)
EP_SIZE=8
# Number of total experts
EXPERTS=8
C=1.2
NUM_EXP_LAYERS=2







mkdir -p $OUTPUT_DIR

NCCL_TREE_THRESHOLD=0 deepspeed ${base_dir}/deepspeed_train.py \
--cf ${base_dir}/bert_base_SVD.json \
--max_seq_length 512 \
--output_dir $OUTPUT_DIR \
--model_name bert-base-uncased \
--deepspeed \
--print_steps 1000 \
--lr_schedule "LL" \
--lr_offset 0.000001 \
--job_name $JOB_NAME \
--deepspeed_config deepspeed_bsz16_lamb_config_seq512.json \
--data_path_prefix ${DATA_PREFIX} \
--validation_data_path_prefix ${DATA_PREFIX} \
--use_nvidia_dataset \
--use_moe \
--use_svd \
--use_rts \
--moe_only \
--ep-world-size ${EP_SIZE} \
--num-experts ${EXPERTS} \
--num_moe_layers ${NUM_EXP_LAYERS} \
--post_moe_layers 1 \
--top-k 1 \
--expert_dropout 0.1 \
--noisy-gate-policy 'Jitter' \
--moe-param-group \
--capacity_factor ${C} \
--use_pretrain \
--pretrained_model_dict ${PRETRAINED_MODEL_DIR}/pytorch_model.bin \
--pretrained_BERT_config ${PRETRAINED_MODEL_DIR}/config.json &> ${LOG_DIR}/${JOB_NAME}.log




#--use_svd \
#--deepspeed_transformer_kernel \
