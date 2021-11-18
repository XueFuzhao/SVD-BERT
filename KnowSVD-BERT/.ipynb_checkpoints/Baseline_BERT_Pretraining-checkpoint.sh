#!/bin/bash

export PATH=~/.local/bin:$PATH

base_dir=`pwd`
#MASTER_PORT=1231
# Where should we save checkpoints and tensorboard events?
JOB_NAME=lamb_64k_seq128
OUTPUT_DIR=/home/users/nus/e0792473/scratch/output_svd_bert/bert_model_outputs
PRETRAINED_MODEL_DIR=/home/users/nus/e0792473/scratch/BERT_base_model
DATA_PREFIX=/home/users/nus/e0792473/scratch/BERT_pretraining_dataset/hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/books_wiki_en_corpus/
LOG_DIR="log"

mkdir -p $OUTPUT_DIR

NCCL_TREE_THRESHOLD=0 deepspeed ${base_dir}/deepspeed_train.py \
--cf ${base_dir}/bert_base_SVD.json \
--max_seq_length 512 \
--output_dir $OUTPUT_DIR \
--model_name bert-base-uncased \
--deepspeed \
--print_steps 100 \
--lr_schedule "EE" \
--lr_offset 10e-4 \
--job_name $JOB_NAME \
--deepspeed_config deepspeed_bsz16_lamb_config_seq512.json \
--data_path_prefix ${DATA_PREFIX} \
--validation_data_path_prefix ${DATA_PREFIX} \
--use_nvidia_dataset \
--use_pretrain \
--pretrained_model_dict ${PRETRAINED_MODEL_DIR}/pytorch_model.bin \
--pretrained_BERT_config ${PRETRAINED_MODEL_DIR}/config.json &> ${LOG_DIR}/${JOB_NAME}.log

#--deepspeed_transformer_kernel \
